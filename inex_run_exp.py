import argparse
import os
import math
from random import randint

import numpy as np
import torch
import ray
from ray import tune

from exp.gpu_trainable import GPUTrainable
from exp.utils import get_best_info, get_score_fn, get_loss_fn, get_seed
from exp.run import run_exp, run_test

from torch.utils.data import DataLoader
from data.tree.utils import trees_collate_fn, TreeDataset
from sklearn.model_selection import train_test_split
from htmn.htmn import HTMN

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument(['--gpus', '-g'], type=int, nargs='*', default=[])
parser.add_argument(['--workers', '-w'], type=int, default=36)
parser.add_argument(['--design', '-d'], action='store_true')
parser.add_argument(['--retrain', '-r'], action='store_true')
parser.add_argument(['--test', '-t'], default='design')

def get_config(name):
    if name == 'inex2005':
        return {
            'model': 'htmn',
            'out': 11,
            'M': 366,
            'L': 32,
            'C': tune.randint(7, 14),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(5, 9)),
            'lr': tune.uniform(5e-3, 1e-2),
            'min_lr': 1e-3,
            'epochs_decay': 80,
            'batch_size': tune.choice([32, 64, 128, 192, 256]),
            'loss': 'ce',
            'score': 'accuracy'
        }

    if name == 'inex2006':
        return {
            'model': 'htmn',
            'out': 18,
            'M': 65,
            'L': 66,
            'C': tune.randint(6, 14),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 9)),
            'lr': tune.uniform(3e-4, 3e-3),
            'min_lr': 5e-4,
            'epochs_decay': 80,
            'batch_size': tune.choice([64, 128, 192, 256]),
            'loss': 'ce',
            'score': 'accuracy'
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset, gpus, workers = args.dataset, args.gpus, args.workers 
    exp_dir = f'HTMN_exp/inex{dataset}'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(f'inex{dataset}')
    config['dataset'] = f'inex{dataset}train'
    dataset = TreeDataset('.', name=f'inex{dataset}train')
    tr_idx, vl_idx = train_test_split(np.arange(len(dataset)), 
                                      test_size=0.2,  
                                      stratify=np.array([t.y for t in dataset]), 
                                      shuffle=True, 
                                      random_state=get_seed())
    config['tr_idx'], config['vl_idx'] = tr_idx.tolist(), vl_idx.tolist()
    if args.design:
        run_exp(
            'design',
            config=config,
            n_samples=300,
            p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 50},
            p_scheduler={'metric': 'vl_loss', 'mode': 'min', 'max_t': 400, 'grace': 50, 'reduction': 2},
            exp_dir=exp_dir,
            chk_score_attr='vl_score',
            log_params={'n_gen': '#gen', 'C': 'C', 'lr': 'LRate', 'batch_size': 'Batch'},
            gpus=gpus,
            gpu_threshold=0.8
        )
    
    # Retraining phase
    best_dict = get_best_info(os.path.join(exp_dir, 'design'))
    t_config = best_dict['config']
    t_config['tr_idx'], t_config['vl_idx'] = list(range(len(dataset))), None
    t_config['out'] = tune.choice([t_config['out']])
    if args.retrain:
        run_exp(
            'test',
            config=t_config,
            n_samples=5,
            p_early={'metric': 'tr_loss', 'mode': 'min', 'patience': 50},
            p_scheduler=None,
            exp_dir=exp_dir,
            chk_score_attr='tr_score',
            log_params={'n_gen': '#gen', 'C': 'C', 'lr': 'LRate', 'batch_size': 'Batch'},
            gpus=gpus,
            gpu_threshold=0.8
        )
    
    # Test phase
    ts_data = TreeDataset('.', f'inex{dataset}test')
    ts_ld = DataLoader(ts_data, 
                       collate_fn=trees_collate_fn, 
                       batch_size=512, 
                       shuffle=False)

    if args.test == 'retrain':
        ts_loss = []
        ts_acc = []
        test_dir = os.path.join(exp_dir, 'test')
        for i, t_dir in enumerate(os.listdir(test_dir)):
            trial_dir = os.path.join(test_dir, t_dir)
            v_loss, v_acc = run_test(
                trial_dir=trial_dir,
                ts_ld=ts_ld,
                model_func=lambda config: HTMN(config['out'], 
                                            math.ceil(config['n_gen']/2), 
                                            math.floor(config['n_gen']/2), 
                                            config['C'], 
                                            config['L'], 
                                            config['M']),
                loss_fn=get_loss_fn('ce'),
                score_fn=get_score_fn('accuracy', t_config['out']),
                gpus=gpus
            )
            ts_loss.append(v_loss)
            ts_acc.append(v_acc)
        
        best_dict = get_best_info(os.path.join(exp_dir, 'design'))
        best_dict['ts_loss'], best_dict['ts_score'] = ts_loss, ts_acc
        torch.save(best_dict, os.path.join(exp_dir, 'test_res.pkl'))

    if args.test == 'design':
        v_loss, v_acc = run_test(
            trial_dir=best_dict['trial_dir'],
            ts_ld=ts_ld,
            model_func=lambda config: HTMN(config['out'], 
                                           math.ceil(config['n_gen']/2), 
                                           math.floor(config['n_gen']/2), 
                                           config['C'], 
                                           config['L'], 
                                           config['M']),
            loss_fn=get_loss_fn('ce'),
            score_fn=get_score_fn('accuracy', t_config['out']),
            gpus=gpus
        )
        ts_loss.append(v_loss)
        ts_acc.append(v_acc)
        
        best_dict = get_best_info(os.path.join(exp_dir, 'design'))
        best_dict['ts_loss'], best_dict['ts_score'] = ts_loss, ts_acc
        torch.save(best_dict, os.path.join(exp_dir, 'test_res.pkl'))
