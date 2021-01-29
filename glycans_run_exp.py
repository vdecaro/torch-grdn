import argparse
import os
import math
from random import randint, randrange, choice

import numpy as np
import torch
import ray
from ray import tune

from exp.run import run_exp, run_test
from exp.utils import get_seed, get_best_info, get_score_fn, get_loss_fn

from data.tree.utils import TreeDataset,trees_collate_fn
from sklearn.model_selection import train_test_split
from htmn.htmn import HTMN
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--gpus', '-g', type=int, nargs='*', default=[])
parser.add_argument('--workers', '-w', type=int, default=36)
parser.add_argument('--design', '-d', type=int, nargs='*', default=list(range(10)))
parser.add_argument('--test', '-t', type=int, nargs='*', default=list(range(10)))

def get_config(name):
    if name == 'cystic':
        return {
            'model': 'htmn',
            'dataset': 'cystic',
            'holdout': 0.4,
            'out': 1,
            'M': 29,
            'L': 3,
            'C': tune.grid_search([2, 4, 8]),
            'n_gen': tune.grid_search(list(range(10, 31, 5))),
            'lr': tune.grid_search([7.5e-4, 1e-3, 2.5e-3]),
            'batch_size': tune.grid_search([4, 8, 16, 32]),
            'loss': 'bce',
            'score': 'roc-auc'
        }

    if name == 'leukemia':
        return {
            'model': 'htmn',
            'dataset': 'leukemia',
            'holdout': 0.4,
            'out': 1,
            'M': 57,
            'L': 3,
            'C': tune.grid_search([2, 4, 8]),
            'n_gen': tune.grid_search(list(range(10, 31, 5))),
            'lr': tune.grid_search([7.5e-4, 1e-3, 2.5e-3]),
            'batch_size': tune.grid_search([8, 16, 32, 48, 64]),
            'loss': 'bce',
            'score': 'roc-auc'
        }

    
if __name__ == '__main__':
    args = parser.parse_args()
    ds_name, gpus, workers = args.dataset, args.gpus, args.workers 
    exp_dir = f'HTMN_exp/{ds_name}'
    ray.init(num_cpus=workers*2)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(ds_name)

    folds = []
    with open(f'data/tree/glycans/{ds_name}/fold_idx') as f:
        for l in f:
            ds_line, ts_line = l.split(':')
            ds_idx = [int(i) for i in ds_line.split(',')]
            ts_idx = [int(i) for i in ts_line.split(',')]
            folds.append((ds_idx, ts_idx))

    dataset = TreeDataset('.', name=ds_name)
    for fold_idx, (ds_idx, ts_idx) in enumerate(folds):
        fold_dir = os.path.join(exp_dir, f'fold_{fold_idx}')
        if fold_idx in args.design:
            print(f"Design {fold_idx}")
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            ds_data = TreeDataset(data=[dataset[i] for i in ds_idx])
            tr_idx, vl_idx = train_test_split(np.array(ds_idx), 
                                              test_size=config['holdout'],  
                                              stratify=np.array([t.y for t in ds_data]), 
                                              shuffle=True, 
                                              random_state=get_seed())
            config['tr_idx'], config['vl_idx'] = tr_idx.tolist(), vl_idx.tolist()
            run_exp(
                'design',
                config=config,
                n_samples=1,
                p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 30},
                p_scheduler={'metric': 'vl_loss', 'mode': 'min', 'max_t': 1300, 'grace': 30, 'reduction': 2},
                exp_dir=fold_dir,
                chk_score_attr='vl_score',
                log_params={'n_gen': '#gen', 'C': 'C', 'lr': 'LRate', 'batch_size': 'Batch'},
                gpus=gpus,
                gpu_threshold=0.9
            )
            
        if fold_idx in args.test:
            best_dict = get_best_info(os.path.join(fold_dir, 'design'), mode='manual')
            t_config = best_dict['config']
            ts_data = TreeDataset(data=[dataset[i] for i in ts_idx])
            ts_ld = DataLoader(ts_data, 
                               collate_fn=trees_collate_fn, 
                               batch_size=512, 
                               shuffle=False)
            best_dict['ts_loss'], best_dict['ts_score'] = run_test(
                trial_dir=best_dict['trial_dir'],
                ts_ld=ts_ld,
                model_func=lambda config: HTMN(config['out'], 
                                            math.ceil(config['n_gen']/2), 
                                            math.floor(config['n_gen']/2), 
                                            config['C'], 
                                            config['L'], 
                                            config['M']),
                loss_fn=get_loss_fn('bce'),
                score_fn=get_score_fn('roc-auc', t_config['out']),
                gpus=gpus
            )
            del t_config['tr_idx'], t_config['vl_idx']
            print(best_dict)

            torch.save(best_dict, os.path.join(fold_dir, 'test_res.pkl'))
