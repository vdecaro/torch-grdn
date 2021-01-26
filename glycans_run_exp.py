import sys
import os
import math
from random import randint, randrange, choice

import numpy as np
import torch
import ray
from ray import tune

from exp.run import run_exp, run_test
from exp.utils import get_seed, get_best_info, get_score_fn

from data.tree.utils import TreeDataset,trees_collate_fn
from sklearn.model_selection import train_test_split
from htmn.htmn import HTMN
from torch.utils.data import DataLoader

def get_config(name):
    if name == 'cystic':
        return {
            'model': 'htmn',
            'dataset': 'cystic',
            'out': 1,
            'M': 29,
            'L': 3,
            'C': tune.randint(2, 10),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(5, 9)),
            'lr': tune.uniform(5e-5, 5e-3),
            'batch_size': tune.choice([4, 8, 16, 32, 64]),
            'loss': 'bce',
            'score': 'roc-auc'
        }

    if name == 'leukemia':
        return {
            'model': 'htmn',
            'dataset': 'leukemia',
            'out': 1,
            'M': 57,
            'L': 3,
            'C': tune.randint(2, 10),
            'n_gen': tune.qrandint(5, 80, 5),
            'lr': tune.uniform(5e-5, 5e-3),
            'batch_size': tune.choice([8, 16, 32, 48, 64, 128]),
            'loss': 'bce',
            'score': 'roc-auc'
        }

    
if __name__ == '__main__':
    dataset = sys.argv[1]
    exp_dir = f'HTMN_exp/{dataset}'
    ray.init(num_cpus=int(sys.argv[2])*2)
    gpus = [int(i) for i in sys.argv[3].split(',')] if len(sys.argv) == 4 else []
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(dataset)

    folds = []
    with open(f'data/tree/glycans/{dataset}/fold_idx') as f:
        for l in f:
            ds_line, ts_line = l.split(':')
            ds_idx = [int(i) for i in ds_line.split(',')]
            ts_idx = [int(i) for i in ts_line.split(',')]
            folds.append((ds_idx, ts_idx))

    dataset = TreeDataset('.', name=dataset)
    for fold_idx, (ds_idx, ts_idx) in enumerate(folds):
        fold_dir = os.path.join(exp_dir, f'fold_{fold_idx}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        ds_data = TreeDataset(data=[dataset[i] for i in ds_idx])
        tr_idx, vl_idx = train_test_split(np.array(ds_idx), 
                                          test_size=0.2,  
                                          stratify=np.array([t.y for t in ds_data]), 
                                          shuffle=True, 
                                          random_state=get_seed())
        config['tr_idx'], config['vl_idx'] = tr_idx.tolist(), vl_idx.tolist()
        run_exp(
            'design',
            config=config,
            n_samples=300,
            p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 50},
            p_scheduler={'metric': 'vl_loss', 'mode': 'min', 'max_t': 400, 'grace': 50, 'reduction': 2},
            exp_dir=fold_dir,
            chk_score_attr='vl_score',
            log_params={'n_gen': '#gen', 'C': 'C', 'lr': 'LRate', 'batch_size': 'Batch'},
            gpus=gpus
        )

        best_dict = get_best_info(fold_dir)
        t_config = best_dict['config']
        ts_data = TreeDataset('.', name=dataset)
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
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            score_fn=get_score_fn('roc-auc'),
            gpus=gpus
        )

        torch.save(best_dict, os.path.join(fold_dir, 'test_res.pkl'))
