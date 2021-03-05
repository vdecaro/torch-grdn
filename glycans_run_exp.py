import argparse
import os
import math
from random import randint, randrange, choice
from collections import defaultdict
import json

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
            'out': 1,
            'M': 29,
            'L': 3,
            'C': tune.grid_search([2, 4, 8]),
            'n_gen': tune.grid_search(list(range(10, 31, 10))),
            'lr': 1e-3,
            #'l2': tune.grid_search([1e-2]),
            'batch_size': tune.grid_search([4, 8, 16, 32]),
            'loss': 'bce',
            'score': 'roc-auc',
            'rank': 'raw'
        }

    if name == 'leukemia':
        return {
            'model': 'htmn',
            'dataset': 'leukemia',
            'out': 1,
            'M': 57,
            'L': 3,
            'C': tune.grid_search([2, 4, 8]),
            'n_gen': tune.grid_search(list(range(10, 31, 10))),
            'lr': 1e-3,
            #'l2': 0.01,
            'batch_size': tune.grid_search([8, 16, 32, 48]),
            'loss': 'bce',
            'score': 'roc-auc',
            'rank': 'raw'
        }

    
if __name__ == '__main__':
    args = parser.parse_args()
    ds_name, gpus, workers = args.dataset, args.gpus, args.workers 
    exp_dir = f'HTMN_exp/{ds_name}'
    ray.init(num_cpus=workers)
    
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
    zeros, ones = 0, 0
    for d in dataset:
        if d.y[0] == 0:
            zeros += 1
        else:
            ones += 1
    print(zeros, ones)
    for fold_idx, (ds_i, ts_i) in enumerate(folds):
        if fold_idx in args.design:
            ds_data = TreeDataset(data=[dataset[curr] for curr in ds_i])
            tr_i, vl_i = train_test_split(np.array(ds_i), 
                                          test_size=0.33,  
                                          stratify=np.array([g.y for g in ds_data]), 
                                          shuffle=True, 
                                          random_state=get_seed())
            config['tr_idx'], config['vl_idx'] = tr_i.tolist(), vl_i.tolist()
            labs = [dataset[curr].y[0] for curr in vl_i.tolist()]
            print(labs)
            run_exp(
                f'fold_{fold_idx}',
                config=config,
                n_samples=1,
                p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 20},
                p_scheduler=None,
                exp_dir=exp_dir,
                chk_score_attr='min-vl_loss',
                log_params={'n_gen': '#gen', 'C': 'C', 'lr': 'LRate', 'l2': 'L2', 'batch_size': 'Batch'},
                gpus=gpus,
                gpu_threshold=0.9
            )
            
        if fold_idx in args.test:
            best_dict = get_best_info(os.path.join(exp_dir, f'fold_{fold_idx}'), mode='manual')
            t_config = best_dict['config']
            
            ts_ld = DataLoader(TreeDataset(data=[dataset[curr] for curr in ts_i]), 
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
                loss_fn=get_loss_fn('bce', None),
                score_fn=get_score_fn('roc-auc', t_config['out']),
                gpus=[]
            )
            print(best_dict)
            torch.save(best_dict, os.path.join(exp_dir, f'fold_{fold_idx}', 'test_res.pkl'))