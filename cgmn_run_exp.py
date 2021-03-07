import argparse
import os
import math
from random import randint, randrange, choice

import torch
import numpy as np
import ray
from ray import tune

from cgmn.cgmn import CGMN

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from exp.utils import get_seed, get_best_info, get_loss_fn, get_score_fn
from exp.run import run_exp, run_test

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--gpus', '-g', type=int, nargs='*', default=[])
parser.add_argument('--workers', '-w', type=int, default=30)
parser.add_argument('--design', '-d', type=int, nargs='*', default=list(range(10)))
parser.add_argument('--test', '-t', type=int, nargs='*', default=list(range(10)))

def get_config(name):
    if name == 'NCI1':
        return {
            'model': 'cgmn',
            'dataset': 'NCI1',
            'out': 1,
            'symbols': 37,
            'depth': tune.grid_search([i for i in range(5, 21, 5)]),
            'C': tune.grid_search([i for i in range(4, 21, 4)]),
            'n_gen': tune.grid_search([i for i in range(25, 46, 5)]),
            'lr': 0.01,
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    if name == 'PROTEINS':
        return {
            'model': 'cgmn',
            'dataset': 'PROTEINS',
            'out': 1,
            'symbols': 3,
            'depth': tune.grid_search([i for i in range(5, 21, 5)]),
            'C': tune.grid_search([i for i in range(4, 21, 4)]),
            'n_gen': tune.grid_search([i for i in range(25, 46, 5)]),
            'lr': 0.0005,
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    if name == 'DD':
        return {
            'model': 'cgmn',
            'dataset': 'DD',
            'out': 1,
            'symbols': 89,
            'depth': tune.grid_search([i for i in range(5, 21, 5)]),
            'C': tune.grid_search([i for i in range(4, 21, 4)]),
            'n_gen': tune.grid_search([i for i in range(15, 61, 5)]),
            'lr': 0.001,
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    
if __name__ == '__main__':
    args = parser.parse_args()
    ds_name, gpus, workers = args.dataset, args.gpus, args.workers 
    exp_dir = f'CGMN_exp/{ds_name}'
    ray.init(num_cpus=workers)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(ds_name)
    config['gpu_ids'] = gpus
    
    def tf_func(data):
            data.y = data.y.unsqueeze(1)
            return data
    dataset = TUDataset('.', ds_name, transform=tf_func)
    
    dataset.data.x = dataset.data.x.argmax(1).detach()
    ext_kfold = StratifiedKFold(10, shuffle=True, random_state=get_seed())
    ext_split = list(ext_kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))
    for fold_idx, (ds_i, ts_i) in enumerate(ext_split):
        fold_dir = os.path.join(exp_dir, f'fold_{fold_idx}')

        if fold_idx in args.design:
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            ds_data = dataset[ds_i.tolist()]
            tr_i, vl_i = train_test_split(ds_i, 
                                        test_size=0.2,  
                                        stratify=np.array([g.y for g in ds_data]), 
                                        shuffle=True, 
                                        random_state=get_seed())
            config['tr_idx'], config['vl_idx'] = tr_i.tolist(), vl_i.tolist()
            run_exp(
                'design',
                config=config,
                n_samples=1,
                p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 50},
                p_scheduler={'metric': 'vl_loss', 'mode': 'min', 'max_t': 400, 'grace': 50, 'reduction': 4},
                exp_dir=fold_dir,
                chk_score_attr='rank_score',
                log_params={'n_gen': '#gen', 'C': 'C', 'depth': 'Depth'},
                gpus=gpus,
                gpu_threshold=0.75
            )

        if fold_idx in args.test:
            best_dict = get_best_info(os.path.join(fold_dir, 'design'), mode='manual')
            t_config = best_dict['config']
            ts_ld = DataLoader(dataset[ts_i.tolist()], 
                               batch_size=512, 
                               shuffle=False)

            best_dict['ts_loss'], best_dict['ts_score'] = run_test(
                trial_dir=best_dict['trial_dir'],
                ts_ld=ts_ld,
                model_func=lambda config: CGMN(config['out'], config['n_gen'], config['C'], config['symbols'], config['depth']),
                loss_fn=get_loss_fn('bce'),
                score_fn=get_score_fn('accuracy', t_config['out']),
                gpus=[]
            )
            print(best_dict)
            torch.save(best_dict, os.path.join(fold_dir, 'test_res.pkl'))
