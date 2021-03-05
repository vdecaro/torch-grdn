import argparse
import os
import math
from random import randint, randrange, choice

import torch
import numpy as np
import ray
from ray import tune

from graph_htmn.graph_htmn import GraphHTMN
from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from exp.utils import get_seed, get_best_info, get_loss_fn, get_score_fn
from exp.run import run_exp, run_test

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--gpus', '-g', type=int, nargs='*', default=[])
parser.add_argument('--workers', '-w', type=int, default=36)
parser.add_argument('--design', '-d', type=int, nargs='*', default=list(range(10)))
parser.add_argument('--test', '-t', type=int, nargs='*', default=list(range(10)))

def get_config(name):
    if name == 'NCI1':
        return {
            'model': 'ghtmn',
            'dataset': 'NCI1',
            'out': 1,
            'symbols': 37,
            'depth': tune.randint(2, 10),
            'C': tune.randint(2, 8),
            'gen_mode': tune.choice(['bu', 'td', 'both']),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-4, 1e-3),
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    if name == 'PROTEINS':
        return {
            'model': 'ghtmn',
            'dataset': 'PROTEINS',
            'out': 1,
            'symbols': 3,
            'depth': tune.randint(2, 8),
            'C': tune.randint(2, 8),
            'gen_mode': tune.choice(['bu', 'td', 'both']),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-4, 1e-3),
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    if name == 'DD':
        return {
            'model': 'ghtmn',
            'dataset': 'DD',
            'out': 1,
            'symbols': 89,
            'depth': tune.randint(2, 10),
            'C': tune.randint(2, 12),
            'gen_mode': tune.choice(['bu', 'td', 'both']),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(1e-5, 1e-3),
            'batch_size': 100,
            'loss': 'bce',
            'score': 'accuracy',
            'rank': 'raw'
        }

    
if __name__ == '__main__':
    args = parser.parse_args()
    ds_name, gpus, workers = args.dataset, args.gpus, args.workers 
    exp_dir = f'GHTMN_exp/{ds_name}'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(ds_name)
    config['gpu_ids'] = gpus
    
    dataset = TUDataset('.', ds_name)
    ext_kfold = StratifiedKFold(10, shuffle=True, random_state=get_seed())
    ext_split = list(ext_kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))
    for fold_idx, (ds_i, ts_i) in enumerate(ext_split):
        fold_dir = os.path.join(exp_dir, f'fold_{fold_idx}')

        if fold_idx in args.design:
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            ds_data = dataset[ds_i.tolist()]
            tr_i, vl_i = train_test_split(ds_i, 
                                        test_size=0.175,  
                                        stratify=np.array([g.y for g in ds_data]), 
                                        shuffle=True, 
                                        random_state=get_seed())
            config['tr_idx'], config['vl_idx'] = tr_i.tolist(), vl_i.tolist()
            ray.init(num_cpus=workers*2)
            run_exp(
                'design',
                config=config,
                n_samples=300,
                p_early={'metric': 'vl_loss', 'mode': 'min', 'patience': 50},
                p_scheduler={'metric': 'vl_loss', 'mode': 'min', 'max_t': 400, 'grace': 50, 'reduction': 4},
                exp_dir=fold_dir,
                chk_score_attr='rank_score',
                log_params={'n_gen': '#gen', 'C': 'C', 'depth': 'Depth', 'lr': 'LRate', 'batch_size': 'Batch'},
                gpus=gpus,
                gpu_threshold=0.75
            )
            ray.shutdown(True)

        if fold_idx in args.test:
            best_dict = get_best_info(os.path.join(fold_dir, 'design'), mode='manual')
            t_config = best_dict['config']
            ts_data = ParallelTUDataset(os.path.join(ds_name, f'D{t_config["depth"]}'),
                                        ds_name, 
                                        pre_transform=pre_transform(t_config['depth']),
                                        transform=transform(ds_name))
            ts_data.data.x = dataset.data.x.argmax(1).detach()
            ts_ld = DataLoader(ts_data[ts_i.tolist()], 
                               collate_fn=TreeCollater(t_config['depth']), 
                               batch_size=512, 
                               shuffle=False)

            def _get_model(config):
                if config['gen_mode'] == 'bu':
                    n_bu, n_td = t_config['n_gen'], 0
                elif config['gen_mode'] == 'td':
                    n_bu, n_td = 0, t_config['n_gen']
                elif config['gen_mode'] == 'both':
                    n_bu, n_td = math.ceil(t_config['n_gen']/2), math.floor(t_config['n_gen']/2)

                return GraphHTMN(config['out'], n_bu, n_td, t_config['C'], t_config['symbols'])

            best_dict['ts_loss'], best_dict['ts_score'] = run_test(
                trial_dir=best_dict['trial_dir'],
                ts_ld=ts_ld,
                model_func=_get_model,
                loss_fn=get_loss_fn('bce'),
                score_fn=get_score_fn('accuracy', t_config['out']),
                gpus=[]
            )

            torch.save(best_dict, os.path.join(fold_dir, 'test_res.pkl'))
