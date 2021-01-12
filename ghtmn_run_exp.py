import sys
import os
from random import randint, randrange

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from exp.ghtmn_trainable import GHTMNTrainable
from exp.utils import prepare_dir_tree_experiments, get_split, get_seed, prepare_tree_datasets
from exp.early_stopper import TrialNoImprovementStopper


def get_config(name):
    if name == 'NCI1':
        return {
            'dataset': 'NCI1',
            'out': 1,
            'symbols': 37,
            'depth': tune.randint(2, 10),
            'C': tune.randint(2, 8),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-4, 2e-3),
            'batch_size': 100,
            'tree_dropout': tune.uniform(0.4, 0.9)
        }

    if name == 'PROTEINS':
        return {
            'dataset': 'PROTEINS',
            'out': 1,
            'symbols': 3,
            'depth': tune.randint(2, 8),
            'C': tune.randint(2, 8),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-5, 2e-4),
            'batch_size': 100,
            'tree_dropout': tune.uniform(0.15, 0.9)
        }

    if name == 'DD':
        return {
            'dataset': 'DD',
            'out': 1,
            'symbols': 89,
            'depth': tune.randint(2, 10),
            'C': tune.randint(2, 12),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(1e-5, 1e-3),
            'batch_size': 100,
            'tree_dropout': tune.uniform(0.5, 0.9)
        }

    
if __name__ == '__main__':
    DATASET = sys.argv[1]
    N_CPUS = int(sys.argv[2])
    exp_dir = 'GHTMN_{}'.format(DATASET)
    
    ray.init(num_cpus=N_CPUS, dashboard_host='0.0.0.0')
    if not os.path.exists(exp_dir):
        prepare_dir_tree_experiments(DATASET)
    if DATASET == 'PROTEINS':
        depths = range(2, 9)
    if DATASET == 'DD':
        depths = range(2, 13)
    if DATASET == 'NCI1':
        depths = range(2, 11)
    prepare_tree_datasets(DATASET, depths, N_CPUS)
    
    config = get_config(DATASET)
    config['gpu_ids'] = [int(i) for i in sys.argv[3].split(',')]
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=40)
    scheduler = ASHAScheduler(
        metric='vl_loss',
        mode='min',
        max_t=400,
        grace_period=20,
        reduction_factor=2
    )
    
    cpus_per_task = 2
    if torch.cuda.is_available():
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        gpus_per_task = n_gpus / (N_CPUS / cpus_per_task)
        resources = {'cpu': cpus_per_task, 'gpu': gpus_per_task}
    else:
        resources = {'cpu': cpus_per_task, 'gpu': 0}
    
    for i in range(10):
        config['fold'] = i
        name = 'fold_{}'.format(i)
        size_dir = len(os.listdir(os.path.join(exp_dir, name)))
        if size_dir < 204:
            fold_exp = tune.run(
                GHTMNTrainable,
                name=name,
                stop=early_stopping,
                local_dir=exp_dir,
                config=config,
                num_samples=200,
                resources_per_trial= resources,
                keep_checkpoints_num=1,
                checkpoint_score_attr='vl_acc',
                checkpoint_freq=1,
                max_failures=5,
                reuse_actors=False,
                scheduler=scheduler,
                verbose=1,
                resume=size_dir > 3
            )
