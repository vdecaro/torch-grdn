import sys
import os
from random import randint, randrange

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from exp.htmn_trainable import HTMNTrainable
from exp.utils import prepare_dir_tree_experiments, prepare_tree_datasets
from exp.early_stopper import TrialNoImprovementStopper


def get_config(name):
    if name == 'inex2005':
        return {
            'dataset': 'inex2005',
            'out': 11,
            'M': 366,
            'L': 32,
            'C': tune.randint(6, 11),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-4, 2e-2),
            'batch_size': tune.choice([32, 64, 128])
        }

    if name == 'inex2006':
        return {
            'dataset': 'inex2006',
            'out': 18,
            'M': 65,
            'L': 66,
            'C': tune.randint(6, 11),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(5e-4, 2e-2),
            'batch_size': tune.choice([32, 64, 128])
        }

if __name__ == '__main__':
    DATASET = sys.argv[1]
    N_CPUS = int(sys.argv[2])
    exp_dir = 'HTMN_{}'.format(DATASET)
    
    ray.init(num_cpus=N_CPUS)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(DATASET)
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=15)
    scheduler = ASHAScheduler(
        metric='vl_loss',
        mode='min',
        max_t=400,
        grace_period=15,
        reduction_factor=2
    )
    
    cpus_per_task = 1
    if torch.cuda.is_available():
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        gpus_per_task = n_gpus / (N_CPUS / cpus_per_task)
        resources = {'cpu': cpus_per_task, 'gpu': gpus_per_task}
    else:
        resources = {'cpu': cpus_per_task, 'gpu': 0}
    
    tune.run(
        HTMNTrainable,
        stop=early_stopping,
        local_dir=exp_dir,
        config=config,
        num_samples=50,
        resources_per_trial= resources,
        keep_checkpoints_num=1,
        checkpoint_score_attr='vl_acc',
        checkpoint_freq=1,
        max_failures=5,
        reuse_actors=False,
        scheduler=scheduler,
        verbose=1
    )
