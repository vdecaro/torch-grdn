import sys
import os
from random import randint

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
            'depth': tune.randint(2, 8),
            'n_gen': tune.randint(6, 9),
            'lr': tune.uniform(1e-5, 1e-2),
            'batch_size': tune.choice([64, 100, 128]),
            'tree_dropout': tune.uniform(0, 0.8)
        }

    if name == 'PROTEINS':
        return {
            'dataset': 'PROTEINS',
            'out': 1,
            'symbols': 3,
            'depth': tune.randint(2, 8),
            'C': tune.randint(2, 8),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(1e-5, 1e-3),
            'batch_size': 100,
            'tree_dropout': tune.uniform(0, 0.8)
        }

    if name == 'DD':
        return {
            'dataset': 'DD',
            'out': 1,
            'symbols': 89,
            'depth': tune.randint(2, 12),
            'C': tune.randint(2, 11),
            'n_gen': tune.randint(6, 9),
            'lr': tune.uniform(1e-4, 1e-2),
            'batch_size': tune.choice([32, 64, 100]),
            'tree_dropout': tune.uniform(0, 0.8)
        }

    
if __name__ == '__main__':
    DATASET = sys.argv[1]
    exp_dir = f'GHTMN_{DATASET}'
    
    num_cpus = 72
    cpus_per_task = 4
    ray.init(num_cpus=num_cpus)
    if not os.path.exists(exp_dir):
        prepare_dir_tree_experiments(DATASET)
    if DATASET == 'PROTEINS':
        depths = range(2, 9)
    prepare_tree_datasets(DATASET, depths)

    config = get_config(DATASET)
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=50)
    scheduler = ASHAScheduler(
        metric="vl_loss",
        mode="min",
        max_t=400,
        grace_period=20,
        reduction_factor=2
    )
    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    gpus_per_task = n_gpus / (num_cpus / cpus_per_task)
    for i in range(10):
        config['fold'] = i
        name = f'fold_{i}'
        size_dir = len(os.listdir(os.path.join(exp_dir, name)))
        if size_dir < 204:
            fold_exp = tune.run(
                GHTMNTrainable,
                name=name,
                stop=early_stopping,
                local_dir=exp_dir,
                config=config,
                num_samples=200,
                resources_per_trial= {'cpu': cpus_per_task, 'gpu': gpus_per_task},
                keep_checkpoints_num=1,
                checkpoint_score_attr='min-vl_loss',
                checkpoint_freq=1,
                max_failures=3,
                reuse_actors=True,
                scheduler=scheduler,
                verbose=1,
                resume=size_dir > 3
            )
