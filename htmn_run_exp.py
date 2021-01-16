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
            'C': tune.randint(6, 14),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 9)),
            'lr': tune.uniform(5e-5, 2e-2),
            'batch_size': tune.choice([32, 64, 128, 192])
        }

    if name == 'inex2006':
        return {
            'dataset': 'inex2006',
            'out': 18,
            'M': 65,
            'L': 66,
            'C': tune.randint(6, 14),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 9)),
            'lr': tune.uniform(5e-5, 2e-2),
            'batch_size': tune.choice([64, 128, 192, 256])
        }

if __name__ == '__main__':
    DATASET = sys.argv[1]
    N_CPUS = int(sys.argv[2])
    exp_dir = 'HTMN_{}'.format(DATASET)
    
    ray.init(num_cpus=N_CPUS)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config = get_config(DATASET)
    config['wdir'] = os.getcwd()
    config['gpu_ids'] = [int(i) for i in sys.argv[3].split(',')]
    config['holdout'] = 0.2
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=40)
    scheduler = ASHAScheduler(
        metric='vl_loss',
        mode='min',
        max_t=400,
        grace_period=40,
        reduction_factor=4
    )
    
    cpus_per_task = 1
    if torch.cuda.is_available():
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        gpus_per_task = n_gpus / (N_CPUS / cpus_per_task)
        resources = {'cpu': cpus_per_task, 'gpu': gpus_per_task}
    else:
        resources = {'cpu': cpus_per_task, 'gpu': 0}
    n_samples = 400
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter', 
                                    'vl_loss': 'Loss', 
                                    'vl_acc': 'Acc.', 
                                    'best_acc': 'Best Acc.',
                                },
                                parameter_columns={
                                    'n_gen': '#gen', 
                                    'C': 'C', 
                                    'lr': 'LRate',
                                    'batch_size': 'Batch'
                                }, 
                                infer_limit=3,
                                metric='best_acc',
                                mode='max')
    tune.run(
        HTMNTrainable,
        stop=early_stopping,
        local_dir=exp_dir,
        config=config,
        num_samples=n_samples,
        resources_per_trial= resources,
        keep_checkpoints_num=1,
        checkpoint_score_attr='vl_acc',
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        scheduler=scheduler,
        verbose=1
    )
