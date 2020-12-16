import sys
import os
import ray
from ray import tune

from exp.ghtmn_trainable import GHTMNTrainable
from exp.utils import prepare_dir_tree_experiment, get_split, get_seed, prepare_tree_datasets
from exp.early_stopper import TrialNoImprovementStopper


def get_config(name):
    if name == 'NCI1':
        return {
            'dataset': 'NCI1',
            'out': 1,
            'symbols': 37,
            'depth': tune.randint(2, 9),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * tune.randint(6, 9)),
            'lr': tune.uniform(1e-5, 1e-2),
            'batch_size': tune.choice([64, 100, 128]),
            'tree_dropout': tune.uniform(0, 1)
        }

    if name == 'PROTEINS':
        return {
            'dataset': 'PROTEINS',
            'out': 1,
            'symbols': 3,
            'depth': tune.randint(2, 9),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * tune.randint(6, 9)),
            'lr': tune.uniform(1e-5, 1e-2),
            'batch_size': tune.choice([64, 100, 128]),
            'tree_dropout': tune.uniform(0, 1)
        }

    if name == 'DD':
        return {
            'dataset': 'DD',
            'out': 1,
            'symbols': 89,
            'depth': tune.randint(2, 16),
            'C': tune.randint(2, 11),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * tune.randint(6, 9)),
            'lr': tune.uniform(1e-4, 1e-2),
            'batch_size': tune.choice([32, 64, 100]),
            'tree_dropout': tune.uniform(0, 1)
        }
    
    
if __name__ == '__main__':
    DATASET = sys.argv[1]
    exp_dir = f'GHTMN_{DATASET}'
    if DATASET == 'PROTEINS':
        depths = range(2, 9)
    prepare_dir_tree_experiment(DATASET)
    prepare_tree_datasets(DATASET, depths, 64)

    ray.init(num_gpus=2)
    config = get_config(DATASET)
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=50)
    
    experiments = []
    for i in range(10):
        tr_idx, vl_idx, ts_idx = get_split(exp_dir, i)
        fold_dir = os.path.join(exp_dir, f'fold_{i}')
        config['tr_idx'], config['vl_idx'] = tr_idx, vl_idx
        fold_exp = tune.Experiment(
            f'fold_{i}',
            GHTMNTrainable,
            stop=early_stopping,
            local_dir=exp_dir,
            config=config,
            resources_per_trial= {'cpu': 1, 'gpu': 0.5},
            keep_checkpoints_num=3,
            checkpoint_score_attr='min-vl_loss',
            checkpoint_freq=1,
            max_failures=3
        )
        experiments.append(fold_exp)

    scheduler = tune.schedulers.HyperBandForBOHB(
        time_attr="training_iteration",
        metric="vl_loss",
        mode="min",
        max_t=200,
        reduction_factor=2
    )

    search_alg = tune.suggest.bohb.TuneBOHB(
        max_concurrent=4, 
        metric='vl_loss', 
        mode="min",
        seed=get_seed()
    )

    tune.run(experiments,
             search_alg=search_alg,
             scheduler=scheduler,
             reuse_actors=True)


