import sys
import os
import math
from random import randint, randrange, choice

import torch

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from exp.ghtmn_trainable import GHTMNTrainable
from exp.utils import prepare_dir_tree_experiments, get_split, get_seed, prepare_tree_datasets, get_best_info
from exp.early_stopper import TrialNoImprovementStopper

from graph_htmn.graph_htmn import GraphHTMN
from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader
from torch_geometric.utils.metric import accuracy

def get_config(name):
    if name == 'NCI1':
        return {
            'dataset': 'NCI1',
            'out': 1,
            'symbols': 37,
            'depth': tune.randint(2, 10),
            'C': tune.randint(2, 8),
            'gen_mode': tune.choice(['bu', 'td', 'both']),
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
            'gen_mode': tune.choice(['bu', 'td', 'both']),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(1e-5, 2e-4),
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
            'gen_mode': tune.choice(['bu', 'td', 'both']),
            'n_gen': tune.sample_from(lambda spec: spec.config.C * randint(6, 8)),
            'lr': tune.uniform(1e-5, 1e-3),
            'batch_size': 100,
            'tree_dropout': tune.uniform(0.5, 0.9)
        }

    
if __name__ == '__main__':
    DATASET = sys.argv[1]
    N_CPUS = int(sys.argv[2])
    exp_dir = 'GHTMN_{}'.format(DATASET)

    ray.init(num_cpus=N_CPUS)
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
    config['wdir'] = os.getcwd()
    config['gpu_ids'] = [int(i) for i in sys.argv[3].split(',')]
    early_stopping = TrialNoImprovementStopper('vl_loss', mode='min', patience_threshold=40)
    scheduler = ASHAScheduler(
        metric='vl_loss',
        mode='min',
        max_t=400,
        grace_period=40,
        reduction_factor=4
    )
    
    resources = {'cpu': 2, 'gpu': 0.001}
    
    n_samples = 800
    for i in range(10):
        config['fold'] = i
        name = 'fold_{}'.format(i)
        fold_dir = os.path.join(exp_dir, name)
        size_dir = len(os.listdir(fold_dir))
        if size_dir < n_samples + 4:
            
            # Hyperparameter Tuning Phase
            reporter = tune.CLIReporter(metric_columns={
                                            'training_iteration': 'Iter', 
                                            'vl_loss': 'Loss', 
                                            'vl_acc': 'Acc.', 
                                            'best_acc': 'Best Acc.',
                                        },
                                        parameter_columns={
                                            'gen_mode': 'Mode', 
                                            'n_gen': '#gen', 
                                            'C': 'C', 
                                            'depth': 'Depth', 
                                            'lr': 'Lrate',
                                            'tree_dropout': 'Drop.'
                                        }, 
                                        infer_limit=3)
            fold_exp = tune.run(
                GHTMNTrainable,
                name=name,
                stop=early_stopping,
                local_dir=exp_dir,
                config=config,
                num_samples=n_samples,
                resources_per_trial= resources,
                keep_checkpoints_num=1,
                checkpoint_score_attr='vl_acc',
                checkpoint_freq=1,
                max_failures=5,
                scheduler=scheduler,
                verbose=1,
                progress_reporter=reporter,
                resume=size_dir > 3
            )

        if not os.path.exists(os.path.join(fold_dir, 'test_res.pkl')):
            best_dict = get_best_info(fold_dir)
            t_config = best_dict['config']
            _, _, ts_idx = get_split(exp_dir, i)
            ts_data = ParallelTUDataset(
                            f'{DATASET}/D{t_config["depth"]}', 
                            DATASET, 
                            pre_transform=pre_transform(t_config["depth"]),
                            transform=transform(DATASET),
                            pool_size=72
                        )
            ts_data.data.x = ts_data.data.x.argmax(1).detach()
            ts_ld = DataLoader(ts_data[ts_idx], 
                               collate_fn=TreeCollater(t_config['depth']), 
                               batch_size=len(ts_idx), 
                               shuffle=False)
                               
            if config['gen_mode'] == 'bu':
                n_bu, n_td = t_config['n_gen'], 0
            elif config['gen_mode'] == 'td':
                n_bu, n_td = 0, t_config['n_gen']
            elif config['gen_mode'] == 'both':
                n_bu, n_td = math.ceil(t_config['n_gen']/2), math.floor(t_config['n_gen']/2)

            model = GraphHTMN(config['out'], n_bu, n_td, t_config['C'], t_config['symbols'], t_config['tree_dropout'])
            m_state = torch.load(best_dict['chk_file'], map_location='cpu')
            model.load_state_dict(m_state)
            device = f'cuda:{choice([config["gpu_ids"]])}' if config['gpu_ids'] else 'cpu'
            model.to(device)
            loss = torch.nn.BCEWithLogitsLoss()

            model.eval()
            with torch.no_grad():
                ts_loss = 0
                ts_acc = 0
                for _, b in enumerate(ts_ld):
                    b = b.to(device)
                    out = model(b.x, b.trees, b.batch)
                    loss_v = loss(out, b.y).item()
                    acc_v = accuracy(b.y, out.sigmoid().round())
                    w = (torch.max(b.batch)+1).item() /len(ts_idx)
                    ts_loss += w*loss_v
                    ts_acc += w*acc_v

            best_dict['ts_loss'] = ts_loss
            best_dict['ts_acc'] = ts_acc

            torch.save(best_dict, os.path.join(fold_dir, 'test_res.pkl'))
            print(f'Test results for fold {i}: Loss = {ts_loss} ---- Accuracy = {ts_acc}')