import sys
import os
import math
from random import randint, randrange, choice

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from exp.htmn_trainable import HTMNTrainable
from exp.utils import prepare_dir_tree_experiments, prepare_tree_datasets, get_best_info
from exp.early_stopper import TrialNoImprovementStopper

from data.tree.utils import TreeDataset,trees_collate_fn
from htmn.htmn import HTMN
from torch.utils.data import DataLoader
from torch_geometric.utils.metric import accuracy

def get_config(name):
    if name == 'inex2005':
        return {
            'dataset': 'inex2005',
            'phase': 'eval',
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
            'phase': 'eval',
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
    '''
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
    
    resources = {'cpu': 2, 'gpu': 0.0001}
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
        name='design',
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
    '''
    best_dict = get_best_info(os.path.join(exp_dir, 'design'))
    t_config = best_dict['config']
    t_config['wdir'] = os.getcwd()
    t_config['gpu_ids'] = [int(i) for i in sys.argv[3].split(',')]
    t_config['phase'] = 'test'
    #t_config['out'] = tune.choice([t_config['out']])
    early_stopping = TrialNoImprovementStopper('tr_loss', mode='min', patience_threshold=40)
    
    resources = {'cpu': 2, 'gpu': 0.0001}
    n_samples = 5
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter', 
                                    'vl_loss': 'Loss', 
                                    'vl_acc': 'Acc.', 
                                    'best_acc': 'Best Acc.'
                                },
                                parameter_columns={
                                    'n_gen': '#gen', 
                                    'C': 'C', 
                                    'lr': 'LRate',
                                    'batch_size': 'Batch'
                                }, 
                                infer_limit=3,)
    '''
    tune.run(
        HTMNTrainable,
        name='test',
        stop=early_stopping,
        local_dir=exp_dir,
        config=t_config,
        num_samples=n_samples,
        resources_per_trial= resources,
        keep_checkpoints_num=1,
        checkpoint_score_attr='vl_acc',
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        verbose=1
    )
    '''
    
    #t_config['out'] = 
    ts_data = TreeDataset('.', f'{DATASET}test')
    ts_ld = DataLoader(ts_data, 
                       collate_fn=trees_collate_fn, 
                       batch_size=512, 
                       shuffle=False)
    device = f'cuda:{choice(t_config["gpu_ids"])}' if t_config['gpu_ids'] else 'cpu'
    loss = torch.nn.CrossEntropyLoss()
    best_dict['ts_loss'] = []
    best_dict['ts_acc'] = []
    test_dir = os.path.join(exp_dir, 'test')
    for i, t_dir in enumerate(os.listdir(test_dir)):
        trial_dir = os.path.join(test_dir, t_dir)
        if os.path.isdir(trial_dir):
            min_ = 10000
            for f in os.listdir(trial_dir):
                if 'checkpoint' in f:
                    idx = int(f.split('_')[1])
                    min_ = min(min_, idx)

            chk_file = os.path.join(trial_dir, f'checkpoint_{min_}', 'model.pth')
            model = HTMN(t_config['out'], math.ceil(t_config['n_gen']/2), math.floor(t_config['n_gen']/2), t_config['C'], t_config['L'], t_config['M'])
            m_state = torch.load(chk_file, map_location='cpu')
            model.load_state_dict(m_state)
            model.to(device)

            model.eval()
            ts_loss = 0
            ts_acc = 0
            with torch.no_grad():
                for _, b in enumerate(ts_ld):
                    b = b.to(device)
                    out = model(b)
                    loss_v = loss(out, b.y).item()
                    acc_v = accuracy(b.y, out.argmax(-1))
                    w = (torch.max(b.batch)+1).item() /len(ts_data)
                    ts_loss += w*loss_v
                    ts_acc += w*acc_v
            print(f'Test {i} results: Loss = {ts_loss} ---- Accuracy = {ts_acc}')
            best_dict['ts_loss'].append(ts_loss)
            best_dict['ts_acc'].append(ts_acc)

        torch.save(best_dict, os.path.join(exp_dir, 'test_res.pkl'))
    