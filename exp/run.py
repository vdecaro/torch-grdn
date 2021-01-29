import os
import math
import json
import random

import torch
import ray
from ray import tune
from exp.gpu_trainable import GPUTrainable
from ray.tune.schedulers import ASHAScheduler
from exp.early_stopper import TrialNoImprovementStopper

def run_exp(design_or_test,
            config,
            n_samples,
            p_early,
            p_scheduler,
            exp_dir,
            chk_score_attr,
            log_params,
            gpus=[],
            gpu_threshold=None):
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    config['wdir'] = os.getcwd()
    config['gpu_ids'] = gpus
    config['gpu_threshold'] = gpu_threshold
    early_stopping = TrialNoImprovementStopper(metric=p_early['metric'], 
                                               mode=p_early['mode'], 
                                               patience_threshold=p_early['patience'])
    if p_scheduler is not None:
        scheduler = ASHAScheduler(
            metric=p_scheduler['metric'],
            mode=p_scheduler['mode'],
            max_t=p_scheduler['max_t'],
            grace_period=p_scheduler['grace'],
            reduction_factor=p_scheduler['reduction']
        ) 
    else:
        scheduler = None
    
    resources = {'cpu': 2, 'gpu': 0.0001}
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter',
                                    'tr_loss': 'TR-Loss',
                                    'tr_score': 'TR-Score',
                                    'vl_loss': 'VL-Loss', 
                                    'vl_score': 'VL-Score', 
                                    'best_score': 'Top Score',
                                },
                                parameter_columns=log_params,
                                infer_limit=3,
                                metric='best_score',
                                mode='max')
    return tune.run(
        GPUTrainable,
        name=design_or_test,
        stop=early_stopping,
        local_dir=exp_dir,
        config=config,
        num_samples=n_samples,
        resources_per_trial=resources,
        keep_checkpoints_num=1,
        checkpoint_score_attr=chk_score_attr,
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        scheduler=scheduler,
        verbose=1
    )


def run_test(trial_dir,
             ts_ld,
             model_func,
             loss_fn,
             score_fn,
             gpus):
    device = f'cuda:{random.choice(gpus)}' if gpus else 'cpu'
    min_ = 10000
    for f in os.listdir(trial_dir):
        if 'checkpoint' in f:
            idx = int(f.split('_')[1])
            min_ = min(min_, idx)
    with open(os.path.join(trial_dir, 'params.json')) as f:
        t_config = json.load(f)

    chk_file = os.path.join(trial_dir, f'checkpoint_{min_}', 'model.pth')
    model = model_func(t_config)
    m_state = torch.load(chk_file, map_location='cpu')
    model.load_state_dict(m_state)
    model.to(device)

    model.eval()
    y, pred = [], []
    with torch.no_grad():
        for b in ts_ld:
            out = model(b.to(device))
            y.append(b.y)
            pred.append(out)
    y, pred = torch.cat(y, 0), torch.cat(pred, 0)
    
    return loss_fn(pred, y), score_fn(y, pred)
    