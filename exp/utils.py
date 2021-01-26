import os
import json
import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from data.graph.g2t import ParallelTUDataset, pre_transform
from ray.tune import Analysis

from sklearn.metrics import roc_auc_score
from torch_geometric.utils import accuracy

def get_seed():
    return 95

def get_loss_fn(loss_type):
    if loss_type == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    if loss_type == 'ce':
        return torch.nn.CrossEntropyLoss()


def get_score_fn(score_type, outs):
    
    if score_type == 'accuracy':
        if outs == 1:
            def _score_fn(y, pred):
                return accuracy(y, pred.sigmoid().round())
        else:
            def _score_fn(y, pred):
                return accuracy(y, pred.argmax(-1))
            
    if score_type == 'roc-auc':
        def _score_fn(y, pred):
            return roc_auc_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())

    return _score_fn


def get_best_info(exp_dir, metrics=['vl_acc', 'vl_loss'], ascending=[False, True], mode='auto'):
    if mode == 'auto':
        analysis = Analysis(exp_dir, 'vl_acc', 'max')
        df = analysis.dataframe()
        df = df.sort_values(metrics, ascending=ascending)
        trial_dir = df.iloc[0][-1]

        min_ = 10000
        for f in os.listdir(trial_dir):
            if 'checkpoint' in f:
                idx = int(f.split('_')[1])
                min_ = min(min_, idx)
        chk_file = os.path.join(trial_dir, f'checkpoint_{min_}', 'model.pth')
        with open(os.path.join(trial_dir, 'params.json')) as f:
            config = json.load(f)
        with open(os.path.join(trial_dir, 'result.json')) as f:
            res = [json.loads(i) for i in f]
            best_res = res[min_-1]
            
        return {
            'trial_dir': trial_dir, 
            'chk_file': chk_file, 
            'config': config, 
            'tr_loss': best_res['tr_loss'], 
            'tr_acc': best_res['tr_acc'], 
            'vl_loss': best_res['vl_loss'], 
            'vl_acc': best_res['vl_acc']
        }

    elif mode == 'manual':
        best_dict = {
            'trial_dir': None,
            'chk_file': None,
            'config': None,
            'tr_loss': float('inf'),
            'tr_acc': 0,
            'vl_loss': float('inf'),
            'vl_acc': 0
        }
        dirs = [part_dir for part_dir in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, part_dir))]
        for part_dir in dirs:
            trial_dir = os.path.join(exp_dir, part_dir)
            min_ = 400
            for f in os.listdir(trial_dir):
                if 'checkpoint' in f:
                    idx = int(f.split('_')[1])
                    min_ = min(min_, idx)
            with open(os.path.join(trial_dir, 'result.json')) as f:
                for i, d in enumerate(f):
                    if i+1 == min_:
                        curr = json.loads(d)
                        if best_dict['vl_acc'] < curr['vl_acc'] or (best_dict['vl_acc'] == curr['vl_acc'] and best_dict['vl_loss'] > curr['vl_loss']):
                            with open(os.path.join(trial_dir, 'params.json')) as f:
                                config = json.load(f)
                            best_dict = {
                                'trial_dir': trial_dir,
                                'chk_file': os.path.join(trial_dir, f'checkpoint_{min_}/model.pth'),
                                'config': config,
                                'tr_loss': curr['tr_loss'],
                                'tr_acc': curr['tr_acc'],
                                'vl_loss': curr['vl_loss'],
                                'vl_acc': curr['vl_acc']
                            }
        return best_dict