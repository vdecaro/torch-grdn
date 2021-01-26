import os
import math
from ray import tune
import torch

from data.tree.utils import TreeDataset, trees_collate_fn
from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader

from graph_htmn.graph_htmn import GraphHTMN
from htmn.htmn import HTMN

from exp.utils import get_loss_fn, get_score_fn

class TrainWrapper(object):

    def __init__(self, config):

        self.model, self.opt, self.tr_ld, self.vl_ld = _wrapper_init_fn(config)

        self.loss_fn = get_loss_fn(config['loss'])
        self.score_fn = get_score_fn(config['score'], config['out'])
        
        e_min = config['epochs_decay']*(len(config['tr_idx']) //config['batch_size'])
        lr_lambda = lambda e: (config['lr']*(e_min-e)/e_min + config['min_lr']*e/e_min)/config['lr'] if e <= e_min else config['min_lr']
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        self.best_score = 0

    def step(self, device):
        res_dict = {}

        self.model.train()
        tr_y, tr_pred = [], []
        for _, b in enumerate(self.tr_ld):
            out = self.model(b.cuda() if 'cuda' in device else b)
            loss_v = self.loss_fn(out, b.y)
            self.opt.zero_grad()
            loss_v.backward()
            self.opt.step()
            self.lr_scheduler.step()
            tr_y.append(b.y)
            tr_pred.append(out)
        
        tr_y, tr_pred = torch.cat(tr_y, 0), torch.cat(tr_pred, 0)
        res_dict['tr_loss'], res_dict['tr_score'] = self.loss_fn(tr_pred, tr_y).item(), self.score_fn(tr_y, tr_pred)

        if self.vl_ld is not None:
            self.model.eval()
            vl_y, vl_pred = [], []
            for _, b in enumerate(self.vl_ld):
                with torch.no_grad():
                    out = self.model(b.cuda() if 'cuda' in device else b)
                vl_y.append(b.y)
                vl_pred.append(out)

            vl_y, vl_pred = torch.cat(vl_y, 0), torch.cat(vl_pred, 0)
            res_dict['vl_loss'], res_dict['vl_score'] = self.loss_fn(vl_pred, vl_y).item(), self.score_fn(vl_y, vl_pred)
            if res_dict['vl_score'] > self.best_score:
                self.best_score = res_dict['vl_score']
        else:
            if res_dict['tr_score'] > self.best_score:
                self.best_acc = res_dict['tr_score']
        
        res_dict['best_score'] =  self.best_score
        return res_dict


def _wrapper_init_fn(config):
    if config['model'] == 'htmn':
        dataset = TreeDataset(config['wdir'], config['dataset'])
        tr_ld = DataLoader(TreeDataset(data=[dataset[i] for i in config['tr_idx']]), 
                                batch_size=config['batch_size'], 
                                shuffle=True, 
                                collate_fn=trees_collate_fn, 
                                drop_last=len(config['tr_idx']) % config['batch_size'] == 1)
        if config['vl_idx'] is not None:
            vl_ld = DataLoader(TreeDataset(data=[dataset[i] for i in config['vl_idx']]), 
                                    batch_size=config['batch_size'], 
                                    shuffle=False, 
                                    collate_fn=trees_collate_fn, 
                                    drop_last=len(config['vl_idx']) % config['batch_size'] == 1)
        else:
            vl_ld = None

        model = HTMN(config['out'], math.ceil(config['n_gen']/2), math.floor(config['n_gen']/2), config['C'], config['L'], config['M'])
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
    if config['model'] == 'ghtmn':
        tr_idx, vl_idx = config['tr_idx'], config['vl_idx']
        dataset = ParallelTUDataset(
            os.path.join(config['wdir'], config['dataset'], f'D{config["depth"]}'),
            config['dataset'], 
            pre_transform=pre_transform(config['depth']),
            transform=transform(config['dataset'])
        )
        dataset.data.x = dataset.data.x.argmax(1).detach()

        tr_ld = DataLoader(dataset[tr_idx], 
                           collate_fn=TreeCollater(config['depth']), 
                           batch_size=config['batch_size'], 
                           shuffle=True)
        vl_ld = DataLoader(dataset[vl_idx], 
                           collate_fn=TreeCollater(config['depth']), 
                           batch_size=config['batch_size'], 
                           shuffle=False)

        if config['gen_mode'] == 'bu':
            n_bu, n_td = config['n_gen'], 0
        elif config['gen_mode'] == 'td':
            n_bu, n_td = 0, config['n_gen']
        elif config['gen_mode'] == 'both':
            n_bu, n_td = math.ceil(config['n_gen']/2), math.floor(config['n_gen']/2)

        model = GraphHTMN(config['out'], n_bu, n_td, config['C'], config['symbols'], config['tree_dropout'])
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])

    return model, opt, tr_ld, vl_ld