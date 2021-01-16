import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import gc

import torch
import numpy as np
from ray import tune

from math import floor, ceil
from data.tree.utils import TreeDataset, trees_collate_fn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from torch_geometric.utils.metric import accuracy

from htmn.htmn import HTMN
from exp.device_handler import DeviceHandler
from exp.utils import get_seed

class HTMNTrainable(tune.Trainable):

    def setup(self, config):
        self.device_handler = DeviceHandler(self, config['gpu_ids'])

        # Dataset and Loaders setup
        dataset = TreeDataset(work_dir=config['wdir'], name=config['dataset'] + 'train')
        self.tr_idx, self.vl_idx = train_test_split(np.arange(len(dataset)), 
                                                    test_size=config['holdout'],  
                                                    stratify=np.array([t.y for t in dataset]), 
                                                    shuffle=True, 
                                                    random_state=get_seed())
        self.tr_idx, self.vl_idx = self.tr_idx.tolist(), self.vl_idx.tolist()
        tr_data = TreeDataset(data=[dataset[i] for i in self.tr_idx])
        vl_data = TreeDataset(data=[dataset[i] for i in self.vl_idx])
        self.tr_ld = DataLoader(tr_data, batch_size=config['batch_size'], shuffle=True, collate_fn=trees_collate_fn, drop_last=len(self.tr_idx) % config['batch_size'] == 1)
        self.vl_ld = DataLoader(vl_data, batch_size=config['batch_size'], shuffle=False, collate_fn=trees_collate_fn, drop_last=len(self.vl_idx) % config['batch_size'] == 1)

        self.model = HTMN(config['out'], ceil(config['n_gen']/2), floor(config['n_gen']/2), config['C'], config['L'], config['M'])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', factor=0.25)
        self.best_vl_acc = 0

    def step(self):
        self.model.train()
        tr_loss = 0
        tr_acc = 0
        for _, b in enumerate(self.tr_ld):
            b_loss_v, b_acc_v = self._train_step(b)
            w = (torch.max(b.batch)+1).item() /len(self.tr_idx)
            tr_loss += w*b_loss_v
            tr_acc += w*b_acc_v
        
        
        self.model.eval()
        vl_loss = 0
        vl_acc = 0
        for _, b in enumerate(self.vl_ld):
            b_loss_v, b_acc_v = self._test_step(b)
            w = (torch.max(b.batch)+1).item() /len(self.vl_idx)
            vl_loss += w*b_loss_v
            vl_acc += w*b_acc_v
        
        self.device_handler.step()
        self.lr_scheduler.step(vl_acc)
        
        if vl_acc > self.best_vl_acc:
            self.best_vl_acc = vl_acc

        return {
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'vl_loss': vl_loss,
            'vl_acc': vl_acc,
            'best_acc': self.best_vl_acc
        }
    
    def _train_step(self, batch):

        @self.device_handler.forward_manage
        def _func(b):
            self.opt.zero_grad()
            out = self.model(b)
            loss_v = self.loss(out, b.y)
            loss_v.backward()
            self.opt.step()
            loss_v = loss_v.item()
            acc_v = accuracy(b.y, out.argmax(-1))
        
            return loss_v, acc_v
        
        return _func(batch)
                    
    @torch.no_grad()
    def _test_step(self, batch):
        
        @self.device_handler.forward_manage
        def _func(b):
            out = self.model(b)
            loss_v = self.loss(out, b.y).item()
            acc_v = accuracy(b.y, out.argmax(-1))

            return loss_v, acc_v

        return _func(batch)
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pth"))
        torch.save(self.opt.state_dict(), os.path.join(tmp_checkpoint_dir, "opt.pth"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        mod_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pth"), map_location='cpu')
        opt_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "opt.pth"), map_location='cpu')
        self.model.load_state_dict(mod_state_dict)
        self.opt.load_state_dict(opt_state_dict)

    def cleanup(self):
        self.device_handler.cleanup()
