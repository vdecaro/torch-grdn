import time
import random

import torch
from ray import tune
import os

from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader

from torch_geometric.utils.metric import accuracy

from graph_htmn.graph_htmn import GraphHTMN
from exp.utils import get_cores, get_split

class GHTMNTrainable(tune.Trainable):

    def setup(self, config):
        self.device = 'cpu'

        # Dataset info
        self.dataset_name = config['dataset']
        self.out_features = config['out']
        self.depth = config['depth']
        self.symbols = config['symbols']
        self.batch_size = config['batch_size']
        self.tr_idx, self.vl_idx, _ = get_split(f'GHTMN_{self.dataset_name}', config['fold'])

        # Dataset and Loaders setup
        self.dataset = None
        self.tr_ld = None
        self.vl_ld = None
        self._data_setup('all')

        self.model = GraphHTMN(self.out_features, config['n_gen'], 0, config['C'], self.symbols, config['tree_dropout'])
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)


    def step(self):
        while True:
            try:
                tr_loss, tr_acc = self._train_fn()
                vl_loss, vl_acc = self._test_fn()
                break
            except RuntimeError:
                if self.device == 'cuda:0':
                    print("Switch to CPUs")
                    self.device = 'cpu'
                    self.model.to(self.device)
                    self.opt.state = self._optim_to(self.opt.state)
                    torch.cuda.empty_cache()

        self._device_handling()
        
        return {
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'vl_loss': vl_loss,
            'vl_acc': vl_acc
        }
    
    def _train_fn(self):
        self.model.train()
        l_avg = 0
        a_avg = 0
        for i, b in enumerate(self.tr_ld):
            b = b.to(self.device)
            self.opt.zero_grad()

            out = self.model(b.x, b.trees, b.batch)
            loss_v = self.loss(out, b.y)
            loss_v.backward()
            self.opt.step()
            
            w = (torch.max(b.batch)+1).item()/len(self.tr_idx)
            acc_v = accuracy(b.y, out.sigmoid().round())
            l_avg += w*loss_v.item()
            a_avg += w*acc_v
    
        return l_avg, a_avg
    
    @torch.no_grad()
    def _test_fn(self):
        self.model.eval()
        l_avg = 0
        a_avg = 0
        
        for i, b in enumerate(self.vl_ld):
            b = b.to(self.device)
            out = self.model(b.x, b.trees, b.batch)

            loss_v = self.loss(out, b.y).item()
            acc_v = accuracy(b.y, out.sigmoid().round())

            w = (torch.max(b.batch)+1).item()/len(self.vl_idx)
            acc_v = accuracy(b.y, out.sigmoid().round())
            l_avg += w*loss_v
            a_avg += w*acc_v
        
        return l_avg, a_avg

    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pth"))
        torch.save(self.opt.state_dict(), os.path.join(tmp_checkpoint_dir, "opt.pth"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        self.device = 'cpu'
        mod_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pth"), map_location='cpu')
        opt_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "opt.pth"), map_location='cpu')
        self.model.load_state_dict(mod_state_dict)
        self.opt.load_state_dict(opt_state_dict)
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)

    def reset_config(self, new_config):
        self.device = 'cpu'
        load_mode = None
        if new_config['depth'] != self.depth:
            load_mode = 'all'
        elif new_config['batch_size'] != self.batch_size:
            load_mode = 'loaders'
        self._data_setup(load_mode)
        self.model = GraphHTMN(self.out_features, new_config['n_gen'], 0, new_config['C'], self.symbols, new_config['tree_dropout'])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=new_config['lr'])
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)
        torch.cuda.empty_cache()
        
        return True

    def _data_setup(self, mode):
        if mode == 'all':
            self.dataset = ParallelTUDataset(
                f'/code/torch-grdn/{self.dataset_name}/D{self.depth}', 
                self.dataset_name, 
                pre_transform=pre_transform(self.depth),
                transform=transform(self.dataset_name),
                pool_size=get_cores()
            )
            self.dataset.data.x = self.dataset.data.x.argmax(1).detach()
        if mode in ['all', 'loaders']:
            self.tr_ld = DataLoader(self.dataset[self.tr_idx], 
                                    collate_fn=TreeCollater(self.depth), 
                                    batch_size=self.batch_size, 
                                    shuffle=True)
            self.vl_ld = DataLoader(self.dataset[self.vl_idx], 
                                    collate_fn=TreeCollater(self.depth), 
                                    batch_size=self.batch_size, 
                                    shuffle=False)
        
    def _device_handling(self):
        if torch.cuda.is_available():
            t1 = time.time()
            if t1 - self.t0 >= self.t_threshold:
                self.t0 = t1
                self.t_threshold = random.uniform(30, 60)
                self.device = 'cuda:0' if self.device == 'cpu' else 'cpu'
                try:
                    self.model.to(self.device)
                    self.opt.state = self._optim_to(self.opt.state)
                except RuntimeError as err:
                    print("Attempted Switch to GPU and failed. Returning to CPU.")
                    self.device = 'cpu'
                    self.model.to(self.device)
                    self.opt.state = self._optim_to(self.opt.state)
                    torch.cuda.empty_cache()
    
    def _optim_to(self, var):
        for key in var:
            if isinstance(var[key], dict):
                var[key] = self._optim_to(var[key])
            elif torch.is_tensor(var[key]):
                var[key] = var[key].to(self.device)
        
        return var