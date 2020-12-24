import torch
from ray import tune
import os

from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader

from torch_geometric.utils.metric import accuracy

from graph_htmn.graph_htmn import GraphHTMN
from exp.utils import get_split
from exp.device_handler import DeviceHandler

class GHTMNTrainable(tune.Trainable):

    def setup(self, config):
        if torch.cuda.is_available():
            self.device_handler = DeviceHandler(int(os.environ['CUDA_VISIBLE_DEVICES']))

        # Dataset info
        self.dataset_name = config['dataset']
        self.out_features = config['out']
        self.depth = config['depth']
        self.symbols = config['symbols']
        self.batch_size = config['batch_size']
        self.tr_idx, self.vl_idx, _ = get_split('GHTMN_{}'.format(self.dataset_name), config['fold'])

        # Dataset and Loaders setup
        self.dataset = None
        self.tr_ld = None
        self.vl_ld = None
        self._data_setup('all')

        self.model = GraphHTMN(self.out_features, config['n_gen'], 0, config['C'], self.symbols, config['tree_dropout'])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = torch.nn.BCEWithLogitsLoss()

    def step(self):
        while True:
            try:
                tr_loss, tr_acc = self._train_fn()
                vl_loss, vl_acc = self._test_fn()
                break
            except RuntimeError:
                if torch.cuda.is_available() and self.device_handler.device == 'cuda:0':
                    self.model, self.opt = self.device_handler.switch_device(self.model, self.opt)
                else:
                    raise
        
        if torch.cuda.is_available():
            self.model, self.opt = self.device_handler.step(self.model, self.opt)

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
        for _, b in enumerate(self.tr_ld):
            if torch.cuda.is_available() and self.device_handler.device == 'cuda:0':
                b = b.to('cuda:0', non_blocking=True)
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
        
        for _, b in enumerate(self.vl_ld):
            if torch.cuda.is_available() and self.device_handler.device == 'cuda:0':
                b = b.to('cuda:0', non_blocking=True)
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
        mod_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pth"), map_location='cpu')
        opt_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "opt.pth"), map_location='cpu')
        self.model.load_state_dict(mod_state_dict)
        self.opt.load_state_dict(opt_state_dict)

        if torch.cuda.is_available():   
            self.device_handler.reset()

    def reset_config(self, new_config):
        load_mode = None
        if new_config['depth'] != self.depth:
            load_mode = 'all'
        elif new_config['batch_size'] != self.batch_size:
            load_mode = 'loaders'
        self._data_setup(load_mode)
        self.model = GraphHTMN(self.out_features, new_config['n_gen'], 0, new_config['C'], self.symbols, new_config['tree_dropout'])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=new_config['lr'])
        if torch.cuda.is_available():   
            self.device_handler.reset()

        return True

    def _data_setup(self, mode):
        if mode == 'all':
            self.dataset = ParallelTUDataset(
                '~/torch-grdn/{}/D{}'.format(self.dataset_name, self.depth), 
                self.dataset_name, 
                pre_transform=pre_transform(self.depth),
                transform=transform(self.dataset_name)
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
