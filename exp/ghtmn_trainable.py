import os
import math
from ray import tune
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch.utils.data import DataLoader
from torch_geometric.utils.metric import accuracy
from graph_htmn.graph_htmn import GraphHTMN
from exp.utils import get_split
from exp.device_handler import DeviceHandler

class GHTMNTrainable(tune.Trainable):

    def setup(self, config):
        self.device_handler = DeviceHandler(self, config['gpu_ids'])
        
        # Dataset info
        self.tr_idx, self.vl_idx, _ = get_split(os.path.join(config['wdir'], 'GHTMN_{}'.format(config['dataset'])), config['fold'])
        dataset = ParallelTUDataset(
            os.path.join(config['wdir'], config['dataset'], 'D{}'.format(config['depth'])),
            config['dataset'], 
            pre_transform=pre_transform(config['depth']),
            transform=transform(config['dataset'])
        )
        dataset.data.x = dataset.data.x.argmax(1).detach()

        self.tr_ld = DataLoader(dataset[self.tr_idx], 
                                collate_fn=TreeCollater(config['depth']), 
                                batch_size=config['batch_size'], 
                                shuffle=True)
        self.vl_ld = DataLoader(dataset[self.vl_idx], 
                                collate_fn=TreeCollater(config['depth']), 
                                batch_size=config['batch_size'], 
                                shuffle=False)
        if config['gen_mode'] == 'bu':
            n_bu, n_td = config['n_gen'], 0
        elif config['gen_mode'] == 'td':
            n_bu, n_td = 0, config['n_gen']
        elif config['gen_mode'] == 'both':
            n_bu, n_td = math.ceil(config['n_gen']/2), math.floor(config['n_gen']/2)

        self.model = GraphHTMN(config['out'], n_bu, n_td, config['C'], config['symbols'], config['tree_dropout'])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = torch.nn.BCEWithLogitsLoss()

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

        return {
            'tr_loss': tr_loss,
            #'tr_acc': tr_acc,
            'vl_loss': vl_loss,
            'vl_acc': vl_acc
        }
    
    def _train_step(self, batch):

        @self.device_handler.forward_manage
        def _func(b):
            self.opt.zero_grad()
            out = self.model(b.x, b.trees, b.batch)
            loss_v = self.loss(out, b.y)
            loss_v.backward()
            self.opt.step()
            loss_v = loss_v.item()
            acc_v = accuracy(b.y, out.sigmoid().round())
        
            return loss_v, acc_v
        
        return _func(batch)
                    
    @torch.no_grad()
    def _test_step(self, batch):
        
        @self.device_handler.forward_manage
        def _func(b):
            out = self.model(b.x, b.trees, b.batch)
            loss_v = self.loss(out, b.y).item()
            acc_v = accuracy(b.y, out.sigmoid().round())

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
