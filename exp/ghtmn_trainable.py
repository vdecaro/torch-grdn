import torch
from ray import tune
import os

from data.graph.g2t import ParallelTUDataset, TreeCollater, pre_transform, transform
from torch_geometric.data import DataLoader

from torch_geometric.utils.metric import accuracy

from graph_htmn.graph_htmn import GraphHTMN


class GHTMNTrainable(tune.Trainable):

    def setup(self, config):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.n_cpus = config['n_cpus']

        # Dataset info
        self.dataset_name = config['dataset']
        self.out_features = config['out']
        self.depth = config['depth']
        self.symbols = config['symbols']
        self.batch_size = config['batch_size']
        self.tr_idx = config['tr_idx']
        self.vl_idx = config['vl_idx']

        # Dataset and Loaders setup
        self.dataset = None
        self.tr_ld = None
        self.vl_ld = None
        self._data_setup('all')

        self.model = GraphHTMN(self.out_features, config['n_gen'], 0, config['C'], self.symbols)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.loss = torch.nn.BCEWithLogitsLoss()


    def step(self):
        tr_loss, tr_acc = self._train_fn()
        vl_loss, vl_acc = self._test_fn()

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
            b = b.to(self.device, non_blocking=True)
            self.opt.zero_grad()

            out = self.model(b.x, b.edge_index, b.batch)
            loss_v = self.loss(out, b.y)
            loss_v.backward()
            self.opt.step()

            acc_v = accuracy(b.y, out.sigmoid().round())
            l_avg += (loss_v - l_avg) / (i+1)
            a_avg += (acc_v - a_avg) / (i+1)

        return l_avg, a_avg
    
    def _test_fn(self):
        self.model.eval()
        l_avg = 0
        a_avg = 0
        for i, b in self.vl_ld:
            with torch.no_grad():
                b = b.to(self.device, non_blocking=True)
                out = self.model(b.x, b.edge_index, b.batch)

                loss_v = self.loss(out, b.y).item()
                acc_v = accuracy(b.y, out.sigmoid().round())

            l_avg += (loss_v - l_avg) / (i+1)
            a_avg += (acc_v - a_avg) / (i+1)
        
        return loss_v, acc_v

    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pth"))
        torch.save(self.opt.state_dict(), os.path.join(tmp_checkpoint_dir, "opt.pth"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        self.model.load_state_dict(torch.load(os.path.join(tmp_checkpoint_dir, "model.pth")))
        self.opt.load_state_dict(torch.load(os.path.join(tmp_checkpoint_dir, "opt.pth")))

    def reset_config(self, new_config):
        d_config = new_config['DATASET']
        load_mode = 'all' if d_config['depth'] != self.depth else ('loaders' if d_config['batch_size'] != self.batch_size else None)
        self._data_setup(load_mode)
        self.model = GraphHTMN(self.out_features, new_config['n_gen'], 0, new_config['C'], self.symbols, 'cuda:0')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=new_config['lr'])

    def _data_setup(self, mode):
        if mode == 'all':
            self.dataset = ParallelTUDataset(
                f'{self.dataset_name}/D{self.depth}', 
                self.dataset_name, 
                pre_transform=pre_transform(self.depth), 
                transform=transform(self.dataset_name), 
                pool_size=None
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