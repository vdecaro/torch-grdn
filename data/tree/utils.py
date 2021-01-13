import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from data.tree.inex.preproc import load_and_preproc_inex
import random

class TreeData(Data):

    def to(self, device):
        self.x = self.x.to(device)
        self.levels = [l.to(device) for l in self.levels]
        self.leaves = self.leaves.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.batch = self.batch.to(device)

        return self
    
    def cpu(self):
        self.x = self.x.cpu()
        self.levels = [l.cpu() for l in self.levels]
        self.leaves = self.leaves.cpu()
        self.pos = self.pos.cpu()
        self.y = self.y.cpu()
        self.batch = self.batch.cpu()

        return self
    
    def cuda(self):
        self.x = self.x.cuda()
        self.levels = [l.cuda() for l in self.levels]
        self.leaves = self.leaves.cuda()
        self.pos = self.pos.cuda()
        self.y = self.y.cuda()
        self.batch = self.batch.cuda()

        return self

class TreeDataset(Dataset):
    def __init__(self, work_dir=None, name=None, data=None):
        if work_dir is not None and name in ['inex2005train', 'inex2005test', 'inex2006train', 'inex2006test']:
            self.data = load_and_preproc_inex(work_dir, name)
        elif data is not None:
            self.data=data
              
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def shuffle(self):
        random.shuffle(self.data)


def trees_collate_fn(batch):
    x = []
    levels = []
    leaves = []
    pos = []
    y = []
    dim = 0
    batch_idx = []
    for b_idx, t in enumerate(batch):
        x.append(t.x)
        for i, l in enumerate(t.levels):
            if i == len(levels):
                levels.append([])
            levels[i].append(l + dim)
        
        leaves.append(t.leaves + dim)
        pos.append(t.pos)
        y.append(t.y)
        batch_idx += [b_idx for _ in range(t.dim)]
        dim += t.dim

    return TreeData(
        x=torch.cat(x),
        levels=[torch.cat(l, 1) for l in levels],
        leaves=torch.cat(leaves),
        pos=torch.cat(pos),
        y=torch.LongTensor(y),
        dim=dim,
        batch=torch.LongTensor(batch_idx)
    )
