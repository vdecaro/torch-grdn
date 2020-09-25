import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from data.tree.inex.preproc import load_and_preproc_inex
import random

class TreeDataset(Dataset):
    def __init__(self, name=None, data=None):
        if name in ['inex2005train', 'inex2005test', 'inex2006train', 'inex2006test']:
            self.data = load_and_preproc_inex(name)
        if data is not None:
            self.data=data
              
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def shuffle(self):
        random.shuffle(self.data)

class TreesLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(TreesLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=trees_collate_fn, num_workers=0, **kwargs)


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

    return Data(
        x=torch.cat(x),
        levels=[torch.cat(l, 1) for l in levels],
        leaves=torch.cat(leaves),
        pos=torch.cat(pos),
        y=torch.stack(y),
        dim=dim,
        batch=torch.LongTensor(batch_idx)
    )
