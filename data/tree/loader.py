from torch.utils.data import Dataset, DataLoader
from data.tree.inex.preproc import load_and_preproc_inex


def raw_load(dataset):
    if dataset == 'inex2005' or dataset == 'inex2006':
        features, targets = load_and_preproc_inex(dataset)
        return _TreeDataset(features, targets)


def TreeLoader(tree_dataset, batch_size, shuffle=False):
    
    def collate_trees_fn(batch):
        trees, targets = [], []
        for tree, target in batch:
            trees.append(tree)
            targets.append(target)

        return trees, targets
    
    return DataLoader(tree_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_trees_fn)


class _TreeDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
              
    def __getitem__(self, index):
        idx_features = {
            'levels': self.features['levels'][index], 
            'leaves': self.features['leaves'][index],
            'labels': self.features['labels'][index],
            'pos': self.features['pos'][index]
        }
        return idx_features, self.targets[index]
    
    def __len__(self):
        return len(self.targets)
