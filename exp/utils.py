import os
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from data.graph.g2t import ParallelTUDataset, pre_transform

def get_seed():
    return 95

def prepare_dir_tree_experiments(name):
    dataset = TUDataset('.', name)
    exp_dir = 'GHTMN_{}'.format(name)
    os.makedirs(exp_dir)
    ext_kfold = StratifiedKFold(10, shuffle=True, random_state=get_seed())
    ext_split = list(ext_kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))
    for n, (ds_i, ts_i) in enumerate(ext_split):
        ds_data = dataset[ds_i.tolist()]
        tr_i, vl_i = train_test_split(ds_i, 
                                      test_size=0.175,  
                                      stratify=np.array([g.y for g in ds_data]), 
                                      shuffle=True, 
                                      random_state=get_seed())
        
        fold_dir = os.path.join(exp_dir, 'fold_{}'.format(n))
        os.makedirs(fold_dir)
        np.save(os.path.join(fold_dir, 'tr_i.npy'), tr_i)
        np.save(os.path.join(fold_dir, 'vl_i.npy'), vl_i)
        np.save(os.path.join(fold_dir, 'ts_i.npy'), ts_i)



def prepare_tree_datasets(name, depths, num_cpus):
    for d in depths:
        if not os.path.exists('{}/D{}'.format(name, d)):
            _ = ParallelTUDataset('{}/D{}'.format(name, d), name, pre_transform=pre_transform(d), pool_size=num_cpus)


def get_split(exp_dir, fold):
    fold_dir = os.path.join(exp_dir, 'fold_{}'.format(fold))
    tr_i = np.load(os.path.join(fold_dir, 'tr_i.npy'))
    vl_i = np.load(os.path.join(fold_dir, 'vl_i.npy'))
    ts_i = np.load(os.path.join(fold_dir, 'ts_i.npy'))
    
    return tr_i.tolist(), vl_i.tolist(), ts_i.tolist()
