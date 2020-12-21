import torch

from torch_geometric.data import Dataset, Data, Batch

import ray

import time
import sys
import os
import os.path as osp
import shutil

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data


class ParallelTUDataset(InMemoryDataset):

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, pool_size=4):
        self.name = name
        self.cleaned = cleaned
        self.pool_size = pool_size
        super(ParallelTUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            filter_pool = ray.util.ActorPool([self.pre_filter for _ in range(self.pool_size)])
            mask_list = list(filter_pool.map(lambda a, v: a.remote(v), data_list))
            data_list = [data for i, data in enumerate(data_list) if mask_list[i]]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            transform_pool = ray.util.ActorPool([self.pre_transform for _ in range(self.pool_size)])
            transformed_data = []
            for i in range(0, len(data_list), self.pool_size*4):
                last_idx = min(i+(self.pool_size*4), len(data_list))
                transformed_data += list(transform_pool.map(lambda a, v: a.remote(v), data_list[i:last_idx]))
            self.data, self.slices = self.collate(transformed_data)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


def pre_transform(max_depth):
    
    @ray.remote
    def func(data):
        data['trees'] = bfs_transform(data.x, data.edge_index, max_depth)
        return data

    return func 

def transform(dataset):
    
    def func(data):
        data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
        return data

    return func

class TreeDecomposedData(Data):

    def to(self, device, *keys, **kwargs):
        self.x = self.x.to(device=device)
        self.y = self.y.to(device=device)
        self.batch = self.batch.to(device=device)
        for k in self.trees:
            if k != 'dim':
                self.trees[k] = self.trees[k].to(device=device) if k != 'levels' else [l.to(device=device) for l in self.trees[k]]
        return self


class TreeCollater(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def collate(self, batch):
        x = [batch[0].x]
        y = [batch[0].y]
        trees = batch[0].trees

        roots = [trees['roots']]
        levels = [[] for _ in range(self.max_depth-1)]
        for i, l in enumerate(trees['levels']):
            levels[i].append(l)
        leaves = [trees['leaves']]
        inv_map = [trees['inv_map']]
        trees_ind = [trees['trees_ind']]
        dim = trees['dim']

        batch_ind = [0 for _ in range(batch[0].x.size(0))]
        trees_offset = trees['dim']
        graph_offset = batch[0].x.size(0)

        for n_graph, g in enumerate(batch[1:]):
            trees = g.trees
            x.append(g.x)
            y.append(g.y)
            roots.append(trees['roots'] + trees_offset)
            for i, l in enumerate(trees['levels']):
                levels[i].append(l + trees_offset)
            leaves.append(trees['leaves'] + trees_offset)
            inv_map.append(trees['inv_map'] + graph_offset)
            trees_ind.append(trees['trees_ind'] + graph_offset)
            dim += trees['dim']
            batch_ind += [n_graph+1 for _ in range(g.x.size(0))]
            
            trees_offset += trees['dim']
            graph_offset += g.x.size(0)

        x = torch.cat(x)
        y = torch.cat(y)
        coll_t = {'roots': torch.cat(roots),
                'levels': [torch.cat(l, 1) for l in levels if l],
                'leaves': torch.cat(leaves),
                'inv_map': torch.cat(inv_map),
                'trees_ind': torch.cat(trees_ind),
                'dim': dim
                }
        batch_ind = torch.LongTensor(batch_ind)
        
        return TreeDecomposedData(x=x, 
                                  batch=batch_ind, 
                                  trees=coll_t, 
                                  y=y)


    def __call__(self, batch):
        return self.collate(batch)


def bfs_transform(x, edge_index, max_depth):
    '''
    Breadth-first visit on a torch-geometric Data object. 
    Args:
        g (torch_geometric.Data object): graph on which the visit is performed
        root (int): index of the root node
    Returns: torch.tensor with size [2, #tree_edges] of the tree's edges.
    '''
    roots = []
    edges = [[] for _ in range(max_depth-1)]
    leaves = []
    inverse_mappings = []
    trees_ind = []
    n = 0
    n_trees = 0
    
    for root in range(x.size(0)):
        inverse_mappings.append(root)
        queue = [(root, n, 0)]
        roots.append(n)
        visited = [root]
        trees_ind.append(n_trees)
        n += 1
        while queue:
            curr, mapping, level = queue.pop(0)

            leaf = True
            if level < max_depth - 1:
                low, high = binary_search(edge_index[0, :], curr)
                for idx in range(low, high):
                    if edge_index[1, idx] not in visited:
                        leaf = False
                        edges[level].append(torch.LongTensor([mapping, n]))
                        inverse_mappings.append(edge_index[1, idx])
                        queue.append((edge_index[1, idx], n, level+1))
                        visited.append(edge_index[1, idx])
                        trees_ind.append(n_trees)
                        n += 1

            if leaf:
                leaves.append(mapping)
        n_trees += 1
    
    trees = {'roots': torch.LongTensor(roots),
             'levels': [torch.stack(level, 1) for level in edges if level],
             'leaves': torch.LongTensor(leaves),
             'inv_map': torch.LongTensor(inverse_mappings),
             'trees_ind': torch.LongTensor(trees_ind),
             'dim': n
            }

    return trees

def binary_search(arr, x): 
    low = 0
    high = len(arr) - 1
    mid = 0
  
    while low <= high: 
        mid = (high + low) // 2
        if arr[mid] < x: 
            low = mid + 1
  
        elif arr[mid] > x: 
            high = mid - 1
        else:
            break

    low = mid - 1
    high = mid + 1
    while low > -1 and arr[low] == arr[mid]:
        low -= 1

    while high < len(arr) and arr[high] == arr[mid]:
        high += 1

    return low + 1, high
