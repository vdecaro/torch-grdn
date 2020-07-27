import torch

from torch_geometric.data import Dataset, Data, Batch

import time
import sys

class Graph2TreesLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, max_depth, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(Graph2TreesLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=TreeCollater(max_depth, follow_batch), **kwargs)


class TreeCollater(object):
    def __init__(self, max_depth, follow_batch):
        self.follow_batch = follow_batch
        self.max_depth = max_depth

    def collate(self, batch):
        x = [batch[0].x]
        y = [batch[0].y]
        trees = batch[0].trees[0]

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
            trees = g.trees[0]
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
        
        return Data(x=x, batch=batch_ind, trees=coll_t, y=y)


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
