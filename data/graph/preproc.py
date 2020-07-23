import torch

from torch_geometric.data import Dataset, Data, Batch

import time
def bfs(batch, max_depth):
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
    batch_ind = []
    n = 0
    n_trees = 0
    offset = 0
    t1 = time.time()
    for n_graph, g in enumerate(batch):
        edge_index = g.edge_index
        for root in range(g.x.size(0)):
            inverse_mappings.append(root + offset)
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
                            inverse_mappings.append(edge_index[1, idx] + offset)
                            queue.append((edge_index[1, idx], n, level+1))
                            visited.append(edge_index[1, idx])
                            trees_ind.append(n_trees)
                            n += 1

                if leaf:
                    leaves.append(mapping)
            n_trees += 1
            batch_ind.append(n_graph)
        offset += g.x.size(0)
            

    trees = {'roots': torch.LongTensor(roots),
             'levels': [torch.stack(level, 1) for level in edges],
             'leaves': torch.LongTensor(leaves),
             'inv_map': torch.LongTensor(inverse_mappings),
             'trees_ind': torch.LongTensor(trees_ind),
             'dim': n
            }
    batch_ind = torch.LongTensor(batch_ind)
    t2 = time.time()
    print(f"Preproc = {t2-t1}")
    return trees, batch_ind


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

class TreeCollater(object):
    def __init__(self, follow_batch, max_depth):
        self.follow_batch = follow_batch
        self.max_depth = max_depth

    def collate(self, batch):
        new_batch = Batch.from_data_list(batch, self.follow_batch)
        x = torch.cat([b.x for b in batch])
        y = torch.cat([b.y for b in batch])
        trees, batch_ind = bfs(batch, self.max_depth)
        new_batch = Data(x=x, batch=batch_ind, trees=trees, y=y)
        return new_batch


    def __call__(self, batch):
        return self.collate(batch)
        

class Graph2TreesLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, max_depth, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(Graph2TreesLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=TreeCollater(follow_batch, max_depth), **kwargs)