import torch

def bfs(g, root):
    '''
    Breadth-first visit on a torch-geometric Data object. 
    Args:
        g (torch_geometric.Data object): graph on which the visit is performed
        root (int): index of the root node
    Returns: torch.tensor with size [2, #tree_edges] of the tree's edges.
    '''
    
    visited = [root]
    leaves = [i for i in range(len(g.x))]
    positions = [-1 for _ in range(len(g.x))]
    queue = [(root, 0)]
    edges = []
    while queue:
        curr, level = queue.pop(0)
        visited.append(curr)

        if level+1 > len(edges):
            edges.append([])
        
        leaf = True
        pos = 0
        for idx in range(g.edge_index.size(1)):
            if g.edge_index[0, idx] == curr and g.edge_index[1, idx] not in queue and g.edge_index[1, idx] not in visited:
                leaf = False
                edges[level].append(torch.unsqueeze(g.edge_index[:, idx], -1))
                positions[g.edge_index[1, idx]] = pos
                queue.append((g.edge_index[1, idx], level+1))
                visited.append(g.edge_index[1, idx])
                pos += 1

        if not leaf:
            leaves.remove(curr)

    if not edges[-1]:
        edges = edges[:-1]

    return [torch.cat(level, -1) for level in edges], torch.tensor(leaves), torch.tensor(positions)