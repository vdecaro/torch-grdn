from torch.utils.data import Dataset, DataLoader


def GraphLoader(tree_dataset, batch_size, shuffle=False):
    
    def collate_graphs_fn(batch):
        graphs, targets = [], []
        for graph, target in batch:
            graphs.append(graph)
            targets.append(target)

        return graphs, targets
    
    return DataLoader(tree_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graphs_fn)
