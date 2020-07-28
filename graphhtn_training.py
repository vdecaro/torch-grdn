import sys

import torch
from torch_geometric.datasets import TUDataset
from data.graph.preproc import Graph2TreesLoader, bfs_transform
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

from graph_htn.graph_htn import GraphHTN

def nci1_pre_transform(max_depth):
    
    def func(data):
        data['trees'] = bfs_transform(data.x, data.edge_index, max_depth)
        return data

    return func 

def nci1_transform(data):
    data.x = data.x.argmax(1)
    data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
    return data

DEVICE=sys.argv[1]
M = int(sys.argv[2])
C = int(sys.argv[3])
MAX_DEPTH = 5
dataset = TUDataset(f'./NCI1_{MAX_DEPTH}', 'NCI1', pre_transform=nci1_pre_transform(MAX_DEPTH), transform=nci1_transform)

kfold = StratifiedKFold(10, shuffle=True, random_state=15)
split = kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset]))
tr_i, vl_i = next(split)
tr_data, vl_data = dataset[tr_i.tolist()], dataset[vl_i.tolist()]
loader = Graph2TreesLoader(tr_data, max_depth=MAX_DEPTH, batch_size=64, shuffle=True, device=torch.device(DEVICE))
val_loader = Graph2TreesLoader(vl_data, max_depth=MAX_DEPTH, batch_size=len(vl_data), shuffle=False, device=torch.device(DEVICE))

ghtn = GraphHTN(1, M, 0, C, 37, 8, device=DEVICE)
bce = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(ghtn.get_parameters())

for i in range(200):
    loss_avg = 0
    acc_avg = 0
    n = 0
    
    for b in loader:
        out, neg_likelihood = ghtn(b.x, b.trees, b.batch)
        loss = bce(out, b.y)
        opt.zero_grad()
        loss.backward()
        neg_likelihood.backward()
        opt.step()

        accuracy = accuracy_score(b.y.detach().cpu().numpy(), out.detach().cpu().sigmoid().numpy().round())
        loss_avg = loss.cpu().item() if n == 0 else loss_avg + ((loss.cpu().item() - loss_avg)/(n+1))
        acc_avg = accuracy if n == 0 else acc_avg + ((accuracy - acc_avg)/(n+1))
        n += 1
        
        #print(f"Loss = {loss.cpu().item()} ----- Accuracy = {accuracy}")
    print(f"Training avg {i}: Loss = {loss_avg} --  Accuracy = {acc_avg}")

    for b in val_loader:
        with torch.no_grad():
            out, neg_likelihood = ghtn(b.x, b.trees, b.batch)
            loss = bce(out, b.y)
            accuracy = accuracy_score(b.y.detach().cpu().numpy(), out.detach().cpu().sigmoid().numpy().round())

        
    print(f"Validation {i}: Loss = {loss.item()} -- Accuracy = {accuracy}")