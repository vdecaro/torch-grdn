import torch
import sys

from torch_geometric.datasets import TUDataset
from data.graph.preproc import Graph2TreesLoader
from torch_geometric.data import DataLoader
from graph_htn.graph_htn import GraphHTN
from cgmn.cgmn import CGMN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import numpy as np

def nci1_transform(data):
    data.x = data.x.argmax(1)
    data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
    return data

DEVICE=sys.argv[1]
N_GEN = int(sys.argv[2])
C = int(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
EPOCHS = int(sys.argv[5])
PATIENCE = int(sys.argv[6])

dataset = TUDataset('.', 'NCI1', transform=nci1_transform)
kfold = StratifiedKFold(10, shuffle=True, random_state=15)
split = kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset]))
tr_i, vl_i = next(split)
tr_data, vl_data = dataset[tr_i.tolist()], dataset[vl_i.tolist()]

loader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)

cgmn = CGMN(1, N_GEN, C, 37, DEVICE)
bce = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(cgmn.parameters())
device = torch.device(DEVICE)
pat_cnt = 0
val_best = 500
for i in range(EPOCHS):
    loss_avg = 0
    acc_avg = 0
    n = 0
    for b in loader:
        b = b.to(device, non_blocking=True)
        out, neg_likelihood = cgmn(b.x, b.edge_index, b.batch)
        loss = bce(out, b.y)
        opt.zero_grad()
        loss.backward()
        neg_likelihood.backward()
        opt.step()

        accuracy = accuracy_score(b.y.detach().cpu().numpy(), out.detach().cpu().sigmoid().numpy().round())
        loss_avg = loss.cpu().item() if n == 0 else loss_avg + ((loss.cpu().item() - loss_avg)/(n+1))
        acc_avg = accuracy if n == 0 else acc_avg + ((accuracy - acc_avg)/(n+1))
        n += 1

    print(f"Training {i}: Loss = {loss_avg} -- Accuracy = {acc_avg}")

    for b in val_loader:
        with torch.no_grad():
            b = b.to(device, non_blocking=True)
            out, neg_likelihood = cgmn(b.x, b.edge_index, b.batch)
            loss = bce(out, b.y)
            accuracy = accuracy_score(b.y.detach().cpu().numpy(), out.detach().cpu().sigmoid().numpy().round())

    print(f"Validation {i}: Loss = {loss.item()} -- Accuracy = {accuracy}")
    if loss < val_best - 1e-2:
        val_best = loss
        pat_cnt = 0
    else:
        pat_cnt += 1
        if pat_cnt == PATIENCE:
            print(f"Appending Layer {len(cgmn.cgmm.layers)}")
            cgmn.stack_layer()
            opt = torch.optim.Adam(cgmn.parameters())
            pat_cnt = 0
