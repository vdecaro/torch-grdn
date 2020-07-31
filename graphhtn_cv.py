import sys
import os
import time

import torch
import numpy as np

from torch_geometric.datasets import TUDataset
from data.graph.preproc import Graph2TreesLoader, bfs_transform

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from graph_htn.graph_htn import GraphHTN

###################################
#        DATASET SETTING          #
###################################
def nci1_pre_transform(max_depth):
    
    def func(data):
        data['trees'] = bfs_transform(data.x, data.edge_index, max_depth)
        return data

    return func 

def nci1_transform(data):
    data.x = data.x.argmax(1)
    data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
    return data

###################################
#          CV HPARAMS             #
###################################
DEVICE = torch.device(sys.argv[1])
MAX_DEPTH = int(sys.argv[2])
M = int(sys.argv[3])
C = int(sys.argv[4])

BATCH_SIZE = 256
EPOCHS = 500
PATIENCE = 15

chk_path = f"NCI1_{MAX_DEPTH}_{M}_{C}.tar"

if os.path.exists(chk_path):
    CV_CHK = torch.load(chk_path)
else:
    CV_CHK = {
        'fold_i': 0,
        'epoch': 0,
        'best_v_loss': float('inf'),
        'model_state': None,
        'opt_state': None,
        'loss': [],
        'acc': [],
        'restore': False
    }

dataset = TUDataset(f'./NCI1_{MAX_DEPTH}', 'NCI1', pre_transform=nci1_pre_transform(MAX_DEPTH), transform=nci1_transform)

kfold = StratifiedKFold(10, shuffle=True, random_state=15)
split = list(kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

bce = torch.nn.BCEWithLogitsLoss()
for ds_i, ts_i in split[CV_CHK['fold_i']:]:
    ds_data, ts_data = dataset[ds_i.tolist()], dataset[ts_i.tolist()]
    tr_i, vl_i = train_test_split(np.arange(len(ds_data)), test_size=0.1,  stratify=np.array([g.y for g in ds_data]))
    tr_data, vl_data = ds_data[tr_i.tolist()], ds_data[vl_i.tolist()]

    tr_ld = Graph2TreesLoader(tr_data, max_depth=MAX_DEPTH, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    vl_ld = Graph2TreesLoader(vl_data, max_depth=MAX_DEPTH, batch_size=len(vl_data), shuffle=False, pin_memory=True)
    ts_ld = Graph2TreesLoader(vl_data, max_depth=MAX_DEPTH, batch_size=len(ts_data), shuffle=False, pin_memory=True)
    ghtn = GraphHTN(1, M, 0, C, 37, 8, device=DEVICE)
    opt = torch.optim.Adam(ghtn.parameters())
    if CV_CHK['restore']:
        print(f"Restarting from fold {CV_CHK['fold_i']}, epoch {CV_CHK['epoch']} with best loss {CV_CHK['best_v_loss']}")
        ghtn.load_state_dict(CV_CHK['model_state'])
        opt.load_state_dict(CV_CHK['opt_state'])
        restore=False

    pat_cnt = 0
    for i in range(CV_CHK['epoch'], EPOCHS):
        ghtn.train()
        for tr_batch in tr_ld:
            tr_batch.to(DEVICE, non_blocking=True)
            out, neg_likelihood = ghtn(tr_batch.x, tr_batch.trees, tr_batch.batch)
            tr_loss = bce(out, tr_batch.y)
            opt.zero_grad()
            tr_loss.backward()
            neg_likelihood.backward()
            opt.step()
        
        ghtn.eval()
        for vl_batch in vl_ld:
            with torch.no_grad():
                vl_batch.to(DEVICE, non_blocking=True)
                out, neg_likelihood = ghtn(vl_batch.x, vl_batch.trees, vl_batch.batch)
                vl_loss = bce(out, vl_batch.y)
                vl_accuracy = accuracy(vl_batch.y, out.sigmoid().round())
        print(f"Fold {CV_CHK['fold_i']} - Epoch {i}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")
        
        CV_CHK['epoch'] += 1
        if vl_loss.item() < CV_CHK['best_v_loss'] - 1e-2:
            CV_CHK['best_v_loss'] = vl_loss.item()
            CV_CHK['model_state'] = ghtn.state_dict()
            CV_CHK['opt_state'] = opt.state_dict()
            CV_CHK['restore'] = True
            torch.save(CV_CHK, chk_path)
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt == PATIENCE:
                print("Patience over: training stopped.")
                break
        
    ghtn.load_state_dict(CV_CHK['model_state'])
    for ts_batch in ts_ld:
        with torch.no_grad():
            ts_batch.to(DEVICE, non_blocking=True)
            out, neg_likelihood = ghtn(ts_batch.x, ts_batch.trees, ts_batch.batch)
            ts_loss = bce(out, ts_batch.y)
            ts_acc = accuracy(ts_batch.y, out.sigmoid().round())
    print(f"Fold {CV_CHK['fold_i']}: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

    CV_CHK['loss'].append(ts_loss.item())
    CV_CHK['acc'].append(ts_acc.item())
    CV_CHK['fold_i'] += 1
    CV_CHK['epoch'] = 0
    CV_CHK['best_v_loss'] = float('inf')
    CV_CHK['model_state'] = None
    CV_CHK['opt_state'] = None
    CV_CHK['restore'] = False
    torch.save(CV_CHK, chk_path)
