import sys
import os
import time

import torch
import numpy as np

from torch_geometric.datasets import TUDataset
from data.graph.g2t import Graph2TreesLoader, bfs_transform

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from graph_htmn.graph_htmn import GraphHTMN

###################################
#        DATASET SETTING          #
###################################
def pre_transform(max_depth):
    
    def func(data):
        data['trees'] = bfs_transform(data.x, data.edge_index, max_depth)
        return data

    return func 

def transform(dataset):
    if dataset in ['NCI1', 'PROTEINS', 'DD']:
        def func(data):
            data.x = data.x.argmax(1)
            data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
            return data
    
    return func


###################################
#          CV HPARAMS             #
###################################
DEVICE = torch.device(sys.argv[1])
DATASET = sys.argv[2]
MAX_DEPTH = int(sys.argv[3])
M = int(sys.argv[4])
C = int(sys.argv[5])
lr = float(sys.argv[6])
#l2 = float(sys.argv[7])

_R_STATE = 42
BATCH_SIZE = 128
EPOCHS = 5000
PATIENCE = 20

if DATASET == 'NCI1':
    N_SYMBOLS = 37
elif DATASET == 'PROTEINS':
    N_SYMBOLS = 3
elif DATASET == 'DD':
    N_SYMBOLS = 89
    BATCH_SIZE = 32

chk_path = f"GHTN_CV/{DATASET}_{MAX_DEPTH}_{M}_{C}.tar"

if os.path.exists(chk_path):
    CHK = torch.load(chk_path)
else:
    CHK = {
        'CV': {
            'fold': 0,
            'epoch': 0,
            'pat': 0,
            'v_loss': float('+inf'),
            'f_v_loss': [],
            'loss': [],
            'acc': [],
        },
        'MOD': None,
        'OPT': None
    }

dataset = TUDataset(f'./{DATASET}_{MAX_DEPTH}', DATASET, pre_transform=pre_transform(MAX_DEPTH), transform=transform(DATASET))
kfold = StratifiedKFold(10, shuffle=True, random_state=_R_STATE)
split = list(kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

bce = torch.nn.BCEWithLogitsLoss()
for ds_i, ts_i in split[CHK['CV']['fold']:]:
    ds_data, ts_data = dataset[ds_i.tolist()], dataset[ts_i.tolist()]
    tr_i, vl_i = train_test_split(np.arange(len(ds_data)), 
                                  test_size=0.1,  
                                  stratify=np.array([g.y for g in ds_data]), 
                                  shuffle=True, 
                                  random_state=_R_STATE)
    tr_data, vl_data = ds_data[tr_i.tolist()], ds_data[vl_i.tolist()]

    tr_ld = Graph2TreesLoader(tr_data, max_depth=MAX_DEPTH, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, drop_last=True)
    vl_ld = Graph2TreesLoader(vl_data, max_depth=MAX_DEPTH, batch_size=len(vl_data), shuffle=False, pin_memory=False)
    ts_ld = Graph2TreesLoader(ts_data, max_depth=MAX_DEPTH, batch_size=len(ts_data), shuffle=False, pin_memory=False)

    ghtn = GraphHTN(1, M, 0, C, N_SYMBOLS, 8, device=DEVICE)
    opt = torch.optim.Adam(ghtn.parameters(), lr=lr)
    if CHK['OPT'] is not None:
        print(f"Restarting from fold {CHK['CV']['fold']}, epoch {CHK['CV']['epoch']} with best loss {CHK['CV']['v_loss']}")
        ghtn.load_state_dict(CHK['MOD'])
        opt.load_state_dict(CHK['OPT'])

    for i in range(CHK['CV']['epoch'], EPOCHS):
        ghtn.train()
        for tr_batch in tr_ld:
            tr_batch.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            out = ghtn(tr_batch.x, tr_batch.trees, tr_batch.batch)
            tr_loss = bce(out, tr_batch.y)
            tr_loss.backward()
            opt.step()

        ghtn.eval()
        for vl_batch in vl_ld:
            with torch.no_grad():
                vl_batch.to(DEVICE, non_blocking=True)
                out = ghtn(vl_batch.x, vl_batch.trees, vl_batch.batch)
                vl_loss = bce(out, vl_batch.y)
                vl_accuracy = accuracy(vl_batch.y, out.sigmoid().round())
        print(f"Fold {CHK['CV']['fold']} - Epoch {i}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")
        
        CHK['CV']['epoch'] += 1
        if  vl_loss < CHK['CV']['v_loss']:
            CHK['CV']['v_loss'] = vl_loss
            CHK['CV']['pat'] = 0
            CHK['MOD'] = ghtn.state_dict()
            CHK['OPT'] = opt.state_dict()
            torch.save(CHK, chk_path)
        else:
            CHK['CV']['pat'] += 1
            if CHK['CV']['pat'] == PATIENCE:
                print("Patience over: training stopped.")
                break

    ghtn.load_state_dict(CHK['MOD'])
    for ts_batch in ts_ld:
        with torch.no_grad():
            ts_batch.to(DEVICE, non_blocking=True)
            out = ghtn(ts_batch.x, ts_batch.trees, ts_batch.batch)
            ts_loss = bce(out, ts_batch.y)
            ts_acc = accuracy(ts_batch.y, out.sigmoid().round())
    print(f"Fold {CHK['CV']['fold']}: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")
    
    CHK['CV']['f_v_loss'].append(CHK['CV']['v_loss'])
    CHK['CV']['acc'].append(ts_acc)
    CHK['CV']['fold'] += 1
    CHK['CV']['epoch'] = 0
    CHK['CV']['pat'] = 0
    CHK['CV']['v_loss'] = float('+inf')
    CHK['MOD'] = None
    CHK['OPT'] = None
    torch.save(CHK, chk_path)
