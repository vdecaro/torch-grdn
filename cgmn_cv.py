import sys
import os
import time

import torch
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from cgmn.cgmn import CGMN

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
GATE_UNITS = int(sys.argv[6])
lr = float(sys.argv[7])

if DATASET == 'NCI1':
    N_SYMBOLS = 37
elif DATASET == 'PROTEINS':
    N_SYMBOLS = 3
elif DATASET == 'DD':
    N_SYMBOLS = 89

_R_STATE = 42
BATCH_SIZE = 128
EPOCHS = 5000
PATIENCE = 30

chk_path = f"CGMN_CV/{DATASET}_{MAX_DEPTH}_{M}_{C}_{GATE_UNITS}.tar"

if os.path.exists(chk_path):
    CHK = torch.load(chk_path)
    print(f"Restarting from fold {CHK['CV']['fold']}, epoch {CHK['CV']['epoch']} with curr best loss {CHK['CV']['v_loss']} and abs best loss {CHK['CV']['abs_v_loss']}")
else:
    CHK = {
        'CV': {
            'fold': 0,
            'epoch': 0,
            'abs_v_loss': float('inf'),
            'v_loss': float('inf'),
            'abs_v_acc': -float('inf'),
            'v_acc': -float('inf'),
            'pat': [0, float('inf')],
            'f_v_loss': [],
            'loss': [],
            'acc': [],
            'layer_loss': [[] for _ in range(10)],
        },
        'MOD': {
            'best': {
                'L': 1,
                'state': None
            },
            'curr': {
                'L': 1,
                'state': None
            },
        },
        'OPT': None
    }
preproc_from_layer = 0

dataset = TUDataset(f'./{DATASET}', DATASET, transform=transform(DATASET))
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
    
    tr_ld = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)
    ts_ld = DataLoader(ts_data, batch_size=len(ts_data), shuffle=False, pin_memory=True)
    while CHK['MOD']['curr']['L'] < MAX_DEPTH:
        cgmn = CGMN(1, M, C, None, N_SYMBOLS, gate_units=GATE_UNITS, device=DEVICE)
        for _ in range(len(cgmn.cgmm.layers), CHK['MOD']['curr']['L']):
            cgmn.stack_layer()

        if CHK['MOD']['curr']['state'] is not None:
            cgmn.load_state_dict(CHK['MOD']['curr']['state'])

            # This condition checks whether the current state of the CGMN is fully trained
            if CHK['OPT'] is None:
                if CHK['MOD']['curr']['L'] + 1 > MAX_DEPTH:
                    break
                else:
                    cgmn.stack_layer()
                    CHK['MOD']['curr']['state'] = cgmn.state_dict()
                    CHK['MOD']['curr']['L'] += 1
        
        opt = torch.optim.Adam(cgmn.parameters(), lr=lr)
        if CHK['OPT'] is not None:
            opt.load_state_dict(CHK['OPT'])                
        torch.save(CHK, chk_path)
        for i in range(CHK['CV']['epoch'], EPOCHS):
            cgmn.train()
            for tr_batch in tr_ld:
                tr_batch = tr_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else tr_batch
                opt.zero_grad()
                out = cgmn(tr_batch.x, tr_batch.edge_index, tr_batch.batch)
                tr_loss = bce(out, tr_batch.y)
                tr_loss.backward()
                opt.step()

            cgmn.eval()
            for vl_batch in vl_ld:
                with torch.no_grad():
                    vl_batch = vl_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else vl_batch
                    out = cgmn(vl_batch.x, vl_batch.edge_index, vl_batch.batch)
                    vl_loss = bce(out, vl_batch.y)
                    vl_accuracy = accuracy(vl_batch.y, out.sigmoid().round())
            print(f"Fold {CHK['CV']['fold']} - Layer {CHK['MOD']['curr']['L']} - Epoch {i}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")
            
            CHK['CV']['epoch'] += 1
            if vl_accuracy > CHK['CV']['v_acc'] or (vl_accuracy == CHK['CV']['v_acc'] and vl_loss.item() < CHK['CV']['v_loss']):
                CHK['CV']['v_acc'] = vl_accuracy
                CHK['CV']['v_loss'] = vl_loss.item()
                CHK['MOD']['curr']['state'] = cgmn.state_dict()
                CHK['OPT'] = opt.state_dict()
                if  vl_accuracy > CHK['CV']['abs_v_acc'] or (vl_accuracy == CHK['CV']['abs_v_acc'] and vl_loss.item() < CHK['CV']['abs_v_loss']):
                    CHK['CV']['abs_v_acc'] = vl_accuracy
                    CHK['CV']['abs_v_loss'] = vl_loss.item()
                    CHK['MOD']['best']['state'] = cgmn.state_dict()
                    CHK['MOD']['best']['L'] = CHK['MOD']['curr']['L']
                
            if vl_loss.item() < CHK['CV']['pat'][1]:
                CHK['CV']['pat'][0] = 0
                CHK['CV']['pat'][1] = vl_loss.item()
            else:
                CHK['CV']['pat'][0] += 1
                if CHK['CV']['pat'][0] >= PATIENCE:
                    CHK['CV']['layer_loss'][CHK['CV']['fold']].append(CHK['CV']['v_loss'])
                    print("Patience over: training stopped.")
                    break
            torch.save(CHK, chk_path)
            
                    
        CHK['CV']['epoch'] = 0
        CHK['CV']['v_acc'] = -float('inf')
        CHK['CV']['v_loss'] = float('inf')
        CHK['OPT'] = None
        CHK['CV']['pat'] = [0, float('inf')]
        torch.save(CHK, chk_path)

    # TESTING
    cgmn = CGMN(1, M, C, None, N_SYMBOLS, gate_units=GATE_UNITS, device=DEVICE)
    for _ in range(len(cgmn.cgmm.layers), CHK['MOD']['best']['L']):
        cgmn.stack_layer()
    cgmn.load_state_dict(CHK['MOD']['best']['state'])
    cgmn.eval()
    for ts_batch in ts_ld:
        with torch.no_grad():
            ts_batch = ts_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else ts_batch
            out = cgmn(ts_batch.x, ts_batch.edge_index, ts_batch.batch)
            ts_loss = bce(out, ts_batch.y)
            ts_acc = accuracy(ts_batch.y, out.sigmoid().round())
    print(f"Fold {CHK['CV']['fold']}: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

    CHK['CV']['f_v_loss'].append(CHK['CV']['abs_v_loss'])
    CHK['CV']['loss'].append(ts_loss.item())
    CHK['CV']['acc'].append(ts_acc)
    CHK['CV']['fold'] += 1
    CHK['CV']['epoch'] = 0
    CHK['CV']['pat'] = [0, float('inf')]
    CHK['CV']['abs_v_loss'] = float('inf')
    CHK['CV']['v_loss'] = float('inf')
    CHK['CV']['abs_v_acc'] = -float('inf')
    CHK['CV']['v_acc'] = -float('inf')
    CHK['MOD'] = {
        'best': {
            'L': 1,
            'state': None
        },
        'curr': {
            'L': 1,
            'state': None
        },
    }
    CHK['OPT'] = None
    torch.save(CHK, chk_path)
