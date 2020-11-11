import sys
import os
import time

import torch
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from cgmn.cgmn import CGMN

def pre_transform(dataset):
    if dataset in ['NCI1', 'PROTEINS', 'DD']:
        def func(data):
            data.x = data.x.argmax(1, keepdims=True)
            return Data(x=data.x.argmax(1, keepdims=True), edge_index=data.edge_index, y=data.y)
    return func

def transform(dataset):
    
    if dataset in ['NCI1', 'PROTEINS', 'DD']:
        def func(data):
            data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
            return data
    
    return func

###################################
#          CV HPARAMS             #
###################################
DEVICE = torch.device(sys.argv[1])
DATASET = sys.argv[2]
M = int(sys.argv[3])
C = int(sys.argv[4])
GATE_UNITS = int(sys.argv[5])
lr = float(sys.argv[6])

if DATASET == 'NCI1':
    N_SYMBOLS = 37
    PATIENCE = 30
    L_PATIENCE = 2
elif DATASET == 'PROTEINS':
    N_SYMBOLS = 3
    PATIENCE = 50
    L_PATIENCE = 2
elif DATASET == 'DD':
    N_SYMBOLS = 89
    PATIENCE = 40
    L_PATIENCE = 3
    
_R_STATE = [42, 15, 11, 59, 10, 92, 95, 320, 666, 280]
BATCH_SIZE = 100 if DATASET == 'DD' else 128
EPOCHS = 5000
MAX_DEPTH = 20

chk_path = f"CGMN_CV/{DATASET}_{M}_{C}_{GATE_UNITS}.tar"

if os.path.exists(chk_path):
    CHK = torch.load(chk_path)
    print(f"Restarting from fold {CHK['CV'][CHK['IDX']]['fold']}, epoch {CHK['CV'][CHK['IDX']]['epoch']} with curr best loss {CHK['CV'][CHK['IDX']]['v_loss']} and abs best loss {CHK['CV'][CHK['IDX']]['abs_v_loss']}")
else:
    CHK = {'CV': [{
                'fold': 0,
                'epoch': 0,
                'abs_v_loss': float('inf'),
                'v_loss': float('inf'),
                'abs_v_acc': -float('inf'),
                'v_acc': -float('inf'),
                'pat': [0, float('inf')],
                'l_pat': 0,
                'f_v_loss': [],
                'loss': [],
                'acc': [],
                'layer_loss': [[] for _ in range(10)],
            } for _ in range(10)],
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
            'OPT': None,
            'IDX': 0
        }

dataset = TUDataset(f'.', DATASET, transform=transform(DATASET))
dataset.data.x = dataset.data.x.argmax(1).detach()
for idx in range(CHK['IDX'], 10):
    CV = CHK['CV'][idx]
    kfold = StratifiedKFold(10, shuffle=True, random_state=_R_STATE[idx])
    split = list(kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

    bce = torch.nn.BCEWithLogitsLoss()

    for ds_i, ts_i in split[CV['fold']:]:
        ds_data, ts_data = dataset[ds_i.tolist()], dataset[ts_i.tolist()]
        tr_i, vl_i = train_test_split(np.arange(len(ds_data)), 
                                      test_size=0.1,  
                                      stratify=np.array([g.y for g in ds_data]), 
                                      shuffle=True, 
                                      random_state=_R_STATE[idx])
        tr_data, vl_data = ds_data[tr_i.tolist()], ds_data[vl_i.tolist()]

        tr_ld = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
        vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)
        ts_ld = DataLoader(ts_data, batch_size=len(ts_data), shuffle=False, pin_memory=False)

        while CHK['MOD']['curr']['L'] < MAX_DEPTH and CV['l_pat'] <= L_PATIENCE:
            cgmn = CGMN(1, M, C, None, N_SYMBOLS, gate_units=GATE_UNITS, device=DEVICE)
            for _ in range(len(cgmn.cgmm.layers), CHK['MOD']['curr']['L']):
                cgmn.stack_layer()

            if CHK['MOD']['curr']['state'] is not None:
                cgmn.load_state_dict(CHK['MOD']['curr']['state'])

                # This condition checks whether the current state of the CGMN is fully trained
                if CHK['OPT'] is None:
                    cgmn.stack_layer()
                    CHK['MOD']['curr']['state'] = cgmn.state_dict()
                    CHK['MOD']['curr']['L'] += 1

            opt = torch.optim.Adam(cgmn.parameters(), lr=lr)
            if CHK['OPT'] is not None:
                opt.load_state_dict(CHK['OPT'])                
            torch.save(CHK, chk_path)
            for i in range(CV['epoch'], EPOCHS):
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
                        vl_loss = bce(out, vl_batch.y).item()
                        vl_accuracy = accuracy(vl_batch.y, out.sigmoid().round())
                print(f"CV {idx} - Fold {CV['fold']} - Layer {CHK['MOD']['curr']['L']} - Epoch {i}: Loss = {vl_loss} ---- Accuracy = {vl_accuracy}")
                CV['epoch'] += 1
                if vl_loss < CV['v_loss']:
                    CV['v_acc'] = vl_accuracy
                    CV['v_loss'] = vl_loss
                    CHK['MOD']['curr']['state'] = cgmn.state_dict()
                    CHK['OPT'] = opt.state_dict()
                    if  vl_loss < CV['abs_v_loss']:
                        CV['abs_v_acc'] = vl_accuracy
                        CV['abs_v_loss'] = vl_loss
                        CHK['MOD']['best']['state'] = cgmn.state_dict()
                        CHK['MOD']['best']['L'] = CHK['MOD']['curr']['L']
                        CV['l_pat'] = 0

                if vl_loss < CV['pat'][1]:
                    CV['pat'][0] = 0
                    CV['pat'][1] = vl_loss
                else:
                    CV['pat'][0] += 1
                    if CV['pat'][0] >= PATIENCE:
                        CV['layer_loss'][CV['fold']].append(CV['v_loss'])
                        print("Patience over: training stopped.")
                        break
                torch.save(CHK, chk_path)
            CV['l_pat'] += 1
            CV['epoch'] = 0
            CV['v_acc'] = -float('inf')
            CV['v_loss'] = float('inf')
            CHK['OPT'] = None
            CV['pat'] = [0, float('inf')]
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
        print(f"Fold {CV['fold']}: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

        CV['f_v_loss'].append(CV['abs_v_loss'])
        CV['loss'].append(ts_loss.item())
        CV['acc'].append(ts_acc)
        CV['fold'] += 1
        CV['epoch'] = 0
        CV['pat'] = [0, float('inf')]
        CV['l_pat'] = 0
        CV['abs_v_loss'] = float('inf')
        CV['v_loss'] = float('inf')
        CV['abs_v_acc'] = -float('inf')
        CV['v_acc'] = -float('inf')
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
    CHK['IDX'] += 1
    torch.save(CHK, chk_path)
