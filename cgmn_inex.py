import sys
import os
import time
import random

import torch
import numpy as np

from data.graph.inex.preproc import load_and_preproc_inex
from torch_geometric.data import Batch

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from cgmn.cgmn import CGMN

DEVICE = torch.device(sys.argv[1])
DATASET = sys.argv[2]
MAX_DEPTH = int(sys.argv[3])
M = int(sys.argv[4])
C = int(sys.argv[5])
lr = float(sys.argv[6])

_R_STATE = 42
if DATASET == 'inex2005':
    N_SYMBOLS = 366
    CLASSES = 11
    L = 32
    BATCH_SIZE = 128
    PATIENCE = 30
if DATASET == 'inex2006':
    N_SYMBOLS = 65
    CLASSES = 18
    L = 66
    BATCH_SIZE = 128
    PATIENCE = 40
EPOCHS = 2000

chk_path = f"CGMN_CV/{DATASET}_{MAX_DEPTH}_{M}_{C}.tar"

if os.path.exists(chk_path):
    CHK = torch.load(chk_path)
    print(f"Restarting from epoch {CHK['CV']['epoch']} with curr best loss {CHK['CV']['v_loss']} and abs best loss {CHK['CV']['abs_v_loss']}")
else:
    CHK = {
        'CV': {
            'epoch': 0,
            'pat': 0,
            'v_loss': float('+inf'),
            'abs_v_loss': float('+inf'),
            't_acc': None,
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

ds_data, ts_data = load_and_preproc_inex(DATASET + 'train'), load_and_preproc_inex(DATASET + 'test')

tr_i, vl_i = train_test_split(np.arange(len(ds_data)), 
                              test_size=0.2,  
                              stratify=np.array([t.y for t in ds_data]), 
                              shuffle=True, 
                              random_state=_R_STATE)

tr_data = [ds_data[i] for i in tr_i.tolist()]
vl_data = [ds_data[i] for i in vl_i.tolist()]

ce_loss = torch.nn.CrossEntropyLoss()

while CHK['MOD']['curr']['L'] < MAX_DEPTH:
    cgmn = CGMN(CLASSES, M, C, L, N_SYMBOLS, device=DEVICE)
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

    for e in range(CHK['CV']['epoch'], EPOCHS):
        random.shuffle(tr_data)
        cgmn.train()
        for i in range(0, len(tr_data), BATCH_SIZE):
            tr_batch = Batch.from_data_list(tr_data[i:min(i+BATCH_SIZE, len(tr_data))])
            tr_batch.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            out = cgmn(tr_batch.x, tr_batch.edge_index, tr_batch.batch, tr_batch.pos)
            tr_loss = ce_loss(out, tr_batch.y)
            tr_loss.backward()
            opt.step()
            tr_accuracy = accuracy(tr_batch.y, out.argmax(1))

        cgmn.eval()
        with torch.no_grad():
            vl_batch = Batch.from_data_list(vl_data)
            vl_batch.to(DEVICE, non_blocking=True)
            out = cgmn(vl_batch.x, vl_batch.edge_index, vl_batch.batch, vl_batch.pos)
            vl_loss = ce_loss(out, vl_batch.y)
            vl_accuracy = accuracy(vl_batch.y, out.argmax(1))
        print(f"Layer {CHK['MOD']['curr']['L']} - Epoch {e}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")

        CHK['CV']['epoch'] += 1
        if vl_loss.item() < CHK['CV']['v_loss']:
            CHK['CV']['v_loss'] = vl_loss.item()
            CHK['MOD']['curr']['state'] = cgmn.state_dict()
            CHK['OPT'] = opt.state_dict()
            if vl_loss.item() < CHK['CV']['abs_v_loss']:
                CHK['CV']['abs_v_loss'] = vl_loss.item()
                CHK['MOD']['best']['state'] = cgmn.state_dict()
                CHK['MOD']['best']['L'] = CHK['MOD']['curr']['L']
            CHK['CV']['pat'] = 0
            torch.save(CHK, chk_path)
            
        else:
            CHK['CV']['pat'] += 1
            if CHK['CV']['pat'] == PATIENCE:
                print("Patience over: training stopped.")
                break
        
    CHK['CV']['epoch'] = 0
    CHK['CV']['v_loss'] = float('inf')
    CHK['OPT'] = None
    CHK['CV']['pat'] = 0
    torch.save(CHK, chk_path)

cgmn = CGMN(CLASSES, M, C, L, N_SYMBOLS, device=DEVICE)
for _ in range(len(cgmn.cgmm.layers), CHK['MOD']['best']['L']):
    cgmn.stack_layer()
cgmn.load_state_dict(CHK['MOD']['best']['state'])
cgmn.eval()
with torch.no_grad():
    ts_batch = Batch.from_data_list(ts_data)
    ts_batch.to(DEVICE, non_blocking=True)
    out = cgmn(ts_batch.x, ts_batch.edge_index, ts_batch.batch, ts_batch.pos)
    ts_loss = ce_loss(out, ts_batch.y)
    ts_acc = accuracy(ts_batch.y, out.argmax(1))
print(f"Test: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

CHK['CV']['t_acc'] = ts_acc
torch.save(CHK, chk_path)
