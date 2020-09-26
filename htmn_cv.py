import sys
import os
import time

import torch
import torch.nn.functional as F
import numpy as np

from data.tree.loader import TreeDataset, TreesLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.utils.metric import accuracy

from htmn.htmn import HTMN

def batch_to(b, device):
    b['x'] = b['x'].to(device)
    b['levels'] = [l.to(device) for l in b['levels']]
    b['leaves'] = b['leaves'].to(device)
    b['pos'] = b['pos'].to(device)
    b['y'] = b['y'].to(device)
    b['batch'] = b['batch'].to(device)

###################################
#          CV HPARAMS             #
###################################
DEVICE = torch.device(sys.argv[1])
DATASET = sys.argv[2]
M = int(sys.argv[3])
C = int(sys.argv[4])
if sys.argv[5] not in ['T', 'F']:
    sys.exit()
B_NORM = sys.argv[5] == 'T'
lr = float(sys.argv[6])

_R_STATE = 42
if DATASET == 'inex2005':
    N_SYMBOLS = 366
    L = 32
    CLASSES = 11
    BATCH_SIZE = 64
    PATIENCE = 10
if DATASET == 'inex2006':
    N_SYMBOLS = 65
    L = 66
    CLASSES = 18
    BATCH_SIZE = 128
    PATIENCE = 15
EPOCHS = 2000


chk_path = f"HTMN_CV/{DATASET}_{M}_{C}_{sys.argv[5]}.tar"

if os.path.exists(chk_path):
    CHK = torch.load(chk_path)
else:
    CHK = {
        'CV': {
            'epoch': 0,
            'pat': 0,
            'v_loss': float('+inf'),
            't_acc': None
        },
        'MOD': None,
        'OPT': None
    }

ds_data, ts_data = TreeDataset(DATASET + 'train'), TreeDataset(DATASET + 'test')

tr_i, vl_i = train_test_split(np.arange(len(ds_data)), 
                              test_size=0.2,  
                              stratify=np.array([t.y for t in ds_data]), 
                              shuffle=True, 
                              random_state=_R_STATE)

tr_data = TreeDataset(data=[ds_data[i] for i in tr_i.tolist()])
vl_data = TreeDataset(data=[ds_data[i] for i in vl_i.tolist()])

tr_ld = TreesLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
vl_ld = TreesLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)

htmn = HTMN(CLASSES, M//2, M//2, C, L, N_SYMBOLS, B_NORM, device=DEVICE)
opt = torch.optim.Adam(htmn.parameters(), lr=lr)
ce_loss = torch.nn.CrossEntropyLoss()
if CHK['OPT'] is not None:
    print(f"Restarting from epoch {CHK['CV']['epoch']} with best loss {CHK['CV']['v_loss']}")
    htmn.load_state_dict(CHK['MOD'])
    opt.load_state_dict(CHK['OPT'])

if CHK['CV']['pat'] < PATIENCE:
    for i in range(CHK['CV']['epoch'], EPOCHS):
        htmn.train()
        for tr_batch in tr_ld:
            batch_to(tr_batch, DEVICE)
            opt.zero_grad()
            out = htmn(tr_batch)
            tr_loss = ce_loss(out, tr_batch.y[:, 0])
            tr_loss.backward()
            opt.step()
            tr_accuracy = accuracy(tr_batch.y[:, 0], out.argmax(1))
            print(f"Training: Loss = {tr_loss.item()} ---- Accuracy = {tr_accuracy}")

        htmn.eval()
        for vl_batch in vl_ld:
            with torch.no_grad():
                batch_to(vl_batch, DEVICE)
                out = htmn(vl_batch)
                vl_loss = ce_loss(out, vl_batch.y[:, 0])
                vl_accuracy = accuracy(vl_batch.y[:, 0], out.argmax(1))
        print(f"\n\n Epoch {i}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")

        CHK['CV']['epoch'] += 1
        if  vl_loss < CHK['CV']['v_loss']:
            CHK['CV']['v_loss'] = vl_loss
            CHK['CV']['pat'] = 0
            CHK['MOD'] = htmn.state_dict()
            CHK['OPT'] = opt.state_dict()
            torch.save(CHK, chk_path)
        else:
            CHK['CV']['pat'] += 1
            if CHK['CV']['pat'] == PATIENCE:
                print("Patience over: training stopped.")
                break

ts_ld = TreesLoader(ts_data, batch_size=len(ts_data), shuffle=False)
htmn.load_state_dict(CHK['MOD'])
for ts_batch in ts_ld:
    with torch.no_grad():
        batch_to(ts_batch, DEVICE)
        out = htmn(ts_batch)
        ts_loss = ce_loss(out, ts_batch.y[:, 0])
        ts_acc = accuracy(ts_batch.y[:, 0], out.argmax(1))
print(f"Test: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

CHK['CV']['t_acc'] = ts_acc
torch.save(CHK, chk_path)
