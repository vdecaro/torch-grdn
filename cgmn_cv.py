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
lr = float(sys.argv[5])
l2 = float(sys.argv[6])

BATCH_SIZE = 128
EPOCHS = 500
PATIENCE = 20

chk_path = f"CV_CGMN_NCI1_{MAX_DEPTH}_{M}_{C}_chk.tar"

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
            'loss': [],
            'acc': []
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

dataset = TUDataset(f'./NCI1', 'NCI1', transform=nci1_transform)

kfold = StratifiedKFold(10, shuffle=True, random_state=15)
split = list(kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

bce = torch.nn.BCEWithLogitsLoss()

for ds_i, ts_i in split[CHK['CV']['fold']:]:
    ds_data, ts_data = dataset[ds_i.tolist()], dataset[ts_i.tolist()]
    tr_i, vl_i = train_test_split(np.arange(len(ds_data)), test_size=0.1, random_state=15, stratify=np.array([g.y for g in ds_data]))
    tr_data, vl_data = ds_data[tr_i.tolist()], ds_data[vl_i.tolist()]

    tr_ld = DataLoader(tr_data,batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)
    ts_ld = DataLoader(ts_data,batch_size=len(ts_data), shuffle=False, pin_memory=True)

    restore_ld = DataLoader(ds_data, batch_size=2048)
    while True:
        cgmn = CGMN(1, M, C, 37, device=DEVICE)
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
                    CHK['MOD']['curr']['L'] += 1
            if preproc_from_layer < CHK['MOD']['curr']['L'] - 1:
                lhood, h_v, h_i = [], [], []
                for b in restore_ld:
                    b = b.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else b
                    if preproc_from_layer > 0:
                        b_lhood, b_h_v, b_h_i = cgmn.cgmm(b.x, b.edge_index, (b.h_v, b.h_i), preproc_from_layer, True)
                    else:
                        b_lhood, b_h_v, b_h_i = cgmn.cgmm(b.x, b.edge_index, from_layer=0, no_last=True)
                    lhood.append(b_lhood)
                    h_v.append(b_h_v)
                    h_i.append(b_h_i)
                lhood = torch.cat(lhood, dim=0)
                h_v = torch.cat(h_v, dim=0)
                h_i = torch.cat(h_i, dim=0)
                for i, d in enumerate(dataset):
                    if preproc_from_layer > 0:
                        d.likelihood = torch.cat([d.likelihood, lhood[i]], dim=0)
                        d.h_v = torch.cat([d.h_v, h_v[i]], dim=0)
                        d.h_i = torch.cat([d.h_i, h_i[i]], dim=0)
                    else:
                        d.likelihood = lhood[i]
                        d.h_v = h_v[i]
                        d.h_i = h_i[i]
                preproc_from_layer = CHK['MOD']['curr']['L'] - 1

        opt = torch.optim.AdamW(cgmn.parameters(), lr=lr, weight_decay=l2)
        if CHK['OPT'] is not None:
            opt.load_state_dict(CHK['OPT'])

        print(f"Training of layer {CHK['MOD']['curr']['L']}")
        pat_cnt = 0
        for i in range(CHK['CV']['epoch'], EPOCHS):
            cgmn.train()
            for tr_batch in tr_ld:
                tr_batch = tr_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else tr_batch
                b_h_prev = (tr_batch.h_v, tr_batch.h_i) if preproc_from_layer > 0 else None
                b_lhood_prev = tr_batch.likelihood if preproc_from_layer > 0 else None
                out, neg_likelihood = cgmn(tr_batch.x, tr_batch.edge_index, tr_batch.batch, b_h_prev, b_lhood_prev)
                tr_loss = bce(out, tr_batch.y)
                opt.zero_grad()
                tr_loss.backward()
                neg_likelihood.backward()
                opt.step()

            cgmn.eval()
            for vl_batch in vl_ld:
                with torch.no_grad():
                    vl_batch = vl_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else vl_batch
                    out, neg_likelihood = cgmn(vl_batch.x, vl_batch.edge_index, vl_batch.batch)
                    vl_loss = bce(out, vl_batch.y)
                    vl_accuracy = accuracy(vl_batch.y, out.sigmoid().round())
            print(f"Fold {CHK['CV']['fold']} - Epoch {i}: Loss = {vl_loss.item()} ---- Accuracy = {vl_accuracy}")
            
            CHK['CV']['epoch'] += 1
            if vl_loss.item() < CHK['CV']['v_loss']:
                CHK['CV']['v_loss'] = vl_loss.item()
                CHK['MOD']['curr']['state'] = cgmn.state_dict()
                CHK['OPT'] = opt.state_dict()
                if vl_loss.item() < CHK['CV']['abs_v_loss']:
                    CHK['CV']['abs_v_loss'] = vl_loss.item()
                    CHK['MOD']['best']['state'] = cgmn.state_dict()
                    CHK['MOD']['best']['L'] = CHK['MOD']['curr']['L']
                torch.save(CHK, chk_path)

                pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt == PATIENCE:
                    print("Patience over: training stopped.")
                    break
        CHK['CV']['epoch'] = 0
        CHK['CV']['v_loss'] = float('inf')
        CHK['OPT'] = None
        torch.save(CHK, chk_path)

    # TESTING
    cgmn = CGMN(1, M, C, 37, device=DEVICE)
    for _ in range(len(cgmn.cgmm.layers), CHK['MOD']['best']['L']):
        cgmn.stack_layer()
    cgmn.load_state_dict(CHK['MOD']['best']['state'])

    for ts_batch in ts_ld:
        with torch.no_grad():
            ts_batch = ts_batch.to(DEVICE, non_blocking=True) if sys.argv[1] != 'cpu:0' else ts_batch
            out, neg_likelihood = cgmn(ts_batch.x, ts_batch.edge_index, ts_batch.batch)
            ts_loss = bce(out, ts_batch.y)
            ts_acc = accuracy(ts_batch.y, out.sigmoid().round())
    print(f"Fold {CHK['CV']['fold']}: Loss = {ts_loss.item()} ---- Accuracy = {ts_acc}")

    preproc_from_layer = 0
    CHK['CV']['loss'].append(ts_loss.item())
    CHK['CV']['acc'].append(ts_acc)
    CHK['CV']['fold'] += 1
    CHK['CV']['epoch'] = 0
    CHK['CV']['abs_v_loss'] = float('inf')
    CHK['CV']['v_loss'] = float('inf')
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
