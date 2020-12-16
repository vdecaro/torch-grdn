import sys
import os

import torch
import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold
from cv_utils import get_cv_dict, get_cgmn, cgmn_incr_train, eval_model

from cgmn.cgmn import CGMN

def transform(dataset):
    
    if dataset in ['NCI1', 'PROTEINS', 'DD']:
        def func(data):
            data.y = data.y.unsqueeze(1).type(torch.FloatTensor)
            return data
    
    return func

###################################
#          CV HPARAMS             #
###################################
RD_STATES = [59, 92, 95]

DEVICE = torch.device(sys.argv[1])
DATASET = sys.argv[2]
S_FOLD = int(sys.argv[3])
F_FOLD = int(sys.argv[4])

if DATASET == 'NCI1':
    OUT_FEATURES = 1
    N_SYMBOLS = 37
    EPOCHS = 5000
    E_PAT = 30
    L_PAT = 2
    MAX_DEPTH = 20
    BATCH_SIZE = 100
    loss = torch.nn.BCEWithLogitsLoss()
    HPARAMS = []

elif DATASET == 'PROTEINS':
    OUT_FEATURES = 1
    N_SYMBOLS = 3
    EPOCHS = 5000
    E_PAT = 100
    L_PAT = 2
    MAX_DEPTH = 20
    BATCH_SIZE = 100
    loss = torch.nn.BCEWithLogitsLoss()
    HPARAMS = []

elif DATASET == 'DD':
    OUT_FEATURES = 1
    N_SYMBOLS = 89
    EPOCHS = 5000
    E_PAT = 40
    L_PAT = 3
    MAX_DEPTH = 20
    BATCH_SIZE = 80
    loss = torch.nn.BCEWithLogitsLoss()
    HPARAMS = []

chk_path = f"CGMN_CV/{DATASET}_{S_FOLD}_{F_FOLD}.tar"
CHK, restart = get_cv_dict(chk_path, S_FOLD)
EXT = CHK['EXT']
CURR = CHK['CURR']
if restart:
    if CHK['INT'][EXT['fold']]['fold'] < 5:
        INT = CHK['INT'][EXT['fold']]
        print(f"Restarting from EXT {EXT['fold']} - INT {INT['fold']} - CONF {HPARAMS[INT['hparams_idx']]} - Epoch {CURR['epoch']} with curr best acc {CURR['v_acc']} and abs best acc {CURR['abs_v_acc']}")
    else:
        print(f"Restarting from EXT {EXT['fold']} - TEST - CONF {EXT['best'][EXT['fold'] - S_FOLD]} - Epoch {CURR['epoch']} with curr best acc {CURR['v_acc']} and abs best acc {CURR['abs_v_acc']}")


dataset = TUDataset(f'.', DATASET, transform=transform(DATASET))
dataset.data.x = dataset.data.x.argmax(1).detach()

# EXTERNAL 10-FOLD CROSS-VALIDATION
ext_kfold = StratifiedKFold(10, shuffle=True, random_state=42)
ext_split = list(ext_kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

for int_i, ext_i in ext_split[EXT['fold']:F_FOLD]:
    int_data, ext_data = dataset[int_i.tolist()], dataset[ext_i.tolist()]
    int_kfold = StratifiedKFold(5, shuffle=True, random_state=15)
    int_split = list(int_kfold.split(X=np.zeros(len(int_data)), y=np.array([g.y for g in int_data])))
    INT = CHK['INT'][EXT['fold']]

    # INTERNAL 5-FOLD CROSS-VALIDATION
    for tr_i, vl_i in int_split[INT['fold']:]:
        tr_data, vl_data = int_data[tr_i.tolist()], int_data[vl_i.tolist()]
        tr_ld = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, drop_last=(len(tr_data)%BATCH_SIZE == 1))
        vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=False)

        for hparams in HPARAMS[INT['hparams_idx']:]:
            hparams_ = (OUT_FEATURES, hparams[0], hparams[1], N_SYMBOLS, hparams[2], hparams[3], hparams[4])
            cgmn_incr_train(CHK, hparams_, loss, tr_ld, vl_ld, EPOCHS, E_PAT, L_PAT, MAX_DEPTH, DEVICE)
            INT['v_acc'][INT['fold']].append(CURR['abs_v_acc'])
            INT['tr_acc'][INT['fold']].append(CURR['abs_tr_acc'])

            CURR['MOD'] = {
                'best': {
                    'L': 0,
                    'state': None
                },
                'curr': {
                    'L': 0,
                    'state': None
                },
            }
            CURR['trained'] = False
            CURR['abs_v_acc'] = 0
            CURR['abs_tr_acc'] = 0
            INT['hparams_idx'] += 1
            torch.save(CHK, CHK['PATH'])
        INT['hparams_idx'] = 0
        INT['fold'] += 1
        torch.save(CHK, CHK['PATH'])
    torch.save(CHK, CHK['PATH'])

    # DEFINING THE BEST CONFIGURATION OF HPARAMS ACROSS THE INTERNAL 5-FOLD CV
    if len(EXT['best']) == (EXT['fold'] - S_FOLD):
        best_vl = np.mean(INT['v_acc'], axis=0)
        best_tr = np.mean(INT['tr_acc'], axis=0)

        best_idx = []
        best_vl_value = 0
        for idx, value in enumerate(best_vl):
            if value > best_vl_value:
                best_vl_value = value
                best_idx = [idx]
            elif value == best_vl_value:
                best_idx.append(idx)
        
        if len(best_idx) > 1:
            best_tr_value = 0
            final_idx = None
            for idx in best_idx:
                if best_tr[idx] > best_tr_value:
                    final_idx = idx
                    best_tr_value = best_tr[idx]
        else:
            final_idx = best_idx[0]
        
        EXT['best'].append(HPARAMS[final_idx])
        torch.save(CHK, CHK['PATH'])
    best_params = EXT['best'][EXT['fold'] - S_FOLD]

    # EXTERNAL TEST: three attempts on three different hold-out splits
    for rd_state in RD_STATES[EXT['state']:]:
        tr_i, vl_i = train_test_split(np.arange(len(int_data)), 
                                      test_size=0.1,  
                                      stratify=np.array([g.y for g in int_data]), 
                                      shuffle=True, 
                                      random_state=rd_state)
        tr_data, vl_data = int_data[tr_i.tolist()], int_data[vl_i.tolist()]

        tr_ld = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
        vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)
        ts_ld = DataLoader(ext_data, batch_size=len(ext_data), shuffle=False, pin_memory=False)
        for attempt in range(EXT['attempt'], 3):
            hparams_ = (OUT_FEATURES, best_params[0], best_params[1], N_SYMBOLS, best_params[2], best_params[3], best_params[4])
            cgmn_incr_train(CHK, hparams_, loss, tr_ld, vl_ld, EPOCHS, E_PAT, L_PAT, MAX_DEPTH, DEVICE)
            
            cgmn = get_cgmn(CURR, hparams_, 'best', DEVICE)
            ts_loss, ts_acc = eval_model(cgmn, loss, ts_ld, DEVICE)
            EXT['v_acc'][EXT['fold']].append(CURR['abs_v_acc'])
            EXT['t_loss'][EXT['fold']].append(ts_loss)
            EXT['t_acc'][EXT['fold']].append(ts_acc)
            print(f"Test {EXT['fold']} - State {rd_state} - Attempt {attempt}: Vl acc = {CURR['abs_v_acc']} ---- Ts acc = {ts_acc}")

            CURR['MOD'] = {
                'best': {
                    'L': 0,
                    'state': None
                },
                'curr': {
                    'L': 0,
                    'state': None
                },
            }
            CURR['trained'] = False
            CURR['abs_v_acc'] = 0
            CURR['abs_tr_acc'] = 0
            EXT['attempt'] += 1
            torch.save(CHK, CHK['PATH'])

        EXT['attempt'] = 0
        EXT['state'] += 1
        torch.save(CHK, CHK['PATH'])
    EXT['state'] = 0
    EXT['fold'] += 1
    torch.save(CHK, chk_path)
