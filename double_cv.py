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
    E_PAT = 50
    L_PAT = 2
    MAX_DEPTH = 20
    BATCH_SIZE = 100
    loss = torch.nn.BCEWithLogitsLoss()
    HPARAMS = [
        (15, 3, 8, 1e-4), (15, 3, 16, 1e-4), (15, 3, 32, 1e-4),
        (25, 4, 16, 1e-4), (25, 4, 32, 1e-4), (25, 4, 48, 1e-4),
        (30, 4, 16, 1e-4), (30, 4, 32, 1e-4), (30, 4, 48, 1e-4),
        (40, 5, 32, 1e-4), (40, 5, 48, 1e-4), (40, 5, 64, 1e-4),
    ]

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

chk_path = f"CGMN_CV/{DATASET}.tar"
CHK, restart = get_cv_dict(chk_path)
EXT = CHK['EXT']
CURR = CHK['CURR']
if restart:
    if CHK['INT'][EXT['fold']]['fold'] < 5:
        INT = CHK['INT'][EXT['fold']]
        print(f"Restarting from EXT {EXT['fold']} - INT {INT['fold']} - CONF {HPARAMS[INT['hparams_idx']]} - Epoch {CURR['epoch']} with curr best loss {CURR['v_loss']} and abs best loss {CURR['abs_v_loss']}")
    else:
        print(f"Restarting from EXT {EXT['fold']} - TEST - CONF {EXT['best'][EXT['fold']]} - Epoch {CURR['epoch']} with curr best loss {CURR['v_loss']} and abs best loss {CURR['abs_v_loss']}")


dataset = TUDataset(f'.', DATASET, transform=transform(DATASET))
dataset.data.x = dataset.data.x.argmax(1).detach()

# EXTERNAL 10-FOLD CROSS-VALIDATION
ext_kfold = StratifiedKFold(10, shuffle=True, random_state=42)
ext_split = list(ext_kfold.split(X=np.zeros(len(dataset)), y=np.array([g.y for g in dataset])))

for int_i, ext_i in ext_split[EXT['fold']:]:
    int_data, ext_data = dataset[int_i.tolist()], dataset[ext_i.tolist()]
    int_kfold = StratifiedKFold(5, shuffle=True, random_state=15)
    int_split = list(int_kfold.split(X=np.zeros(len(int_data)), y=np.array([g.y for g in int_data])))
    INT = CHK['INT'][EXT['fold']]

    # INTERNAL 5-FOLD CROSS-VALIDATION
    for tr_i, vl_i in int_split[INT['fold']:]:
        tr_data, vl_data = int_data[tr_i.tolist()], int_data[vl_i.tolist()]
        tr_ld = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=(len(tr_data)%BATCH_SIZE == 1))
        vl_ld = DataLoader(vl_data, batch_size=len(vl_data), shuffle=False, pin_memory=True)

        for hparams in HPARAMS[INT['params_idx']:]:
            hparams_ = (OUT_FEATURES, hparams[0], hparams[1], N_SYMBOLS, hparams[2], hparams[3])
            cgmn_incr_train(CHK, hparams_, loss, tr_ld, vl_ld, EPOCHS, E_PAT, L_PAT, MAX_DEPTH, DEVICE)
            INT['loss'][INT['fold']].append(CURR['abs_v_loss'])

            CURR['MOD'] = {
                'best': {
                    'L': 1,
                    'state': None
                },
                'curr': {
                    'L': 1,
                    'state': None
                },
            }
            CURR['trained'] = False
            CURR['abs_v_loss'] = float('inf')
            INT['hparams_idx'] += 1
            torch.save(CHK, CHK['PATH'])
        INT['hparams_idx'] = 0
        INT['fold'] += 1
        torch.save(CHK, CHK['PATH'])
    torch.save(CHK, CHK['PATH'])

    # DEFINING THE BEST CONFIGURATION OF HPARAMS ACROSS THE INTERNAL 5-FOLD CV
    if len(EXT['best']) == EXT['fold']:
        best_hparams_idx = np.argmin(np.mean(INT['loss'], axis=0))
        EXT['best'].append(HPARAMS[best_hparams_idx])
        torch.save(CHK, CHK['PATH'])
    best_params = EXT['best'][EXT['fold']]

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
            hparams_ = (OUT_FEATURES, best_params[0], best_params[1], N_SYMBOLS, best_params[2], best_params[3])
            cgmn_incr_train(CHK, hparams_, loss, tr_ld, vl_ld, EPOCHS, E_PAT, L_PAT, MAX_DEPTH, DEVICE)
            
            cgmn = get_cgmn(CURR, best_params, 'best', DEVICE)
            ts_loss, ts_acc = eval_model(cgmn, loss, ts_ld, DEVICE)
            EXT['loss'][EXT['fold']].append(ts_loss)
            EXT['acc'][EXT['fold']].append(ts_acc)
            print(f"Test {EXT['fold']} - State {rd_state} - Attempt {attempt}: Loss = {ts_loss} ---- Accuracy = {ts_acc}")

            CURR['MOD'] = {
                'best': {
                    'L': 1,
                    'state': None
                },
                'curr': {
                    'L': 1,
                    'state': None
                },
            }
            CURR['trained'] = False
            CURR['abs_v_loss'] = float('inf')
            EXT['attempt'] += 1
            torch.save(CHK, CHK['PATH'])

        EXT['attempt'] = 0
        EXT['state'] += 1
        torch.save(CHK, CHK['PATH'])
    EXT['state'] = 0
    EXT['fold'] += 1
    torch.save(CHK, chk_path)