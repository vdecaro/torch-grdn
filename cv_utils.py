import os

import torch
from torch_geometric.utils.metric import accuracy
from cgmn.cgmn import CGMN

def get_cv_dict(chk_path):
    if os.path.exists(chk_path):
        restart = True
        CHK = torch.load(chk_path)
    else:
        restart = False
        CHK = {
            'EXT': {
                'fold': 0,
                'state': 0,
                'attempt': 0,
                'best': [],
                'loss': [[] for _ in range(10)],
                'acc': [[] for _ in range(10)],
            },

            'INT': [{
                'fold': 0,
                'loss': [[] for _ in range(5)],
                'hparams_idx': 0,
            } for _ in range(10)],

            'CURR': {
                'epoch': 0,
                'abs_v_loss': float('inf'),
                'v_loss': float('inf'),
                'e_pat': 0,
                'l_pat': 0,
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
                'trained': False,
            },
            'PATH': chk_path
        }
    return CHK, restart

def cgmn_incr_train(chk, hparams, loss, tr_ld, vl_ld, epochs, e_pat, l_pat, max_depth, device, verbose=True):
    EXT = chk['EXT']
    INT = chk['INT']
    CURR = chk['CURR']
    MOD = chk['CURR']['MOD']
    
    while not CURR['trained']:
        cgmn = get_cgmn(CURR, hparams[:-1], 'curr', device)
        if CURR['OPT'] is None:
            cgmn.stack_layer()
            MOD['curr']['state'] = cgmn.state_dict()
            MOD['curr']['L'] += 1
        opt = get_opt(CURR, cgmn, hparams[-1])
        for i in range(CURR['epoch'], epochs):
            _ = train_model(cgmn, opt, loss, tr_ld, cgmn.device)
            vl_loss, vl_acc = eval_model(cgmn, loss, vl_ld, cgmn.device)
            if verbose:
                print(f"EXT {EXT['fold']} - INT {INT['fold']} - CONF {hparams} - Layer {MOD['curr']['L']} - Epoch {i}: Loss = {vl_loss} ---- Accuracy = {vl_acc}")
            CURR['epoch'] += 1
            if vl_loss < CURR['v_loss']:
                CURR['v_loss'] = vl_loss
                MOD['curr']['state'] = cgmn.state_dict()
                CURR['OPT'] = opt.state_dict()
                if  vl_loss < CURR['abs_v_loss']:
                    CURR['abs_v_loss'] = vl_loss
                    MOD['best']['state'] = cgmn.state_dict()
                    MOD['best']['L'] = MOD['curr']['L']
                    CURR['l_pat'] = 0
                if vl_loss < CURR['e_pat']:
                    CURR['e_pat'] = 0
                else:
                    CURR['e_pat'][0] += 1
                    if CURR['e_pat'] >= e_pat:
                        break 
            torch.save(chk, chk['PATH'])
        CURR['v_loss'] = float('inf')
        CURR['OPT'] = None
        CURR['e_pat'] = 0
        CURR['epoch'] = 0
        CURR['l_pat'] += 1
        curr['trained'] = MOD['curr']['L'] < max_depth and CURR['l_pat'] <= l_pat
        torch.save(chk, chk['PATH'])


######################################################
#         INIT AND RECOVERING OF MODEL AND OPT       #
######################################################
def get_cgmn(curr_chk, hparams, which, device):
    CURR = curr_chk
    MOD = CURR['MOD']
    cgmn = CGMN(hparams[0], hparams[1], hparams[2], None, hparams[3], hparams[4], hparams[5])
    for _ in range(len(cgmn.cgmm.layers), MOD[which]['L']):
        cgmn.stack_layer()

    if MOD[which]['state'] is not None:
        cgmn.load_state_dict(MOD[which]['state'])

    return cgmn

def get_opt(curr_chk, cgmn, lr):
    CURR = curr_chk
    opt = torch.optim.Adam(cgmn.parameters(), lr=lr)
    if CURR['OPT'] is not None:
        opt.load_state_dict(CURR['OPT'])                 
    return opt


######################################################
#             TRAINING AND EVALUATION                #
######################################################
def train_model(mod, opt, loss, loader, device):
    mod.train()
    m_avg = 0
    for i, b in enumerate(loader):
        b = b.to(device, non_blocking=True)
        opt.zero_grad()
        out = mod(b.x, b.edge_index, b.batch)
        loss_v = loss(out, b.y)
        loss_v.backward()
        opt.step()
        m_avg += (loss_v - m_avg) / (i+1)
    return m_avg


def eval_model(mod, loss, loader, device):
    mod.eval()
    for b in loader:
        with torch.no_grad():
            b = b.to(device, non_blocking=True)
            out = mod(b.x, b.edge_index, b.batch)
            loss_v = loss(out, b.y).item()
            acc_v = accuracy(b.y, out.sigmoid().round())
    return loss_v, acc_v
