import os

import torch
from torch_geometric.utils.metric import accuracy
from cgmn.cgmn import CGMN

def get_cv_dict(chk_path, s_fold):
    if os.path.exists(chk_path):
        restart = True
        CHK = torch.load(chk_path)
    else:
        restart = False
        CHK = {
            'EXT': {
                'fold': s_fold,
                'state': 0,
                'attempt': 0,
                'best': [],
                't_loss': [[] for _ in range(10)],
                't_acc': [[] for _ in range(10)],
                'v_acc': [[] for _ in range(10)],
            },

            'INT': [{
                'fold': 0,
                'v_acc': [[] for _ in range(5)],
                'tr_acc': [[] for _ in range(5)],
                'hparams_idx': 0,
            } for _ in range(10)],

            'CURR': {
                'epoch': 0,
                'abs_v_acc': 0,
                'abs_tr_acc': 0,
                'v_acc': 0,
                'tr_acc': 0,
                'e_pat': 0,
                'l_pat': 0,
                'MOD': {
                    'best': {
                        'L': 0,
                        'state': None
                    },
                    'curr': {
                        'L': 0,
                        'state': None
                    },
                },
                'G_OPT': None,
                'N_OPT': None,
                'trained': False,
            },
            'PATH': chk_path
        }
    return CHK, restart

def cgmn_incr_train(chk, hparams, loss, tr_ld, vl_ld, epochs, e_pat, l_pat, max_depth, device, verbose=True):
    EXT = chk['EXT']
    INT = chk['INT'][EXT['fold']]
    CURR = chk['CURR']
    MOD = CURR['MOD']
    
    while not CURR['trained']:
        cgmn = get_cgmn(CURR, hparams[:-2], 'curr', device)
        if MOD['curr']['L'] > 0 and CURR['G_OPT'] is None and CURR['N_OPT'] is None:
            cgmn.stack_layer()
            
        if CURR['G_OPT'] is None and CURR['N_OPT'] is None:
            MOD['curr']['L'] += 1
            
        MOD['curr']['state'] = cgmn.state_dict()
        g_opt, n_opt = get_opt(CURR, cgmn, hparams[-2], hparams[-1])
        for i in range(CURR['epoch'], epochs):
            tr_loss, tr_acc = train_model(cgmn, g_opt, n_opt, loss, tr_ld, cgmn.device)
            vl_loss, vl_acc = eval_model(cgmn, loss, vl_ld, cgmn.device)
            if verbose:
                params_str = f"{hparams[1]}, {hparams[2]}, {hparams[4]}, {hparams[6]}"
                int_phase_str = f"INT {INT['fold']}" if INT['fold'] < 5 else "TEST"
                print(f"EXT {EXT['fold']} - {int_phase_str} - CONF ({params_str}) - Layer {MOD['curr']['L']} - Epoch {i}: Tr Acc = {round(tr_acc, 3)}  ---- Vl Acc = {round(vl_acc, 3)}")
            CURR['epoch'] += 1
            if vl_acc >= CURR['v_acc'] and tr_acc > CURR['tr_acc']:
                CURR['v_acc'] = vl_acc
                CURR['tr_acc'] = tr_acc
                MOD['curr']['state'] = cgmn.state_dict()
                CURR['G_OPT'] = g_opt.state_dict()
                CURR['N_OPT'] = n_opt.state_dict()
                if  vl_acc >= CURR['abs_v_acc'] and tr_acc > CURR['abs_tr_acc']:
                    CURR['abs_v_acc'] = vl_acc
                    CURR['abs_tr_acc'] = tr_acc
                    MOD['best']['state'] = cgmn.state_dict()
                    MOD['best']['L'] = MOD['curr']['L']
                    CURR['l_pat'] = 0
                CURR['e_pat'] = 0
            else:
                CURR['e_pat'] += 1
                if CURR['e_pat'] >= e_pat:
                    break 
            torch.save(chk, chk['PATH'])
        CURR['v_acc'] = 0
        CURR['tr_acc'] = 0
        CURR['G_OPT'] = None
        CURR['N_OPT'] = None
        CURR['e_pat'] = 0
        CURR['epoch'] = 0
        CURR['trained'] = MOD['curr']['L'] == max_depth or CURR['l_pat'] == l_pat
        CURR['l_pat'] += 1
        torch.save(chk, chk['PATH'])
    CURR['l_pat'] = 0

######################################################
#         INIT AND RECOVERING OF MODEL AND OPT       #
######################################################
def get_cgmn(curr_chk, hparams, which, device):
    CURR = curr_chk
    MOD = CURR['MOD']
    cgmn = CGMN(hparams[0], hparams[1], hparams[2], None, hparams[3], hparams[4], device)
    for _ in range(len(cgmn.cgmm.layers), MOD[which]['L']):
        cgmn.stack_layer()

    if MOD[which]['state'] is not None:
        cgmn.load_state_dict(MOD[which]['state'])

    return cgmn

def get_opt(curr_chk, cgmn, lr, l2=None):
    CURR = curr_chk
    g_p, n_p = cgmn.get_params()
    g_opt = torch.optim.Adam(g_p, lr=lr)
    n_opt = torch.optim.Adam(n_p, lr=lr) if l2 is None else torch.optim.AdamW(n_p, lr=lr, weight_decay=l2)
    if CURR['G_OPT'] is not None and CURR['N_OPT'] is not None:
        g_opt.load_state_dict(CURR['G_OPT'])
        n_opt.load_state_dict(CURR['N_OPT'])
        
    return g_opt, n_opt


######################################################
#             TRAINING AND EVALUATION                #
######################################################
def train_model(mod, g_opt, n_opt, loss, loader, device):
    mod.train()
    l_avg = 0
    a_avg = 0
    for i, b in enumerate(loader):
        b = b.to(device, non_blocking=True)
        g_opt.zero_grad()
        n_opt.zero_grad()
        out = mod(b.x, b.edge_index, b.batch)
        loss_v = loss(out, b.y)
        loss_v.backward()
        g_opt.step()
        n_opt.step()
        acc_v = accuracy(b.y, out.sigmoid().round())
        l_avg += (loss_v - l_avg) / (i+1)
        a_avg += (acc_v - a_avg) / (i+1)
    return l_avg, a_avg


def eval_model(mod, loss, loader, device):
    mod.eval()
    for b in loader:
        with torch.no_grad():
            b = b.to(device, non_blocking=True)
            out = mod(b.x, b.edge_index, b.batch)
            loss_v = loss(out, b.y).item()
            acc_v = accuracy(b.y, out.sigmoid().round())
    
    return loss_v, acc_v
