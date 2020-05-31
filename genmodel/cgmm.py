import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_scatter.scatter import scatter


class CGMM(nn.Module):

    def __init__(self, C_0, M, A=None):
        self.M = M
        self.A = A

        self.layers = [CGMMLayer_0(C_0, self.M)]

    def forward(self, g):
        h_states = []

        for i, l in enumerate(self.layers):
            l_h_states = l(g, torch.cat(h_states, dim=1)) if i > 0 else l(g)
            h_states.append(l_h_states)
        
        out = l_h_states

        return out
        
    def stack_layer(self, C):
        self.layers[-1].freeze()
        self.layers[-1].h_states_only = True

        self.layers.append(CGMMLayer(C, [l.C for l in self.layers], len(self.layers), A))




class CGMMLayer_0(nn.Module):

    def __init__(self, C, M):
        self.C = C
        self.M = M

        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C))))
        self.h_states_only = False
    
    def forward(self, g):
        sm_B, sm_Pi = self._softmax_reparameterization()
        if self.h_states_only:
            return sm_Pi.unsqueeze(0) * sm_B[:, g.x].T
        else:
            return sm_Pi.unsqueeze(0).repeat(g.x.size(0))

    def _softmax_reparameterization(self):
        if self.h_states_only:
            return None, F.softmax(self.Pi, dim=0)
        else:
            return F.softmax(self.B, dim=1), F.softmax(self.Pi, dim=0)





class CGMMLayer(MessagePassing):

    def __init__(self, C, C_prev, L, M, A=None):
        super(CGMMLayer, self).__init__(aggr="sum" if A is not None else "mean", flow="target_to_source")
        self.C = C
        self.C_prev = C_prev
        self.L = L
        self.A = A
        self.M = M

        if A is not None:
            self.S = nn.Parameter(nn.init.normal(torch.empty(A)))
            self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, sum(C_prev), A))))
        else:
            self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, sum(C_prev)))))

        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M))))
        self.SP = nn.Parameter(nn.init.normal_(torch.empty((L))))

        self.h_states_only = True
        
    def forward(self, prev_h, g):
        g_dim = g.x.size(0)
        
        sm_Q_neigh, sm_B, sm_SP, sm_S = self._softmax_reparameterization()

        i = 0
        sm_Q_SP_neigh = []
        for L, C in enumerate(self.C_prev):
            sm_Q_SP_neigh.append(sm_SP[L]*sm_Q_neigh[:, i:C])
            i = C
        sm_Q_SP_neigh = torch.cat(sm_Q_SP_neigh, dim=1)

        if self.A is not None:
            t_neigh = sm_Q_SP_neigh * sm_S.view(1, 1, -1)
        else:
            t_neigh = sm_Q_SP_neigh

        edge_labels = g['edge_labels'] if self.A is not None else None
        idx = torch.range(0, g_dim) if self.A is not None else None

        Qu_i = self.propagate(g.edge_index, 
                              size=(g_dim, g_dim), 
                              prev_h=prev_h, 
                              edge_labels=edge_labels, 
                              idx=idx,
                              t_neigh=t_neigh)

        if self.h_states_only:
            to_return = Qu_i
        else:
            to_return = Qu_i * sm_B[:, g.x].T
        
        return to_return

    def message(self, prev_h_j, edge_labels, idx_i, idx_j, t_neigh):
        if self.A is not None:
            tmp_trans_i = t_neigh[:, :, edge_labels[idx_i, idx_j]] * prev_h_j.view(1, -1)
        else:
            tmp_trans_i = t_neigh[:, :] * prev_h_j.view(1, -1)

        i = 0
        trans_i = []
        for C in self.C_prev:
            trans_i.append(tmp_trans_i[:, i:C].sum(dim=1))
        trans_i = sum(trans_i)

        if self.A is not None:
            n_same_e_labels = (edge_labels[idx_i, :].to_dense() == edge_labels[idx_i, idx_j]).sum(dim=0)
            trans_i /= n_same_e_labels

        return trans_i

    def _softmax_reparameterization(self):
        if self.A is not None:
            sm_Q_neigh = []
            for i in range(self.A):
                sm_Q_neigh_i = []
                idx = 0
                for C in self.C_prev:
                    sm_Q_neigh_i.append(F.softmax(self.Q_neigh[:, idx:C, i], dim=0))
                    idx = C
                sm_Q_neigh.append(torch.cat(sm_Q_neigh_i, dim=1))
            sm_Q_neigh = torch.stack(sm_Q_neigh, dim=-1)

            sm_S = F.softmax(self.S, dim=0)
        else:
            sm_Q_neigh = []
            idx = 0
            for C in self.C_prev:
                sm_Q_neigh.append(F.softmax(self.Q_neigh[:, idx:C, i], dim=0))
                idx = C
            sm_Q_neigh = torch.cat(sm_Q_neigh, dim=1)

            sm_S = None
        
        if self.h_states_only:
            sm_B = None
        else:
            sm_B = F.softmax(self.B, dim=1)
        
        sm_SP = F.softmax(self.SP, dim=0)

        return sm_Q_neigh, sm_B, sm_SP, sm_S

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
