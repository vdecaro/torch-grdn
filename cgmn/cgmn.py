import torch
import torch.nn as nn

from contrastive import contrastive_matrix

from cgmn.cgmm import CGMM
from torch_geometric.nn import GlobalAttention
from torch_scatter.scatter import scatter

import time

class CGMN(nn.Module):

    def __init__(self, out_features, n_gen, C, L=None, M=None, gate_units=None, device='cpu:0'):
        super(CGMN, self).__init__()
        self.device = torch.device(device)
        self.cgmm = CGMM(n_gen, C, L, M, device=device)
        self.b_norm = nn.ModuleList([nn.BatchNorm1d(self.cgmm.n_gen, affine=False)])
        self.contrastive = contrastive_matrix(self.cgmm.n_gen, self.device)
        self.node_features = self.contrastive.size(1)
        self.gate_units = gate_units

        self.pooling = nn.ModuleList([GlobalAttention(_GateNN(self.node_features, gate_units))])
        self.output = nn.ModuleList([nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), out_features)])

        self.to(device=self.device)
    
    def forward(self, x, edge_index, batch, pos=None):
        log_likelihood = self.cgmm(x, edge_index, pos)
        b_norm_lhood = torch.stack([b(log_likelihood[:, i]) for i, b in enumerate(self.b_norm)], dim=1)
        c_neurons = (b_norm_lhood @ self.contrastive).tanh().detach_()
        
        r_i = []
        for i, att in enumerate(self.pooling):
            if i < len(self.pooling)-1:
                with torch.no_grad():
                    r_i.append(att(c_neurons[:, i], batch))
            else:
                r_i.append(att(c_neurons[:, i], batch))
        r_i = torch.cat(r_i, -1)
        output = self.output[-1](r_i)
        
        return output

    def stack_layer(self):
        self.cgmm.stack_layer()
        self.b_norm.append(nn.BatchNorm1d(self.cgmm.n_gen, affine=False))
        self.b_norm[-1].to(device=self.device)
        
        self.pooling.append(GlobalAttention(_GateNN(self.node_features, self.gate_units)))
        self.pooling[-1].to(device=self.device)

        self.output.append(nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), self.output[-1].out_features))
        self.output[-1].to(device=self.device)
    
    def train(self, mode=True):
        super(CGMN, self).train(mode=mode)  # will turn on batchnorm (buffers not params).

        for b in self.b_norm[:-1]:
            b.eval()

        return self


class _GateNN(nn.Module):
    def __init__(self, input_features, gate_units):
        super(_GateNN, self).__init__()
        self.h = nn.Linear(input_features, gate_units)
        self.out = nn.Linear(gate_units, 1)

    def forward(self, x):
        return self.out(self.h(x).tanh())