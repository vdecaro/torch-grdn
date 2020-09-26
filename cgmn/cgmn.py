import torch
import torch.nn as nn

from contrastive import contrastive_matrix

from cgmn.cgmm import CGMM
from torch_geometric.nn import Set2Set
from torch_scatter.scatter import scatter

import time

class CGMN(nn.Module):

    def __init__(self, out_features, n_gen, C, L=None, M=None, device='cpu:0'):
        super(CGMN, self).__init__()
        self.device = torch.device(device)
        self.cgmm = CGMM(n_gen, C, L, M, device=device)
        self.b_norm = nn.ModuleList([nn.BatchNorm1d(self.cgmm.n_gen, affine=False)])
        self.contrastive = contrastive_matrix(self.cgmm.n_gen, self.device)
        
        self.output = nn.ModuleList([nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), out_features)])

        self.to(device=self.device)
    
    def forward(self, x, edge_index, batch, pos=None):
        log_likelihood = self.cgmm(x, edge_index, pos)
        log_likelihood = scatter(log_likelihood, batch, dim=0)
        b_norm_lhood = torch.stack([b(log_likelihood[:, i]) for i, b in enumerate(self.b_norm)], dim=1)

        c_neurons = (b_norm_lhood @ self.contrastive).tanh().detach_()
        c_neurons = c_neurons.flatten(start_dim=-2)
        output = self.output[-1](c_neurons)
        
        return output

    def stack_layer(self):
        self.cgmm.stack_layer()
        self.b_norm.append(nn.BatchNorm1d(self.cgmm.n_gen, affine=False, momentum=0.4))
        self.b_norm[-1].to(device=self.device)
        
        self.output.append(nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), self.output[-1].out_features))
        self.output[-1].to(device=self.device)
    
    def train(self, mode=True):
        super(CGMN, self).train(mode=mode)  # will turn on batchnorm (buffers not params).

        for b in self.b_norm[:-1]:
            b.eval()

        return self


