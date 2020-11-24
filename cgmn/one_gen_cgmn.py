import torch
import torch.nn as nn

from contrastive import contrastive_matrix

from cgmn.cgmm import CGMM
from torch_geometric.nn import GlobalAttention
from torch_scatter.scatter import scatter

import time

class OneGenCGMN(nn.Module):

    def __init__(self, out_features, C, L=None, M=None, device='cpu:0'):
        super(OneGenCGMN, self).__init__()
        self.device = torch.device(device)
        self.cgmm = CGMM(1, C, L, M, device=device)
        self.contrastive = contrastive_matrix(C, self.device)
        self.features = self.contrastive.size(1)

        self.to(device=self.device)
    
    def forward(self, x, edge_index, batch, pos=None):
        log_states = self.cgmm(x, edge_index, pos, True)
        log_states = scatter(log_states, batch, dim=0).squeeze(dim=-1)
        c_neurons = (log_states @ self.contrastive).tanh().flatten(start_dim=-2).detach()
        output = self.output[-1](c_neurons)
        return output
    
    def stack_layer(self):
        self.cgmm.stack_layer()
        
        self.output.append(nn.Linear(self.features*len(self.cgmm.layers), self.output[-1].out_features))
        self.output[-1].to(device=self.device)
