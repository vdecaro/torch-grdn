import torch
import torch.nn as nn

from contrastive import contrastive_matrix
from graph_htn.uni_bhtmm import UniformBottomUpHTMM
from graph_htn.thtmm import TopDownHTMM
from torch_geometric.nn import Set2Set, BatchNorm
from torch_scatter.scatter import scatter

from math import factorial as fact
import random

import time

class GraphHTMN(nn.Module):

    def __init__(self, out_features, n_bu, n_td, C, M, set2set_steps=10, device='cpu:0'):
        super(GraphHTMN, self).__init__()
        self.device = torch.device(device)
        self.bu = UniformBottomUpHTMM(n_bu, C, M, device) if n_bu is not None and n_bu > 0 else None
        self.td = TopDownHTMM(n_td, C, M, device) if n_td is not None and n_td > 0 else None

        self.td_norm = BatchNorm(n_td, affine=False) if n_td is not None and n_td > 0 else None
        self.bu_norm = BatchNorm(n_bu, affine=False) if n_bu is not None and n_bu > 0 else None

        self.contrastive = contrastive_matrix(n_bu + n_td, self.device)
        self.set2set = Set2Set(self.contrastive.size(1), 8, 1)
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(2*self.contrastive.size(1), out_features)
        self.to(device=self.device)
    
    def forward(self, x, trees, batch):
        to_contrastive = []
        if self.bu is not None:
            g_bu_likelihood = self.bu(x, trees, batch)
            to_contrastive.append(self.bu_norm(g_bu_likelihood))

        if self.td is not None:
            g_td_likelihood = self.td(x, trees)
            to_contrastive.append(self.td_norm(g_td_likelihood))

        if len(to_contrastive) == 2:
            to_contrastive = torch.cat(to_contrastive, dim=1)
        else:
            to_contrastive = to_contrastive[0]
        
        c_neurons = (to_contrastive @ self.contrastive).tanh().detach_()
        c_neurons = self.dropout(c_neurons)
        g_pooling = self.set2set(c_neurons, batch)
        output = self.output(g_pooling)
        
        return output
