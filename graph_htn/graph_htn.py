import torch
import torch.nn as nn

from contrastive import contrastive_matrix
from graph_htn.uni_bhtmm import UniformBottomUpHTMM
from graph_htn.thtmm import TopDownHTMM
from torch_geometric.nn import Set2Set, global_add_pool
from torch_scatter.scatter import scatter

from math import factorial as fact
import random

class GraphHTN(nn.Module):

    def __init__(self, out_features, n_bu, n_td, C, M, set2set_steps=2):
        super(GraphHTN, self).__init__()
        self.bu = UniformBottomUpHTMM(n_bu, C, M) if n_bu is not None and n_bu > 0 else None
        self.td = TopDownHTMM(n_td, C, M) if n_td is not None and n_td > 0 else None

        self.td_batch_norm = nn.BatchNorm1d(n_td, affine=False) if n_td is not None and n_td > 0 else None
        self.bu_batch_norm = nn.BatchNorm1d(n_bu, affine=False) if n_bu is not None and n_bu > 0 else None

        self.contrastive = contrastive_matrix(n_bu + n_td)
        self.set2set = Set2Set(self.contrastive.size(1), set2set_steps)
        self.output = nn.Linear(2*self.contrastive.size(1), out_features)
    
    def forward(self, x, trees, batch):

        if self.bu is not None and self.td is not None:
            g_neg_td_likelihood = self.td(x, trees)
            g_neg_bu_likelihood = self.bu(x, trees)
            norm_td = self.td_batch_norm(g_neg_td_likelihood)
            norm_bu = self.bu_batch_norm(g_neg_bu_likelihood)
            to_contrastive = torch.cat([norm_td, norm_bu], dim=1)
            neg_log_likelihood = torch.cat([g_neg_td_likelihood, g_neg_bu_likelihood], dim=1)

        elif self.bu is not None:
            g_neg_bu_likelihood = self.bu(x, trees)
            to_contrastive = self.bu_batch_norm(g_neg_bu_likelihood)
            neg_log_likelihood = g_neg_bu_likelihood

        elif self.td is not None:
            g_neg_td_likelihood = self.td(x, trees)
            to_contrastive = self.td_batch_norm(g_neg_td_likelihood)
            neg_log_likelihood = g_neg_td_likelihood
        
        c_neurons = (to_contrastive @ self.contrastive).tanh().detach_()
        g_pooling = self.set2set(c_neurons, batch)
        output = self.output(g_pooling)
        
        return output, neg_log_likelihood.mean(0).sum()

    def get_parameters(self):
        params = [p for p in self.parameters()]
        params = params + [p for p in self.bu.parameters()] if self.bu is not None else params
        params += [p for p in self.td.parameters()] if self.td is not None else params
        return params