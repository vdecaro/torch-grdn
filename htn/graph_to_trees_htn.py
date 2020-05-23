import torch
import torch.nn as nn
from data.graph.preproc import bfs
from htn.contrastive import contrastive_matrix

from genmodel.bhtmm import BottomUpHTMM
from genmodel.thtmm import TopDownHTMM
from torch_geometric.nn import Set2Set

from math import factorial as fact
import random

class GraphHTN(nn.Module):

    def __init__(self, outputs, n_bu, n_td, C, L, M, set2set_steps=8):
        super(GraphHTN, self).__init__()
        self.bu = BottomUpHTMM(n_bu, C, L, M)
        self.td = TopDownHTMM(n_td, C, L, M)
        
        if self.bu and self.td:
            self.mode = 'both'
            self.td_batch_norm = nn.BatchNorm1d(n_td, affine=False)
            self.bu_batch_norm = nn.BatchNorm1d(n_bu, affine=False)
        elif self.bu:
            self.mode = 'bu'
        elif self.td:
            self.mode = 'td'

        self.contrastive = contrastive_matrix(n_bu + n_td)
        self.set2one = Set2Set(self.contrastive.size(1), set2set_steps, num_layers=1)
        self.output = nn.Linear(self.contrastive.size(1), outputs)
    
    def forward(self, graph_batch):
        neg_log_likelihood = []
        to_contrastive = []
        batch_idx = []
        for idx, g in enumerate(graph_batch):
            g_dim = g.x.size(0)
            g_trees = [bfs(g, root) for root in range(g_dim)]
            batch_idx += [idx for _ in range(g_dim)]

            if self.mode == 'both':
                g_neg_td_likelihood = self.td(g_trees)
                g_neg_bu_likelihood = self.bu(g_trees)
                norm_td = self.td_batch_norm(g_neg_td_likelihood)
                norm_bu = self.bu_batch_norm(g_neg_bu_likelihood)
                to_contrastive.append(torch.cat([norm_td, norm_bu], dim=1))
                neg_log_likelihood.append(torch.cat([g_neg_td_likelihood, g_neg_bu_likelihood], dim=1).mean(0))

            elif self.mode == 'bu':
                g_neg_bu_likelihood = self.bu(g_trees).detach()
                to_contrastive.append(g_neg_bu_likelihood)
                neg_log_likelihood.append(g_neg_bu_likelihood.mean(0))

            elif self.mode == 'td':
                g_neg_td_likelihood = self.td(g_trees).detach()
                to_contrastive.append(g_neg_td_likelihood)
                neg_log_likelihood.append(g_neg_td_likelihood.mean(0))
        
        c_neurons = (torch.cat(to_contrastive, dim=0) @ self.contrastive).tanh()
        g_pooling = self.set2one(c_neurons, batch_idx)
        output = self.output(g_pooling)
        
        return output, torch.stack(neg_log_likelihood).mean(0).detach()

    def get_gen_parameters(self):
        params = []

        if self.bu is not None:
            params += self.bu.parameters

        if self.td is not None:
            params += self.td.parameters

        return params
