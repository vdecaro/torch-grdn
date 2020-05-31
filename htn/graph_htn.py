import torch
import torch.nn as nn

from htn.contrastive import contrastive_matrix

from genmodel.cgmm import CGMM
from torch_geometric.nn import Set2Set

from math import factorial as fact
import random


class GraphHTN(nn.Module):

    def __init__(self, outputs, n_gen, disaggregated, C, L, M, set2set_steps):
        super(GraphHTN, self).__init__()
        self.cgmm = CGMM()
        self.disaggregated = disaggregated
        self.contrastive = contrastive_matrix(n_gen)
        if disaggregated:
            self.set2one = Set2Set(self.contrastive.size(1), set2set_steps, num_layers=1)
            self.output = nn.Linear(2*self.contrastive.size(1), outputs)
        else:
            self.output = nn.Linear(self.contrastive.size(1), outputs)
    
    def forward(self, graph_batch):
        neg_log_likelihood = []
        batch_idx = []
        for idx, g in enumerate(graph_batch):
            g_dim = g.x.size(0)
            batch_idx += [idx for _ in range(g_dim)]

            g_neg_likelihood = self.cgmm(g).detach()
            neg_log_likelihood.append(g_neg_likelihood)

        c_neurons = (torch.stack(g_neg_likelihood, dim=0) @ self.contrastive).tanh()
        if self.disaggregated:
            g_pooling = self.set2one(c_neurons, batch_idx)
            output = self.output(g_pooling)
        else:
            output = self.output(c_neurons)
        
        return output, torch.stack(neg_log_likelihood).mean(0)

    def get_gen_parameters(self):
        return self.cgmm.parameters
    
    def stack_cgmm_layer(self):
        for p in self.cgmm.parameters:
            
