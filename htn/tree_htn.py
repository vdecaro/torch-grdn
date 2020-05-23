import torch
import torch.nn as nn

from genmodel.bhtmm import BottomUpHTMM
from genmodel.thtmm import TopDownHTMM

from htn.contrastive import contrastive_matrix

class HTN(nn.Module):

    def __init__(self, n_bu, n_td, C, L, M, outputs):
        super(HTN, self).__init__()
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
        self.output = nn.Linear(self.contrastive.size(1), outputs)
    
    def forward(self, tree_batch):
        if self.mode == 'both':
            neg_td_likelihood = self.td(tree_batch)
            neg_bu_likelihood = self.bu(tree_batch)
            norm_td = self.td_batch_norm(neg_td_likelihood)
            norm_bu = self.bu_batch_norm(neg_bu_likelihood)
            to_contrastive = torch.cat([norm_td, norm_bu], dim=1)
            neg_log_likelihood = torch.cat([neg_td_likelihood, neg_bu_likelihood], dim=1)

        elif self.mode == 'bu':
            to_contrastive = self.bu(tree_batch)
            neg_log_likelihood = to_contrastive

        elif self.mode == 'td':
            to_contrastive = self.td(tree_batch)
            neg_log_likelihood = to_contrastive
        
        c_neurons = (to_contrastive @ self.contrastive).tanh()
        output = self.output(c_neurons)

        return output, neg_log_likelihood.mean(dim=0).detach()

    def get_gen_parameters(self):
        params = []
        for htmm in self.td + self.bu:
            params += htmm.parameters()

        return params
