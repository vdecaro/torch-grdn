import torch
import torch.nn as nn

from contrastive import contrastive_matrix

from cgmn.cgmm import CGMM


class CGMN(nn.Module):

    def __init__(self, out_features, n_cgmm, C_0, M, cgmm=None):
        super(CGMN, self).__init__()
        if cgmm is None:
            self.cgmm = CGMM(n_cgmm, C_0, M)
        else:
            self.cgmm = cgmm

        self.contrastive = contrastive_matrix(self.cgmm.n_gen)
        self.output_backup = None
        self.b_norm = [nn.BatchNorm1d(self.cgmm.n_gen, affine=False)]
        self.output = nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), out_features)
    
    def forward(self, x, edge_index, batch):
        neg_likelihood = self.cgmm(x, edge_index, batch)
        b_norm_lhood = torch.stack([b(neg_likelihood[:, i]) for i, b in enumerate(self.b_norm)], dim=1)
        c_neurons = (b_norm_lhood @ self.contrastive).tanh().detach_()
        c_neurons = c_neurons.flatten(start_dim=-2)
        output = self.output(c_neurons)
        return output, neg_likelihood.mean(0).sum()

    def stack_layer(self, C):
        self.cgmm.stack_layer(C)
        self.b_norm.append(nn.BatchNorm1d(self.cgmm.n_gen, affine=False))
        self.output_backup = self.output
        self.output = nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), self.output_backup.out_features)
    
    def rollback(self):
        self.cgmm.layers = self.cgmm.layers[:-1]
        self.output = self.output_backup
    
    def get_parameters(self):
        params = []
        params += [p for p in self.parameters()]
        params += [p for p in self.cgmm.get_parameters()]
        return params