import torch
import torch.nn as nn

from contrastive import contrastive_matrix

from cgmn.cgmm import CGMM


class CGMN(nn.Module):

    def __init__(self, out_features, n_gen, C, M, cgmm=None, device='cpu:0'):
        super(CGMN, self).__init__()
        self.device = torch.device(device)
        if cgmm is None:
            self.cgmm = CGMM(n_gen, C, M, device)
        else:
            self.cgmm = cgmm
        
        self.b_norm = nn.ModuleList([nn.BatchNorm1d(self.cgmm.n_gen, affine=False)])
        self.contrastive = contrastive_matrix(self.cgmm.n_gen, self.device)

        self.output_backup = None
        self.output = nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), out_features)
    
    def forward(self, x, edge_index, batch):
        print("CGMN---1")
        neg_likelihood = self.cgmm(x, edge_index, batch)
        print("CGMN---2")
        b_norm_lhood = torch.stack([b(neg_likelihood[:, i]) for i, b in enumerate(self.b_norm)], dim=1)

        c_neurons = (b_norm_lhood @ self.contrastive).tanh().detach_()
        c_neurons = c_neurons.flatten(start_dim=-2)
        output = self.output(c_neurons)
        return output, neg_likelihood.mean(0).sum()

    def stack_layer(self):
        self.cgmm.stack_layer()
        self.b_norm.append(nn.BatchNorm1d(self.cgmm.n_gen, affine=False))
        self.output_backup = self.output
        self.output = nn.Linear(self.contrastive.size(1) * len(self.cgmm.layers), self.output_backup.out_features)
    
    def rollback(self):
        self.cgmm.layers = self.cgmm.layers[:-1]
        self.output = self.output_backup
