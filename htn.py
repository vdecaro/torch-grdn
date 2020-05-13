import torch
import torch.nn as nn
from math import factorial as fact

from genmodel.bhtmm import BottomUpHTMM
from genmodel.thtmm import TopDownHTMM


class HTN(nn.Module):

    def __init__(self, n_bu, n_td, C, L, M):
        super(HTN, self).__init__()
        self.bu = [BottomUpHTMM(C, L, M) for _ in n_bu]
        self.td = [TopDownHTMM(C, L, M) for _ in n_td]

        self.contrastive = _contrastive_matrix(n_bu + n_td)

    
    def forward(self, tree_batch):
        bu_likelihood = torch.tensor([[bu(tree) for bu in self.bu] for tree in tree_batch])
        td_likelihood = torch.tensor([[td(tree) for td in self.td] for tree in tree_batch])

        likelihood = torch.cat([bu_likelihood, td_likelihood], dim=0)
        c_neurons = (likelihood @ self.contrastive).tanh()

        return c_neurons


def _contrastive_matrix(N_GEN):
    contrastive_units = fact(N_GEN) // (2*fact(N_GEN-2))
    contrastive_matrix = torch.zeros((N_GEN, contrastive_units))

    p = 0
    s = 1
    for i in range(contrastive_units):
        contrastive_matrix[p, i] = 1
        contrastive_matrix[s, i] = -1
        if s == N_GEN - 1:
            p = p + 1
            s = p
        s = s + 1
    return contrastive_matrix
