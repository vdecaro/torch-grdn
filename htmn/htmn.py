import torch
import torch.nn as nn

from htmn.bhtmm import BottomUpHTMM
from htmn.pos_thtmm import PositionalTopDownHTMM


class HTMN(nn.Module):

    def __init__(self, out, n_bu, n_td, C, L, M):
        super(HTMN, self).__init__()
        self.bu = BottomUpHTMM(n_bu, C, L, M) if n_bu is not None and n_bu > 0 else None
        self.td = PositionalTopDownHTMM(n_td, C, L, M) if n_td is not None and n_td > 0 else None
        
        self.b_norm = nn.BatchNorm1d(n_bu + n_td, affine=False)

        self.contrastive = nn.Parameter(_contrastive_matrix(n_bu + n_td), requires_grad=False)
        self.output = nn.Linear(self.contrastive.size(1), out)
    
    def forward(self, tree):
        log_likelihood = []
        if self.bu is not None:
            log_likelihood.append(self.bu(tree))
        if self.td is not None:
            log_likelihood.append(self.td(tree))
        
        to_contrastive = torch.cat(log_likelihood, 1)
        to_contrastive = self.b_norm(to_contrastive)

        c_neurons = (to_contrastive @ self.contrastive).tanh().detach()
        output = self.output(c_neurons)

        return output

def _contrastive_matrix(N_GEN):
    contrastive_units = (N_GEN*(N_GEN-1)) // 2
    contrastive_matrix = torch.zeros(N_GEN, contrastive_units)

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