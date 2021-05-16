import torch
import torch.nn as nn

from htmn.htmm import HiddenTreeMarkovModel

class HTMN(nn.Module):

    def __init__(self, out, mode, n_gen, C, L, M):
        super(HTMN, self).__init__()
        self.htmm = HiddenTreeMarkovModel(mode, n_gen, C, L, M)
        self.contrastive = nn.Parameter(_contrastive_matrix(n_gen), requires_grad=False)
        self.output = nn.Linear(self.contrastive.size(1), out, bias=False)
    
    def forward(self, tree):
        log_likelihood = self.htmm(tree)
        c_neurons = (log_likelihood @ self.contrastive).tanh()
        out = self.output(c_neurons)

        return out

def _contrastive_matrix(N_GEN):
    contrastive_units = (N_GEN*(N_GEN-1)) // 2
    contrastive_matrix = torch.zeros(N_GEN, contrastive_units)

    p = 0
    s = 1
    for i in range(contrastive_units):
        contrastive_matrix[p, i] = 1
        contrastive_matrix[s, i] = -1
        if s == N_GEN - 1:
            p += 1
            s = p
        s += 1

    return contrastive_matrix