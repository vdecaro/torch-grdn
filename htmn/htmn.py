import torch
import torch.nn as nn

from htmn.bhtmm import BottomUpHTMM
from htmn.pos_thtmm import PositionalTopDownHTMM

from contrastive import contrastive_matrix

class HTMN(nn.Module):

    def __init__(self, out, n_bu, n_td, C, L, M, b_norm=True, device='cpu:0'):
        super(HTMN, self).__init__()
        self.device = torch.device(device)
        self.bu = BottomUpHTMM(n_bu, C, L, M, device) if n_bu is not None and n_bu > 0 else None
        self.td = PositionalTopDownHTMM(n_td, C, L, M, device) if n_td is not None and n_td > 0 else None
        
        self.b_norm = nn.BatchNorm1d(n_bu + n_td, affine=False) if b_norm else None

        self.contrastive = contrastive_matrix(n_bu + n_td, self.device)
        self.output = nn.Linear(self.contrastive.size(1), out)
        self.to(device=self.device)
    
    def forward(self, tree):
        log_likelihood = []
        if self.bu is not None:
            log_likelihood.append(self.bu(tree))
        if self.td is not None:
            log_likelihood.append(self.td(tree))
        
        to_contrastive = torch.cat(log_likelihood, 1).detach()
        if self.b_norm is not None:
            to_contrastive = self.b_norm(to_contrastive)

        c_neurons = (to_contrastive @ self.contrastive).tanh()
        output = self.output(c_neurons)

        return output
