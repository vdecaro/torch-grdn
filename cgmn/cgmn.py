import torch
import torch.nn as nn

from cgmn.cgmm import CGMM
from torch_geometric.nn import GlobalAttention
from torch_scatter.scatter import scatter

class CGMN(nn.Module):

    def __init__(self, out_features, n_gen, C, M, n_layers):
        super(CGMN, self).__init__()
        self.cgmm = CGMM(n_gen, C, M, n_layers)
        self.contrastive = nn.Parameter(_contrastive_matrix(self.cgmm.n_gen), requires_grad=False)
        self.node_features = self.contrastive.size(1)
        self.output = nn.Linear(self.contrastive.size(1)*n_layers, out_features, bias=False)
   
    def forward(self, x, edge_index, batch, pos=None):
        log_likelihood = self.cgmm(x, edge_index, pos)
        log_likelihood = scatter(log_likelihood, batch, dim=0)

        c_neurons = (log_likelihood @ self.contrastive).tanh().flatten(start_dim=-2)

        out = self.output[-1](c_neurons)

        return out

def _contrastive_matrix(N_GEN):
    contrastive_units = (N_GEN * (N_GEN-1)) // 2
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