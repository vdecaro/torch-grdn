import torch
import torch.nn as nn

from graph_htmn.uni_bhtmm import UniformBottomUpHTMM
from graph_htmn.thtmm import TopDownHTMM
from torch_geometric.nn import Set2Set, BatchNorm, GlobalAttention
from torch_scatter.scatter import scatter


class GraphHTMN(nn.Module):

    def __init__(self, out_features: int, n_bu: int, n_td: int, C: int, M: int, tree_dropout: float):
        super(GraphHTMN, self).__init__()
        self.bu = UniformBottomUpHTMM(n_bu, C, M, tree_dropout) if n_bu > 0 else None
        self.td = TopDownHTMM(n_td, C, M, tree_dropout) if n_td > 0 else None

        self.b_norm = BatchNorm(n_bu + n_td, affine=False)

        self.contrastive = nn.Parameter(_contrastive_matrix(n_bu + n_td), requires_grad=False)
        self.pooling = Set2Set(self.contrastive.size(1), 2, 1)
        self.output = nn.Linear(2*self.contrastive.size(1), out_features)
        
    
    def forward(self, x, trees, batch):
        to_contrastive = []
        if self.bu is not None:
            to_contrastive += [self.bu(x, trees, batch)]

        if self.td is not None:
            to_contrastive += [self.td(x, trees, batch)]

        if len(to_contrastive) == 2:
            to_contrastive = torch.cat(to_contrastive, dim=1)
        else:
            to_contrastive = to_contrastive[0]

        to_contrastive = self.b_norm(to_contrastive)
        c_neurons = (to_contrastive @ self.contrastive).tanh().detach()
        g_pooling = self.pooling(c_neurons, batch)
        output = self.output(g_pooling)
        
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