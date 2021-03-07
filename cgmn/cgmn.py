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
        self.h0 = nn.Parameter(nn.init.uniform_(torch.empty((1, 1, self.contrastive.size(1)))))
        self.gru = nn.GRU(self.contrastive.size(1), self.contrastive.size(1), batch_first=True)
        self.output = nn.Linear(self.contrastive.size(1)*n_layers, out_features, bias=False)
   
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        log_likelihood = self.cgmm(x, edge_index)
        log_likelihood = scatter(log_likelihood, batch, dim=0)

        c_neurons = (log_likelihood @ self.contrastive).tanh()
        to_out, _ = self.gru(c_neurons, self.h0.repeat(1, batch.max()+1, 1))

        out = self.output(to_out.flatten(start_dim=-2))
        
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