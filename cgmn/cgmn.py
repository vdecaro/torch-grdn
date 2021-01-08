import torch
import torch.nn as nn

from cgmn.cgmm import CGMM
from torch_geometric.nn import GlobalAttention
from torch_scatter.scatter import scatter

class CGMN(nn.Module):

    def __init__(self, out_features, n_gen, C, L=None, M=None, gate_units=None):
        super(CGMN, self).__init__()
        self.cgmm = CGMM(n_gen, C, L, M)
        self.contrastive = nn.Parameter(_contrastive_matrix(self.cgmm.n_gen), requires_grad=False)
        self.node_features = self.contrastive.size(1)
        self.gate_units = gate_units
        
        if gate_units > 0:
            self.pooling = nn.ModuleList([GlobalAttention(_GateNN(self.node_features, gate_units))])
        self.output = nn.ModuleList([nn.Linear(self.node_features, out_features)])

    
    def forward(self, x, edge_index, batch, pos=None):
        if self.gate_units > 0:
            return self.fw_with_att(x, edge_index, batch, pos)
        else:
            return self.fw_no_att(x, edge_index, batch, pos)
    
    def fw_no_att(self, x, edge_index, batch, pos=None):
        log_likelihood = self.cgmm(x, edge_index, pos)
        log_likelihood = scatter(log_likelihood, batch, dim=0)
        c_neurons = (log_likelihood @ self.contrastive).tanh().flatten(start_dim=-2).detach()
        output = self.output[-1](c_neurons)
        return output
    
    def fw_with_att(self, x, edge_index, batch, pos=None):
        log_likelihood = self.cgmm(x, edge_index, pos)
        c_neurons = (log_likelihood @ self.contrastive).tanh().detach()
        
        r_i = []
        for i, att in enumerate(self.pooling):
            if i < len(self.pooling)-1:
                with torch.no_grad():
                    r_i.append(att(c_neurons[:, i], batch))
            else:
                r_i.append(att(c_neurons[:, i], batch))
        r_i = torch.cat(r_i, -1)
        
        output = self.output[-1](r_i)
        return output
    
    def stack_layer(self):
        self.cgmm.stack_layer()
        
        if self.gate_units > 0:
            self.pooling.append(GlobalAttention(_GateNN(self.node_features, self.gate_units)))
            self.pooling[-1].to(device=self.pooling[-2])
        
        self.output.append(nn.Linear(self.node_features*len(self.cgmm.layers), self.output[-1].out_features))
        self.output[-1].to(device=self.device)
    
    def get_params(self):
        if self.gate_units > 0:
            o_params = list(self.output.parameters())
            p_params = list(self.pooling.parameters())
            return list(self.cgmm.parameters()), (o_params + p_params)
        else:
            return list(self.cgmm.parameters()), list(self.output.parameters())


class _GateNN(nn.Module):
    def __init__(self, input_features, gate_units):
        super(_GateNN, self).__init__()
        self.h = nn.Linear(input_features, gate_units)
        self.out = nn.Linear(gate_units, 1)

    def forward(self, x):
        return self.out(self.h(x).tanh())


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