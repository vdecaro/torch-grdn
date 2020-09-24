import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter
import time

class CGMM(nn.Module):

    def __init__(self, n_gen, C, M, device='cpu:0'):
        super(CGMM, self).__init__()
        self.device = torch.device(device)
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.layers = nn.ModuleList([CGMMLayer_0(n_gen, C, M, device)])

    def forward(self, x, edge_index):
        log_likelihood = []
        h_prev = []
        for i, l in enumerate(self.layers):
            if i == 0:
                l_log_likelihood, l_h_state = l(x)
            else:
                l_log_likelihood, l_h_state = l(x, h_prev[i-1], edge_index)
            h_prev.append(l_h_state)
            log_likelihood.append(l_log_likelihood)

        log_likelihood = torch.stack(log_likelihood, dim=1)   # nodes x L x n_gen
        
        return log_likelihood

    def stack_layer(self):
        for p in self.layers[-1].parameters():
            p.requires_grad = False
        self.layers[-1].frozen = True
        self.layers.append(CGMMLayer(self.n_gen, self.C, self.M, self.device))


class CGMMLayer_0(nn.Module):

    def __init__(self, n_gen, C, M, device):
        super(CGMMLayer_0, self).__init__()
        self.device = device
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=1.25))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=1.25))
        self.to(device=self.device)
        self.frozen = False
    
    def forward(self, x):
        B, Pi = self._softmax_reparameterization()

        unnorm_posterior = Pi.unsqueeze(0) * B[:, x].permute(1, 0, 2) + 1e-8
        posterior = unnorm_posterior / (unnorm_posterior.sum(1, keepdim=True))

        if self.training and not self.frozen:
            exp_likelihood = (posterior * (Pi.unsqueeze(0).log() + B[:, x].permute(1, 0, 2).log())).sum()
            (-exp_likelihood).backward()

        log_likelihood = unnorm_posterior.sum(1).log()
        return log_likelihood, posterior

    def _softmax_reparameterization(self):
        sm_B, sm_Pi = [], []

        for j in range(self.n_gen):
            sm_B.append(F.softmax(self.B[:, :, j], dim=1))
            sm_Pi.append(F.softmax(self.Pi[:, j], dim=0))

        return torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


class CGMMLayer(nn.Module):

    def __init__(self, n_gen, C, M, device):
        super(CGMMLayer, self).__init__()
        self.device = device
        self.n_gen = n_gen
        self.C = C
        self.M = M
        
        self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, C, n_gen)), std=1.25))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=1.25))

        self.to(device=self.device)
        self.frozen = False
    
    def forward(self, x, prev_h, edge_index):
        Q_neigh, B = self._softmax_reparameterization()
        
        prev_h_neigh = prev_h[edge_index[1]]
        prev_h_neigh_aggr = scatter(prev_h_neigh, edge_index[0], dim=0, reduce='mean')
        
        B_nodes = B[:, x].permute(1, 0, 2).unsqueeze(2)   # nodes x C x 1 x n_gen
        prev_h_neigh_aggr = prev_h_neigh_aggr.unsqueeze(1) # nodes x 1 x C x n_gen
        unnorm_posterior = B_nodes * (Q_neigh * prev_h_neigh_aggr) + 1e-8
        
        posterior_il = unnorm_posterior / unnorm_posterior.sum([1, 2], keepdim=True) # nodes x C x C x n_gen
        posterior_i = posterior_il.sum(2)
        if self.training and not self.frozen:
            B_nodes = B[:, x].permute(1, 0, 2)  # nodes x C x n_gen, necessary for backpropagating in the new, detached graph
            exp_likelihood = (posterior_il * Q_neigh.log().unsqueeze(0)).sum() + (posterior_i * B_nodes.log()).sum()
            (-exp_likelihood).backward()

        likelihood = unnorm_posterior.sum([1, 2]).log()
        return likelihood, posterior_i

    def _softmax_reparameterization(self):
        sm_Q_neigh, sm_B = [], []
        
        for j in range(self.n_gen):
            sm_Q_neigh.append(F.softmax(self.Q_neigh[:, :, j], dim=0))
            sm_B.append(F.softmax(self.B[:, :, j], dim=1))

        sm_Q_neigh = torch.stack(sm_Q_neigh, dim=-1)
        sm_B = torch.stack(sm_B, dim=-1)

        return sm_Q_neigh, sm_B
