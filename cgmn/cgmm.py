import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter


class CGMM(nn.Module):

    def __init__(self, n_gen, C, M, device='cpu:0'):
        super(CGMM, self).__init__()
        self.device = torch.device(device)
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.layers = nn.ModuleList([CGMMLayer_0(n_gen, C, M, device)])

    def forward(self, x, edge_index, batch, likelihood_prev=None, h_prev=None):
        if likelihood_prev is None and h_prev is None:
            h_value = []
            h_ind = []
            likelihood = []

            for i, l in enumerate(self.layers):
                if i == 0:
                    l_likelihood, l_h_value, l_h_ind = l(x, batch)
                else:
                    h_prev = torch.stack(h_value, dim=2), torch.stack(h_ind, dim=1)
                    l_likelihood, l_h_value, l_h_ind = l(x, h_prev, edge_index, batch)
                    
                likelihood.append(l_likelihood)
                h_value.append(l_h_value)
                h_ind.append(l_h_ind)
            likelihood = torch.stack(likelihood, dim=1)   # nodes x L x n_gen

        else:
            l_likelihood, _ = l(x, h_prev, edge_index, batch)
            likelihood = torch.cat([likelihood_prev, l_likelihood.unsqueeze(1)], dim=1)
            
        return -likelihood
        
    def stack_layer(self):
        for p in self.layers[-1].parameters():
            p.requires_grad = False
        self.layers.append(CGMMLayer(self.n_gen, self.C, len(self.layers), self.M, self.device))


class CGMMLayer_0(nn.Module):

    def __init__(self, n_gen, C, M, device):
        super(CGMMLayer_0, self).__init__()
        self.device = device
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen)), a=-5, b=5))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, n_gen)), a=-5, b=5))
        self.to(device=self.device)
    
    def forward(self, x, batch):
        sm_B, sm_Pi = self._softmax_reparameterization()
        
        numerator = sm_Pi.unsqueeze(0) * sm_B[:, x].permute(1, 0, 2)
        posterior = numerator / numerator.sum(1, keepdim=True)
        
        likelihood = (posterior * numerator.log()).sum(1)
        likelihood = scatter(likelihood, batch, dim=0)

        h_max = posterior.max(dim=1, keepdim=True)

        return likelihood, h_max[0], h_max[1].squeeze()

    def _softmax_reparameterization(self):
        sm_B, sm_Pi = [], []

        for j in range(self.n_gen):
            sm_B.append(F.softmax(self.B[:, :, j], dim=1))
            sm_Pi.append(F.softmax(self.Pi[:, j], dim=0))

        return torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


class CGMMLayer(nn.Module):

    def __init__(self, n_gen, C, L, M, device):
        super(CGMMLayer, self).__init__()
        self.device = device
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M
        
        self.Q_neigh = nn.Parameter(nn.init.uniform_(torch.empty((C, C, L, n_gen)),  a=-5, b=5))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen)),  a=-5, b=5))
        if self.L > 1:
            self.SP = nn.Parameter(nn.init.uniform_(torch.empty((L, n_gen)),  a=-5, b=5))
        else:
            self.SP = 1

        self.to(device=self.device)
    
    def forward(self, x, prev_h, edge_index, batch):
        sm_Q_neigh, sm_B, sm_SP = self._softmax_reparameterization()

        trans_i = [[] for _ in range(self.n_gen)]
        for m in range(self.n_gen):
            trans_i_m = sm_Q_neigh[:, :, :, m]
            h_ind_m = prev_h[1][:, :, m]
            for l in range(self.L):
                trans_i[m].append(trans_i_m[:, h_ind_m[:, l], l])

        trans_i = torch.stack([torch.stack(l, dim=-1) for l in trans_i], dim=-1).permute(1, 0, 2, 3)
        trans_i_neigh = trans_i * prev_h[0]
        trans_i_neigh = trans_i_neigh * sm_SP.unsqueeze(0).unsqueeze(0) if self.L > 1 else trans_i_neigh

        # Joint transition
        Qu_i = scatter(trans_i_neigh[edge_index[1]], edge_index[0], dim=0, reduce='mean') # nodes x C x L x n_gen

        # Emission
        nodes_emissions = sm_B[:, x].permute(1, 0, 2)   # nodes x C x n_gen

        # Posterior computation
        posterior_unnorm = Qu_i * nodes_emissions.unsqueeze(2)
        posterior_il = posterior_unnorm / (posterior_unnorm.sum((1, 2), keepdim=True)+1e-12) # nodes x C x L x n_gen
        posterior_i = posterior_il.sum(2)
        
        # Q_neigh Likelihood
        Q_neigh_lhood = (posterior_il * trans_i.log()).sum((1, 2))

        # B Likelihood
        B_lhood = (posterior_i * nodes_emissions.log()).sum(1)
        
        likelihood = Q_neigh_lhood + B_lhood

        # SP Likelihood
        if self.L > 1:
            SP_lhood = (posterior_il.sum(1) * sm_SP.unsqueeze(0).log()).sum(1)
            likelihood += SP_lhood

        likelihood = scatter(likelihood, batch, dim=0)
        
        h_max = posterior_i.max(dim=1, keepdim=True)

        return likelihood, h_max[0], h_max[1].squeeze()

    def _softmax_reparameterization(self):
        sm_Q_neigh, sm_B, sm_SP = [], [], []
        
        for j in range(self.n_gen):
            sm_Q_neigh_j = []
            for l in range(self.L):
                sm_Q_neigh_j.append(F.softmax(self.Q_neigh[:, :, l, j], dim=0))
            sm_Q_neigh.append(torch.stack(sm_Q_neigh_j, dim=-1))

            sm_B.append(F.softmax(self.B[:, :, j], dim=1))

            if self.L > 1:
                sm_SP.append(F.softmax(self.SP[:, j], dim=0))

        sm_Q_neigh = torch.stack(sm_Q_neigh, dim=-1)
        sm_B = torch.stack(sm_B, dim=-1)
        sm_SP = torch.stack(sm_SP, dim=-1) if self.L > 1 else 1

        return sm_Q_neigh, sm_B, sm_SP
