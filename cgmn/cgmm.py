import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter


class CGMM(nn.Module):

    def __init__(self, n_gen, C_0, M):
        super(CGMM, self).__init__()
        self.n_gen = n_gen
        self.M = M

        self.layers = [CGMMLayer_0(n_gen, C_0, self.M)]

    def forward(self, x, edge_index, batch):
        h_states = []
        likelihood = []
        
        for i, l in enumerate(self.layers):
            l_likelihood, l_h_states = l(x, h_states, edge_index, batch) if i > 0 else l(x, batch)
            likelihood.append(l_likelihood)
            h_states.append(l_h_states)
        
        return -torch.stack(likelihood, dim=1)   # nodes x L x n_gen
        
    def stack_layer(self, C):
        self.layers[-1].freeze()
        self.layers.append(CGMMLayer(self.n_gen, C, [l.C for l in self.layers], len(self.layers), self.M))
    
    def get_parameters(self):
        return [p for p in self.layers[-1].parameters()]

class CGMMLayer_0(nn.Module):

    def __init__(self, n_gen, C, M):
        super(CGMMLayer_0, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=2))

        self.frozen = False

    
    def forward(self, x, batch):
        sm_B, sm_Pi = self._softmax_reparameterization()
        
        numerator = sm_Pi.unsqueeze(0) * sm_B[:, x].permute(1, 0, 2)
        posterior = numerator / numerator.sum(1, keepdim=True)
        
        likelihood = (posterior * numerator.log()).sum(1)
        likelihood = scatter(likelihood, batch, dim=0)

        h_states = posterior.argmax(dim=1)

        return likelihood, h_states

    def _softmax_reparameterization(self):
        sm_B, sm_Pi = [], []

        for j in range(self.n_gen):
            sm_B.append(F.softmax(self.B[:, :, j], dim=1))
            sm_Pi.append(F.softmax(self.Pi[:, j], dim=0))

        return torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True



class CGMMLayer(nn.Module):

    def __init__(self, n_gen, C, C_prev, L, M):
        super(CGMMLayer, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.C_prev = C_prev
        self.L = L
        self.M = M
        
        self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, sum(C_prev), n_gen)), std=2))
        print(self.Q_neigh.size())
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2))
        if self.L > 1:
            self.SP = nn.Parameter(nn.init.normal_(torch.empty((L, n_gen)), std=2))
        else:
            self.SP = 1

        self.frozen = False
    
    def forward(self, x, prev_h, edge_index, batch):
        sm_Q_neigh, sm_B, sm_SP = self._softmax_reparameterization()
        # Preparation of the transition distribution by multiplying it for the SP variable
        trans_i_neigh_l = []
        trans_i_neigh = []
        
        for m in range(self.n_gen):
            tmp_trans_i_neigh_l = []
            tmp_trans_i_neigh = []
            i = 0
            for L, C in enumerate(self.C_prev):
                trans_distr = sm_Q_neigh[:, i:i+C, m]
                trans_distr = trans_distr[:, prev_h[L][:, m]]
                tmp_trans_i_neigh_l.append(sm_SP[L, m] * trans_distr if self.L > 1 else trans_distr)  # produces a list of elements [C x nodes x n_gen]
                tmp_trans_i_neigh.append(trans_distr)
                i += C
            trans_i_neigh_l.append(torch.stack(tmp_trans_i_neigh_l, dim=2))
            trans_i_neigh.append(torch.stack(tmp_trans_i_neigh, dim=2)) 
        trans_i_neigh_l = torch.stack(trans_i_neigh_l, dim=3).permute(1, 0, 2, 3) # nodes x C x L x n_gen
        trans_i_neigh = torch.stack(trans_i_neigh, dim=3).permute(1, 0, 2, 3) # nodes x C x L x n_gen
        
        # Joint transition
        trans_i_neigh_l_edges = trans_i_neigh_l[edge_index[1]] # n_edges x C x L x n_gen
        Qu_i = scatter(trans_i_neigh_l_edges, edge_index[0], dim=0, reduce='mean') # nodes x C x L x n_gen

        # Emission
        nodes_emissions = sm_B[:, x].permute(1, 0, 2)   # nodes x C x n_gen

        # Posterior computation
        posterior_unnorm = Qu_i * nodes_emissions.unsqueeze(2)
        posterior_il = posterior_unnorm / (posterior_unnorm.sum((1, 2), keepdim=True)+1e-12) # nodes x C x L x n_gen
        posterior_i = posterior_il.sum(2)
        
        # Likelihood computation
        Q_neigh_lhood = (posterior_il * trans_i_neigh.log()).sum((1, 2))
        B_lhood = (posterior_i * nodes_emissions.log()).sum(1)
        
        likelihood = Q_neigh_lhood + B_lhood

        if self.L > 1:
            SP_lhood = (posterior_il.sum(1) * sm_SP.unsqueeze(0).log()).sum(1)
            likelihood += SP_lhood

        likelihood = scatter(likelihood, batch, dim=0)
        
        h_states = posterior_i.argmax(dim=1)
        
        return likelihood, h_states

    def _softmax_reparameterization(self):
        sm_Q_neigh, sm_B, sm_SP = [], [], []
        
        for j in range(self.n_gen):

            sm_Q_neigh_j = []
            idx = 0
            for C in self.C_prev:
                sm_Q_neigh_j.append(F.softmax(self.Q_neigh[:, idx:idx+C, j], dim=0))
                idx += C
            sm_Q_neigh.append(torch.cat(sm_Q_neigh_j, dim=1))

            sm_B.append(F.softmax(self.B[:, :, j], dim=1))

            if self.L > 1:
                sm_SP.append(F.softmax(self.SP[:, j], dim=0))
            else:
                sm_SP = 1

        sm_Q_neigh = torch.stack(sm_Q_neigh, dim=-1)
        sm_B = torch.stack(sm_B, dim=-1)
        sm_SP = torch.stack(sm_SP, dim=-1) if self.L > 1 else 1

        return sm_Q_neigh, sm_B, sm_SP

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True
