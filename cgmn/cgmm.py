import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter
import time

class CGMM(nn.Module):

    def __init__(self, n_gen, C, M, n_layers):
        super(CGMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        layers = [CGMMLayer_0(n_gen, C, M)] + [CGMMLayer(n_gen, C, M) for _ in n_layers-1]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, pos=None):
        log_likelihood = []
        h_prev = None

        for l in self.layers:
            if h_prev is None:
                l_log_likelihood, h_prev = l(x)
            else:
                if pos is None:
                    l_log_likelihood, h_prev = l(x, h_prev, edge_index)
                else:
                    l_log_likelihood, h_prev = l(x, h_prev, edge_index, pos)

            log_likelihood.append(l_log_likelihood)

        log_likelihood = torch.stack(log_likelihood, dim=1)   # nodes x L x n_gen
        
        return log_likelihood


class CGMMLayer_0(nn.Module):

    def __init__(self, n_gen, C, M):
        super(CGMMLayer_0, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=2.5))
        self.frozen = False
    
    def forward(self, x):
        return FirstLayerF.apply(x, self.B, self.Pi)


class FirstLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_B, lambda_Pi):
        B, Pi = [], []

        for j in range(lambda_B.size(-1)):
            B.append(F.softmax(lambda_B[:, :, j], dim=1))
            Pi.append(F.softmax(lambda_Pi[:, j], dim=0))

        B, Pi = torch.stack(B, dim=-1), torch.stack(Pi, dim=-1)

        unnorm_posterior = B[:, x].permute(1, 0, 2) * Pi.unsqueeze(0)
        posterior = (unnorm_posterior / (unnorm_posterior.sum(1, keepdim=True)))

        ctx.saved_input = x
        ctx.save_for_backward(posterior, B, Pi)

        return unnorm_posterior.sum(1).log(), posterior

    @staticmethod
    def backward(ctx, likelihood, posterior):
        x = ctx.saved_input
        posterior, B, Pi = ctx.saved_tensors

        post_nodes = posterior.permute(1, 0, 2)
        B_grad = scatter(post_nodes - post_nodes * B[:, x],
                         index=x,
                         dim=1,
                         out=torch.zeros_like(B, device=B.device))

        Pi_grad = posterior.sum(0) - posterior.size(0)*Pi

        return None, B_grad, Pi_grad



class CGMMLayer(nn.Module):

    def __init__(self, n_gen, C, M):
        super(CGMMLayer, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M
        
        self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, C, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
    
    def forward(self, x, prev_h, edge_index):
        return InwardOutward.apply(x, prev_h, edge_index, self.Q_neigh, self.B)


class InwardOutward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, prev_h, edge_index, lambda_Q, lambda_B):
        Q, B = [], []
        
        for j in range(lambda_Q.size(-1)):
            Q.append(F.softmax(lambda_Q[:, :, j], dim=0))
            B.append(F.softmax(lambda_B[:, :, j], dim=1))

        Q, B = torch.stack(Q, dim=-1), torch.stack(B, dim=-1)

        prev_h_neigh = prev_h[edge_index[1]]
        prev_h_neigh_aggr = scatter(prev_h_neigh, edge_index[0], dim=0, reduce='mean')

        B_nodes = B[:, x].permute(1, 0, 2).unsqueeze(2)   # nodes x C x 1 x n_gen
        prev_h_neigh_aggr = prev_h_neigh_aggr.unsqueeze(1) # nodes x 1 x C x n_gen
        unnorm_posterior = B_nodes * (Q * prev_h_neigh_aggr)

        posterior_il = (unnorm_posterior / unnorm_posterior.sum([1, 2], keepdim=True)) # nodes x C x C x n_gen
        posterior_i = posterior_il.sum(2)

        ctx.saved_input = x, edge_index
        ctx.save_for_backward(posterior_il, posterior_i, Q, B)

        return unnorm_posterior.sum([1, 2]).log(), posterior_i

    @staticmethod
    def backward(ctx, likelihood, posterior_i):
        x, edge_index = ctx.saved_input
        posterior_il, posterior_i, Q, B = ctx.saved_tensors

        post_neigh = scatter(posterior_i[edge_index[1]], edge_index[0], dim=0, reduce='mean').unsqueeze(1)
        Q_grad = (posterior_il - Q * post_neigh).sum(0)
        
        post_nodes = posterior_i.permute(1, 0, 2)
        B_grad = scatter(post_nodes - post_nodes * B[:, x],
                         index=x,
                         dim=1,
                         out=torch.zeros_like(B, device=B.device))

        return None, None, None, Q_grad, B_grad

class PositionalCGMMLayer(nn.Module):

    def __init__(self, n_gen, C, L, M):
        super(PositionalCGMMLayer, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M
        
        self.Q_neigh = nn.Parameter(nn.init.normal_(torch.empty((C, C, L, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))

        self.frozen = False
    
    def forward(self, x, prev_h, edge_index, pos):
        Q_neigh, B = self._softmax_reparameterization()
        
        prev_h_neigh = prev_h[edge_index[1]].unsqueeze(1)
        trans_neigh = Q_neigh[:, :, pos].permute(2, 0, 1, 3)
        
        B_nodes = B[:, x[edge_index[0]]].permute(1, 0, 2).unsqueeze(2)   # edges x C x 1 x n_gen
        unnorm_posterior = B_nodes * trans_neigh * prev_h_neigh # edges x C x C x n_gen
        likelihood = scatter(unnorm_posterior.sum([1, 2], keepdim=True), edge_index[0], dim=0, reduce='mean')
        
        posterior_il = (unnorm_posterior / (likelihood[edge_index[0]] + 1e-16)).detach() # edges x C x C x n_gen
        posterior_i = scatter(posterior_il.sum(2), index=edge_index[0], dim=0).detach() # nodes x C x n_gen
        if self.training and not self.frozen:
            B_nodes = B[:, x].permute(1, 0, 2)  # nodes x C x n_gen, necessary for backpropagating in the new, detached graph
            exp_likelihood = (posterior_il * trans_neigh.log()).sum() + (posterior_i * B_nodes.log()).sum()
            (-exp_likelihood).backward()

        likelihood = likelihood.log().squeeze()
        return likelihood, posterior_i

    def _softmax_reparameterization(self):
        sm_Q_neigh, sm_B = [], []
        
        for j in range(self.n_gen):
            sm_Q_neigh.append(torch.stack([F.softmax(self.Q_neigh[:, :, l, j], dim=0) for l in range(self.L)], dim=-1))
            sm_B.append(F.softmax(self.B[:, :, j], dim=1))

        sm_Q_neigh = torch.stack(sm_Q_neigh, dim=-1)
        sm_B = torch.stack(sm_B, dim=-1)
        
        return sm_Q_neigh, sm_B
