import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter
from torch_scatter.segment_coo import segment_coo

class UniformBottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, M, cuda=[]):
        super(UniformBottomUpHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.A = nn.Parameter(nn.init.uniform_(torch.empty((C, C, n_gen)), a=-5, b=5))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen)), a=-5, b=5))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, n_gen)), a=-5, b=5))
    

    def forward(self, x, trees):
        sm_A, sm_B, sm_Pi = _softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi)

        beta, t_beta = _reversed_upward(x, trees, self.n_gen, sm_A, sm_B, sm_Pi, self.C)
        eps, t_eps = _reversed_downward(x, trees, self.n_gen, sm_A, sm_Pi, beta, t_beta, self.C)

        log_likelihood = _log_likelihood(x, trees, sm_A, sm_B, sm_Pi, eps, t_eps)  # Negative log likelihood
            
        return -log_likelihood


def _softmax_reparameterization(n_gen, A, B, Pi):
    sm_A, sm_B, sm_Pi = [], [], []
    for i in range(n_gen):
        sm_A.append(F.softmax(A[:, :, i], dim=0))
        sm_B.append(F.softmax(B[:, :, i], dim=1))
        sm_Pi.append(F.softmax(Pi[:, i], dim=0))

    return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


def _reversed_upward(x, tree, n_gen, A, B, Pi, C):
    beta = torch.zeros((tree['dim'], C, n_gen))
    t_beta = torch.zeros((tree['dim'], C, n_gen))

    Pi_leaves = Pi.unsqueeze(1)
    leaves_idx = tree['inv_map'][tree['leaves']]
    B_leaves = B[:, x[leaves_idx]]
    beta_leaves = Pi_leaves * B_leaves
    beta_leaves = (beta_leaves / beta_leaves.sum(dim=0, keepdim=True)).permute(1, 0, 2)
    
    beta[tree['leaves']] = beta_leaves

    for l in reversed(tree['levels']):
        # Computing unnormalized beta_uv children = A_ch @ beta_ch
        beta_ch = beta[l[1]]
        t_beta_ch = (A.unsqueeze(0) * beta_ch.unsqueeze(1)).sum(2)
        t_beta = segment_coo(src=t_beta_ch, index=l[0], dim=0, out=t_beta, reduce="mean")

        u_idx = l[0].unique(sorted=False)
        B_l = B[:, x[tree['inv_map'][u_idx]]].permute(1, 0, 2)
        beta_l = t_beta[u_idx] * B_l
        beta_l = beta_l / (beta_l.sum(dim=1, keepdim=True))
        beta[u_idx] = beta_l

    return beta, t_beta


def _reversed_downward(g, tree, n_gen, A, Pi, beta, t_beta, C):
    eps = torch.zeros((tree['dim'], C, n_gen))
    t_eps = torch.zeros((tree['dim'], C, C, n_gen))

    eps[tree['roots']] = beta[tree['roots']]
    for l in tree['levels']:
        # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
        t_beta_pa = t_beta[l[0]].unsqueeze(2)
        eps_pa = eps[l[0]].unsqueeze(2)
        beta_ch = beta[l[1]].unsqueeze(1)
        eps_joint = (eps_pa * A.unsqueeze(0) * beta_ch) / t_beta_pa
        t_eps = segment_coo(src=eps_joint, index=l[0], dim=0, out=t_eps, reduce="mean")
        eps[l[1]] = eps_joint.sum(1)

    return eps, t_eps


def _log_likelihood(x, tree, A, B, Pi, eps, t_eps):
    internal = torch.cat([l[0].unique(sorted=False) for l in tree['levels']])
    all_nodes = torch.cat([internal, tree['leaves']])
    l_hood_size = eps.size(0), eps.size(-1)

    likelihood = torch.zeros(l_hood_size)
    # Likelihood A
    likelihood[internal] += (t_eps[internal] * A.log().unsqueeze(0)).sum([1, 2])

    # Likelihood B
    all_nodes_mappings = tree['inv_map'][all_nodes]
    B_nodes = B[:, x[all_nodes_mappings]].permute(1, 0, 2)
    likelihood[all_nodes] += (eps[all_nodes] * B_nodes.log()).sum(1)

    # Likelihood Pi
    likelihood[tree['leaves']] += (eps[tree['leaves']] * Pi.unsqueeze(0).log()).sum(1)

    return segment_coo(likelihood, tree['trees_ind'], dim=0)
