import torch
import torch.nn as nn
import torch.nn.functional as F


class TopDownHTMM(nn.Module):

    def __init__(self, C, L, M):
        super(TopDownHTMM, self).__init__()
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L))))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C))))
    

    def forward(self, tree):
        sm_A, sm_B, sm_Pi = _softmax_reparameterization(self.A, self.B, self.Pi)

        prior = _preliminary_downward(tree, sm_A, sm_Pi, self.C)
        beta, t_beta = _upward(tree, sm_A, sm_B, prior)
        eps, t_eps = _downward(tree, sm_A, sm_Pi, prior, beta, t_beta)

        log_likelihood = _log_likelihood(tree, sm_A, sm_B, sm_Pi, eps, t_eps)

        return log_likelihood


def _softmax_reparameterization(A, B, Pi):
    sm_A = torch.cat([F.softmax(A[:, :, i], dim=0).unsqueeze(2) for i in range(A.size(2))], dim=2)
    sm_B = F.softmax(B, dim=1)
    sm_Pi = F.softmax(Pi, dim=0)

    return sm_A, sm_B, sm_Pi


def _preliminary_downward(tree, A, Pi, C):
    root = tree['levels'][0][0, 0]
    prior = torch.zeros((tree['dim'], C))
    prior[root] = Pi

    for l in tree['levels']:
        pos_ch = tree['pos'].index_select(0, l[:, 1])
        A_ch = A.index_select(dim=2, index=pos_ch)
        prior_pa = prior.index_select(dim=0, index=l[:, 0])
        prior_l = A_ch @ prior_pa.T
        prior = prior.scatter_(dim=0, index=l[:, 1], src=prior_l)
    
    return prior
        

def _upward(tree, A, B, prior):
    b_u = B.index_select(dim=1, index=tree['label'])
    beta = prior * b_u.T
    t_beta = torch.zeros(beta.size())

    beta_leaves = beta.index_select(dim=0, index=tree['leaves'])
    beta_leaves = beta_leaves / beta_leaves.sum(dim=1, keepdim=True)
    beta = beta.scatter_(dim=0, index=tree['leaves'], src=beta_leaves)

    for l in reversed(tree['levels']):
        # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
        pos_ch = tree['pos'].index_select(0, l[:, 1])
        beta_ch = beta.index_select(0, l[:, 1])
        A_ch = A.index_select(2, pos_ch).permute(2, 1, 0) # Permutation needed because of the application of bayes rule (computing u knowing the child v)
        prior_l = prior.index_select(0, l[:, 0])
        beta_uv = (A_ch @ beta_ch) / prior_l
        t_beta = t_beta.scatter_(dim=0, index=l[:, 1], src=beta_uv)
        
        # Computing beta on level = (\prod_ch beta_uv_ch) * prior_u * 
        for u in l[:, 0].unique(sorted=False):
            ch_idx = (l[:, 0] == u).nonzero()
            beta_u = beta[u] * beta_uv.index_select(0, ch_idx).prod(0)
            beta_u = beta_u / beta_u.sum()
            beta[u] = beta_u
    
    return beta, t_beta


def _downward(tree, A, Pi, prior, beta, t_beta):
    root = tree['levels'][0][0, 0]
    eps = torch.zeros(beta.size())
    t_eps = torch.zeros(beta.size(0), A.size(0), A.size(1))

    eps[root] = beta[root]
    for l in tree['levels']:
        # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
        eps_pa = eps.index_select(0, l[:, 0]).unsqueeze(1)
        pos_ch = tree['pos'].index_select(0, l[:, 1])
        A_ch = A.index_select(2, pos_ch).permute(2, 0, 1)
        beta_ch = beta.index_select(0, l[:, 1]).unsqueeze(2)
        numerator = beta_ch * A_ch * eps_pa

        prior_ch = prior.index_select(0, l[:, 1]).unsqueeze(2)
        t_beta_pa = t_beta.index_select(0, l[:, 0]).unsqueeze(1)
        denominator = prior_ch @ t_beta_pa

        t_eps_ch = numerator / denominator
        t_eps = t_eps.scatter_(dim=0, index=l[:, 1], src=t_eps_ch)

        # Computing eps_u(i)
        num_eps_ch = t_eps_ch.sum(2)
        den_eps_ch = num_eps_ch.sum(1, keepdim=True)

        eps_ch = num_eps_ch / den_eps_ch
        eps = eps.scatter_(dim=0, index=l[:, 1], src=eps_ch)

    return eps, t_eps


def _log_likelihood(tree, A, B, Pi, eps, t_eps):
    root = tree['levels'][0][0, 0]
    no_root = torch.cat([l[:, 1] for l in tree['levels']])

    # Likelihood Pi
    Pi_lhood = eps[root] * Pi.log()

    # Likelihood A
    t_eps_no_root = t_eps.index_select(0, no_root)
    pos_no_root = tree['pos'].index_select(0, t_eps_no_root)
    A_no_root = A.index_select(2, pos_no_root).permute(2, 0, 1)
    A_lhood = t_eps_no_root * A_no_root.log()

    # Likelihood B
    b_nodes = B.index_select(1, tree['labels']).T
    B_lhood = eps * b_nodes.log()

    return Pi_lhood.sum() + A_lhood.sum() + B_lhood.sum()
