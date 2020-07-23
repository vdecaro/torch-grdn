import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalTopDownHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M):
        super(PositionalTopDownHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L, n_gen))))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen))))
    

    def forward(self, tree_batch):
        sm_A, sm_B, sm_Pi = _softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi)
        
        log_likelihood = []
        for tree in tree_batch:

            prior = _preliminary_downward(tree, self.n_gen, sm_A, sm_Pi, self.C)
            beta, t_beta = _upward(tree, self.n_gen, sm_A, sm_B, prior, self.C)
            eps, t_eps = _downward(tree, self.n_gen, sm_A, sm_Pi, prior, beta, t_beta, self.C)

            log_likelihood.append([-_log_likelihood(tree, sm_A, sm_B, sm_Pi, eps, t_eps)])  # Negative log likelihood

        return torch.tensor(log_likelihood)


def _softmax_reparameterization(n_gen, A, B, Pi):
    sm_A, sm_B, sm_Pi = [], [], []
    for i in range(n_gen):
        sm_A.append(torch.cat([F.softmax(A[:, :, j, i], dim=0).unsqueeze(2) for j in range(A.size(2))], dim=2))
        sm_B.append(F.softmax(B[:, :, i], dim=1))
        sm_Pi.append(F.softmax(Pi[:, i], dim=0))

    return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


def _preliminary_downward(tree, n_gen, A, Pi, C):
    root = tree['levels'][0][0, 0]
    prior = torch.zeros((tree['labels'].size(0), C, n_gen))
    prior[root] = Pi

    for l in tree['levels']:
        pos_ch = tree['pos'][l[1]]
        A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
        prior_pa = prior[l[0]].unsqueeze(1)
        prior_l = (A_ch * prior_pa).sum(2)
        prior[l[1]] = prior_l
    
    return prior
        

def _upward(tree, n_gen, A, B, prior, C):
    b_u = B[:, tree['labels']]
    beta = prior * b_u.permute(1, 0, 2)
    t_beta = torch.zeros(beta.size())

    beta_leaves = beta[tree['leaves']]
    beta_leaves = beta_leaves / beta_leaves.sum(dim=1, keepdim=True)
    beta[tree['leaves']] = beta_leaves

    for l in reversed(tree['levels']):
        # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
        pos_ch = tree['pos'][l[1]]
        beta_ch = beta[l[1]].unsqueeze(2)
        A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
        prior_l = prior[l[1]].unsqueeze(2)
        beta_uv = (A_ch * beta_ch / prior_l).sum(1)
        t_beta[l[1]] = beta_uv
        
        # Computing beta on level = (\prod_ch beta_uv_ch) * prior_u * 
        for u in l[0].unique(sorted=False):
            ch_idx = (l[0] == u).nonzero().squeeze()
            beta_u = beta[u] * beta_uv[ch_idx].prod(0)
            beta_u = beta_u / beta_u.sum()
            beta[u] = beta_u
    
    return beta, t_beta


def _downward(tree, n_gen, A, Pi, prior, beta, t_beta, C):
    root = tree['levels'][0][0, 0]
    dim = tree['labels'].size(0)
    eps = torch.zeros((dim, C, n_gen))
    t_eps = torch.zeros((dim, C, C, n_gen))

    eps[root] = beta[root]
    for l in tree['levels']:
        # Computing eps_{u, pa(u)}(i, j) = (beta_u(i) / (prior_u(i)) * \sum_{j} (eps_{pa(u)}(j)*A_ij t_beta_{pa(u), u}(j)))
        eps_pa = eps[l[0]].unsqueeze(1)
        pos_ch = tree['pos'][l[1]]
        A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
        t_beta_ch = t_beta[l[1]].unsqueeze(1)
        pa_factor = (eps_pa * A_ch) / t_beta_ch 

        beta_ch = beta[l[1]].unsqueeze(2)
        prior_ch = prior[l[1]].unsqueeze(2)
        ch_factor = beta_ch / prior_ch

        t_eps_ch = ch_factor * pa_factor
        t_eps[l[1]] = t_eps_ch

        # Computing eps_u(i)
        num_eps_ch = t_eps_ch.sum(2)
        den_eps_ch = num_eps_ch.sum(1, keepdim=True)

        eps_ch = num_eps_ch / den_eps_ch
        eps[l[1]] = eps_ch

    return eps, t_eps


def _log_likelihood(tree, A, B, Pi, eps, t_eps):
    root = tree['levels'][0][0, 0]
    no_root = torch.cat([l[1] for l in tree['levels']])
    
    # Likelihood Pi
    Pi_lhood = eps[root] * Pi.log()
    
    # Likelihood A
    t_eps_no_root = t_eps[no_root]
    pos_no_root = tree['pos'][no_root]
    A_no_root = A[:, :, pos_no_root].permute(2, 0, 1, 3)

    A_lhood = t_eps_no_root * A_no_root.log()
    
    # Likelihood B
    b_nodes = B[:, tree['labels']].permute(1, 0, 2)
    B_lhood = eps * b_nodes.log()
    
    return Pi_lhood.sum() + A_lhood.sum() + B_lhood.sum()
