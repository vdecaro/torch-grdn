import torch
import torch.nn as nn
import torch.nn.functional as F


class BottomUpHTMM(nn.Module):

    def __init__(self, C, L, M):
        super(BottomUpHTMM, self).__init__()
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L))))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, L))))
        self.SP = nn.Parameter(nn.init.normal_(torch.empty((C, L))))
    

    def forward(self, tree):
        sm_A, sm_B, sm_Pi, sm_SP = _softmax_reparameterization(self.A, self.B, self.Pi, self.SP)

        beta, t_beta = _reversed_upward(tree, sm_A, sm_B, sm_Pi, sm_SP, self.C)
        eps, t_eps = _reversed_downward(tree, sm_A, sm_Pi, beta, t_beta)

        log_likelihood = _log_likelihood(tree, sm_A, sm_B, sm_Pi, eps, t_eps)

        return log_likelihood


def _softmax_reparameterization(A, B, Pi, SP):
    sm_A = torch.cat([F.softmax(A[:, :, i], dim=0).unsqueeze(2) for i in range(A.size(2))], dim=2)
    sm_B = F.softmax(B, dim=1)
    sm_Pi = F.softmax(Pi, dim=0)
    sm_SP = F.softmax(SP, dim=0)

    return sm_A, sm_B, sm_Pi, sm_SP


def _reversed_upward(tree, A, B, Pi, SP, C):
    beta = torch.zeros((tree['dim'], C))
    t_beta = torch.zeros((tree['dim'], C))  # It is the joint probability w.r.t. all the children

    pos_leaves = tree['pos'].index_select(0, tree['leaves'])
    Pi_leaves = Pi.index_select(1, pos_leaves)
    B_leaves = B.index_select(dim=1, index=tree['leaves'])
    beta_leaves = Pi_leaves * B_leaves
    beta_leaves = beta_leaves / beta_leaves.sum(dim=1, keepdim=True)
    beta = beta.scatter_(0, tree['leaves'], beta_leaves.T)

    for l in reversed(tree['levels']):
        # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
        pos_ch = tree['pos'].index_select(0, l[:, 1])
        SP_ch = SP.index_select(0, pos_ch).view(-1, 1, 1)
        A_ch = A.index_select(2, pos_ch).permute(2, 0, 1)
        beta_ch = beta.index_select(0, l[:, 1])

        t_beta_l_ch = (SP_ch * A_ch) @ beta_ch
        u_idx = l[:, 0].unique(sorted=False)
        t_beta_l = []
        for u in u_idx:
            ch_idx = (l[:, 0] == u).nonzero()
            t_beta_u = t_beta_l_ch.index_select(0, ch_idx).sum(0)
            t_beta_l.append(t_beta_u)
        t_beta_l = torch.cat(t_beta_l, dim=0)
        t_beta = t_beta.scatter_(0, u_idx, t_beta_l)

        B_l = B.index_select(1, tree['labels'].index_select(0, u_idx)).T
        beta_l = t_beta_l * B_l
        beta_l = beta_l / beta_l.sum(dim=1, keepdim=True)
        beta = beta.scatter_(0, u_idx, beta_l)
 
    return beta, t_beta


def _reversed_downward(tree, A, Pi, SP, beta, t_beta, C, L):
    eps = torch.zeros((tree['dim'], C))
    t_eps = torch.zeros((tree['dim'], C, C, L))

    root = tree['levels'][0][0, 0]
    eps[root] = beta[root]
    for l in tree['levels']:
        # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
        u_idx = l[:, 0].unique(sorted=False)
        for u in u_idx:
            eps_pa = eps[u].view(-1, 1, 1)
            ch_idx = (l[:, 0] == u).nonzero()
            n_children = ch_idx.size(0)
            A_ch = SP[:n_children].view(-1, 1, 1) * A[:, :, :n_children]
        
            beta_ch = beta.index_select(0, ch_idx).view(1, -1, n_children)
            numerator = beta_ch * A_ch * eps_pa

            t_beta_pa = t_beta[u].view(-1, 1, 1)

            t_eps_u = numerator / t_beta_pa
            t_eps[u] = F.pad(t_eps_u, (0, L-n_children, 0, 0, 0, 0))

            eps_ch = t_eps_u.sum(0).T
            eps = eps.scatter_(dim=0, index=ch_idx, src=eps_ch)

    return eps, t_eps


def _log_likelihood(tree, A, B, Pi, SP, eps, t_eps):

    # Likelihood A
    A_lhood = t_eps * A.log().unsqueeze(0)

    # Likelihood B
    B_nodes = B.index_select(1, tree['labels']).T
    B_lhood = eps * B_nodes.log()

    # Likelihood Pi
    leaves_pos = tree['pos'].index_select(0, tree['leaves'])
    Pi_lhood = eps.index_select(0, tree['leaves']) * Pi.index_select(1, leaves_pos).log().T

    # Likelihood SP
    SP_lhood = t_eps * SP.log().view(1, 1, 1, -1)

    return A_lhood.sum() + B_lhood.sum() + Pi_lhood.sum() + SP_lhood.sum()
