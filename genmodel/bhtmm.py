import torch
import torch.nn as nn
import torch.nn.functional as F


class BottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M):
        super(BottomUpHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L, n_gen))))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, L, n_gen))))
        self.SP = nn.Parameter(nn.init.normal_(torch.empty((L, n_gen))))
    

    def forward(self, tree_batch):
        sm_A, sm_B, sm_Pi, sm_SP = _softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi, self.SP)

        log_likelihood = []
        for tree in tree_batch:
            beta, t_beta = _reversed_upward(tree, self.n_gen, sm_A, sm_B, sm_Pi, sm_SP, self.C)
            eps, t_eps = _reversed_downward(tree, self.n_gen, sm_A, sm_Pi, sm_SP, beta, t_beta, self.C, self.L)

            log_likelihood.append([-_log_likelihood(tree, sm_A, sm_B, sm_Pi, sm_SP, eps, t_eps)])   # Negative log likelihood

        return torch.stack(log_likelihood)


def _softmax_reparameterization(n_gen, A, B, Pi, SP):
    sm_A, sm_B, sm_Pi, sm_SP = [], [], [], []
    for i in range(n_gen):
        sm_A.append(torch.cat([F.softmax(A[:, :, j, i], dim=0).unsqueeze(2) for j in range(A.size(2))], dim=2))
        sm_B.append(F.softmax(B[:, :, i], dim=1))
        sm_Pi.append(F.softmax(Pi[:, :, i], dim=0))
        sm_SP.append(F.softmax(SP[:, i], dim=0))

    return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1), torch.stack(sm_SP, dim=-1)


def _reversed_upward(tree, n_gen, A, B, Pi, SP, C):
    dim = tree['labels'].size(0)
    beta = torch.zeros((dim, C, n_gen))
    t_beta = torch.zeros((dim, C, n_gen))  # It is the joint probability w.r.t. all the children

    pos_leaves = tree['pos'][tree['leaves']]
    Pi_leaves = Pi[:, pos_leaves]
    B_leaves = B[:, tree['leaves']]
    beta_leaves = Pi_leaves * B_leaves
    beta_leaves = (beta_leaves / beta_leaves.sum(dim=0, keepdim=True)).permute(1, 0, 2)
    
    beta[tree['leaves'], :] = beta_leaves

    for l in reversed(tree['levels']):
        # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
        pos_ch = tree['pos'][l[:, 1]]
        SP_ch = SP[pos_ch].unsqueeze(1).unsqueeze(2)
        A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
        beta_ch = beta[l[:, 1]].unsqueeze(1)
        
        t_beta_l_ch = (SP_ch * A_ch * beta_ch).sum(2)
        u_idx = l[:, 0].unique(sorted=False)
        for u in u_idx:
            t_beta_u = t_beta_l_ch[l[:, 0] == u]
            t_beta_u = t_beta_u.sum(0)
            t_beta[u] = t_beta_u

        B_l = B[:, tree['labels'][u_idx]].permute(1, 0, 2)
        beta_l = t_beta[u_idx] * B_l
        beta_l = beta_l / (beta_l.sum(dim=1, keepdim=True))
        beta[u_idx] = beta_l

    return beta, t_beta


def _reversed_downward(tree, n_gen, A, Pi, SP, beta, t_beta, C, L):
    dim = tree['labels'].size(0)
    eps = torch.zeros((dim, C, n_gen))
    t_eps = torch.zeros((dim, C, C, L, n_gen))

    root = tree['levels'][0][0, 0]
    eps[root] = beta[root]
    for l in tree['levels']:
        # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
        u_idx = l[:, 0].unique(sorted=False)
        for u in u_idx:
            eps_pa = eps[u].unsqueeze(1).unsqueeze(2)      # [C, 1, 1]
            ch_idx = (l[:, 0] == u).nonzero().squeeze()
            n_children = len(ch_idx)
            A_ch = SP[:n_children].unsqueeze(0).unsqueeze(0) * A[:, :, :n_children]   # [C, C, n_children]
        
            beta_ch = beta[ch_idx].permute(1, 0, 2).unsqueeze(0)     # [1, C, n_children]
            numerator = eps_pa * A_ch * beta_ch
            t_beta_pa = t_beta[u].unsqueeze(1).unsqueeze(2)

            t_eps_u = numerator / t_beta_pa
            t_eps[u] = F.pad(t_eps_u, (0, L-n_children, 0, 0, 0, 0))

            eps_ch = t_eps_u.sum(0).permute(1, 0, 2)
            eps[ch_idx] = eps_ch

    return eps, t_eps


def _log_likelihood(tree, A, B, Pi, SP, eps, t_eps):

    # Likelihood A
    A_lhood = t_eps * A.log().unsqueeze(0)

    # Likelihood B
    B_nodes = B[:, tree['labels']].permute(1, 0, 2)
    B_lhood = eps * B_nodes.log()

    # Likelihood Pi
    leaves_pos = tree['pos'][tree['leaves']]
    Pi_lhood = eps[tree['leaves']] * Pi[:, leaves_pos].log().permute(1, 0, 2)

    # Likelihood SP
    SP_lhood = t_eps * SP.log().unsqueeze(0).unsqueeze(1).unsqueeze(2)

    return A_lhood.sum([0, 1, 2]) + B_lhood.sum([0, 1]) + Pi_lhood.sum([0, 1]) + SP_lhood.sum(0)
