import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.scatter import scatter

class BottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M, device='cpu:0'):
        super(BottomUpHTMM, self).__init__()
        self.device = torch.device(device)
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, L, n_gen)), std=2.5))
        self.SP = nn.Parameter(nn.init.normal_(torch.empty((L, n_gen)), std=2.5))
        self.to(device=self.device)
    
    def forward(self, tree):
        sm_A, sm_B, sm_Pi, sm_SP = self._softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi, self.SP)

        log_likelihood, beta, t_beta = self._reversed_upward(tree, self.n_gen, sm_A, sm_B, sm_Pi, sm_SP, self.C, self.device)
        if self.training:
            eps, t_eps = self._reversed_downward(tree, self.n_gen, sm_A, sm_Pi, sm_SP, beta, t_beta, self.C, self.L, self.device)
            self._compute_gradient(tree, sm_A, sm_B, sm_Pi, sm_SP, eps, t_eps)

        return log_likelihood


    def _reversed_upward(self, tree, n_gen, A, B, Pi, SP, C, device):
        
        beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        pos_leaves = tree['pos'][tree['leaves']]
        Pi_leaves = Pi[:, pos_leaves]
        B_leaves = B[:, tree['x'][tree['leaves']]]
        beta_leaves = (Pi_leaves * B_leaves).permute(1, 0, 2)
        nu = beta_leaves.sum(dim=1)
        
        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        for l in reversed(tree['levels']):
            # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
            pos_ch = tree['pos'][l[1]]
            SP_ch = SP[pos_ch].unsqueeze(1).unsqueeze(2)
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            beta_ch = beta[l[1]].unsqueeze(1)
            
            t_beta_ch = (SP_ch * A_ch * beta_ch).sum(2)
            t_beta = scatter(src=t_beta_ch, index=l[0], dim=0, out=t_beta)

            u_idx = l[0].unique(sorted=False)
            B_l = B[:, tree['x'][u_idx]].permute(1, 0, 2)
            beta_l = t_beta[u_idx] * B_l
            nu = beta_l.sum(dim=1)

            beta[u_idx] = beta_l / nu.unsqueeze(1)
            log_likelihood[u_idx] = nu.log()

        return scatter(log_likelihood, tree['batch'], dim=0), beta, t_beta


    def _reversed_downward(self, tree, n_gen, A, Pi, SP, beta, t_beta, C, L, device):
        eps = torch.zeros((tree['dim'], C, n_gen), device=device)
        t_eps = []
        
        roots = tree['levels'][0][0].unique(sorted=False)
        eps[roots] = beta[roots]
        for l in tree['levels']:
            # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
            pos_ch = tree['pos'][l[1]]
            SP_ch = SP[pos_ch]
            A_ch = A[:, :, pos_ch]

            trans_ch = (SP_ch.unsqueeze(0).unsqueeze(0) * A_ch).permute(2, 0, 1, 3)
            eps_pa = eps[l[0]].unsqueeze(2)
            beta_ch = beta[l[1]].unsqueeze(1)
            t_beta_pa = t_beta[l[0]].unsqueeze(2)

            eps_joint = (eps_pa * trans_ch * beta_ch) / t_beta_pa
            t_eps.append(eps_joint.detach())
            eps[l[1]] = eps_joint.sum(1)

        return eps.detach(), t_eps


    def _softmax_reparameterization(self, n_gen, A, B, Pi, SP):
        sm_A, sm_B, sm_Pi, sm_SP = [], [], [], []
        for i in range(n_gen):
            sm_A.append(torch.cat([F.softmax(A[:, :, j, i], dim=0).unsqueeze(2) for j in range(A.size(2))], dim=2))
            sm_B.append(F.softmax(B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(Pi[:, :, i], dim=0))
            sm_SP.append(F.softmax(SP[:, i], dim=0))

        return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1), torch.stack(sm_SP, dim=-1)


    def _compute_gradient(self, tree, A, B, Pi, SP, eps, t_eps):
        # Likelihood B
        B_nodes = B[:, tree['x']].permute(1, 0, 2)
        exp_likelihood = (eps * B_nodes.log()).sum()

        # Likelihood Pi
        leaves_pos = tree['pos'][tree['leaves']]
        exp_likelihood += (eps[tree['leaves']] * Pi[:, leaves_pos].log().permute(1, 0, 2)).sum()

        for l, eps_joint in zip(tree['levels'], t_eps):
            # Likelihood A
            pos_ch = tree['pos'][l[1]]
            exp_likelihood += (eps_joint * A[:, :, pos_ch].permute(2, 0, 1, 3).log()).sum()
            exp_likelihood += (eps_joint.sum([1, 2]) * SP[pos_ch].log()).sum()
            
        mean_neg_exp_likelihood = - exp_likelihood / tree['batch'][-1]
        mean_neg_exp_likelihood.backward()