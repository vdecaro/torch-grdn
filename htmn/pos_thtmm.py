import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.scatter import scatter

class PositionalTopDownHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M, device):
        super(PositionalTopDownHTMM, self).__init__()
        self.device = torch.device(device)
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, L, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=2.5))
        self.to(device=self.device)

    def forward(self, tree):
        sm_A, sm_B, sm_Pi = self._softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi)
        
        prior = self._preliminary_downward(tree, self.n_gen, sm_A, sm_Pi, self.C, self.device)
        log_likelihood, beta, t_beta = self._upward(tree, self.n_gen, sm_A, sm_B, prior, self.C, self.device)

        if self.training:
            eps, t_eps = self._downward(tree, self.n_gen, sm_A, sm_Pi, prior, beta, t_beta, self.C, self.device)
            self._compute_gradient(tree, sm_A, sm_B, sm_Pi, eps, t_eps)

        return log_likelihood


    def _softmax_reparameterization(self, n_gen, A, B, Pi):
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(n_gen):
            sm_A.append(torch.cat([F.softmax(A[:, :, j, i], dim=0).unsqueeze(2) for j in range(A.size(2))], dim=2))
            sm_B.append(F.softmax(B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(Pi[:, i], dim=0))

        return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


    def _preliminary_downward(self, tree, n_gen, A, Pi, C, device):
        roots = tree['levels'][0][0].unique(sorted=False)
        prior = torch.zeros((tree['dim'], C, n_gen), device=device)
        prior[roots] = Pi

        for l in tree['levels']:
            pos_ch = tree['pos'][l[1]]
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            prior_pa = prior[l[0]].unsqueeze(1)
            prior_l = (A_ch * prior_pa).sum(2)
            prior[l[1]] = prior_l
        
        return prior
            

    def _upward(self, tree, n_gen, A, B, prior, C, device):
        beta = prior * B[:, tree['x']].permute(1, 0, 2)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        beta_leaves = beta[tree['leaves']]
        nu = beta_leaves.sum(1)
        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        for l in reversed(tree['levels']):
            # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
            pos_ch = tree['pos'][l[1]]
            beta_ch = beta[l[1]].unsqueeze(2)
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            prior_l = prior[l[1]].unsqueeze(2)
            beta_uv = (A_ch * beta_ch / prior_l).sum(1)
            t_beta[l[1]] = beta_uv
            
            # Computing beta on level = (\prod_ch beta_uv_ch) * prior_u *
            beta_l = []
            pa_idx = l[0].unique(sorted=False)
            for u in pa_idx:
                ch_idx = (l[0] == u).nonzero().squeeze()
                beta_u = beta[u] * beta_uv[ch_idx].prod(0)
                beta_l.append(beta_u)

            beta_l = torch.stack(beta_l)
            nu = beta_l.sum(1)

            beta[pa_idx] = beta_l / nu.unsqueeze(1)
            log_likelihood[pa_idx] = nu.log()
        
        return scatter(log_likelihood, tree['batch'], dim=0), beta, t_beta


    def _downward(self, tree, n_gen, A, Pi, prior, beta, t_beta, C, device):
        eps = torch.zeros((tree['dim'], C, n_gen), device=device)
        t_eps = torch.zeros((tree['dim'], C, C, n_gen), device=device)

        roots = tree['levels'][0][0].unique(sorted=False)
        eps[roots] = beta[roots]
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

        return eps.detach(), t_eps.detach()


    def _compute_gradient(self, tree, A, B, Pi, eps, t_eps):
        roots = tree['levels'][0][0].unique(sorted=False)
        
        # Likelihood Pi
        exp_likelihood = (eps[roots] * Pi.log()).sum()
        
        # Likelihood A
        exp_likelihood += (t_eps * A[:, :, tree['pos']].permute(2, 0, 1, 3).log()).sum()
        
        # Likelihood B
        b_nodes = B[:, tree['x']].permute(1, 0, 2)
        exp_likelihood += (eps * b_nodes.log()).sum()

        mean_neg_exp_likelihood = - exp_likelihood / tree['batch'][-1]
        mean_neg_exp_likelihood.backward()