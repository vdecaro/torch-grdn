import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter

class TopDownHTMM(nn.Module):

    def __init__(self, n_gen, C, M):
        super(TopDownHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, n_gen))))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen))))
    

    def forward(self, x, trees):
        sm_A, sm_B, sm_Pi = self._softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi)

        prior = self._preliminary_downward(trees, self.n_gen, sm_A, sm_Pi, self.C)
        beta, t_beta = self._upward(x, trees, self.n_gen, sm_A, sm_B, prior, self.C)
        eps, t_eps = self._downward(trees, self.n_gen, sm_A, sm_Pi, prior, beta, t_beta, self.C)
        log_likelihood = self._log_likelihood(x, trees, sm_A, sm_B, sm_Pi, eps, t_eps) 

        return - log_likelihood


    def _softmax_reparameterization(self, n_gen, A, B, Pi):
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(n_gen):
            sm_A.append(F.softmax(A[:, :, i], dim=0))
            sm_B.append(F.softmax(B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(Pi[:, i], dim=0))

        return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


    def _preliminary_downward(self, tree, n_gen, A, Pi, C):
        prior = torch.zeros((tree['dim'], C, n_gen), device=self.A.device)

        prior[tree['roots']] = Pi

        for l in tree['levels']:
            prior_pa = prior[l[0]].unsqueeze(1)
            prior_l = (A.unsqueeze(0) * prior_pa).sum(2)
            prior[l[1]] = prior_l
        
        return prior
            

    def _upward(self, x, tree, n_gen, A, B, prior, C):
        beta = torch.zeros((tree['dim'], C, n_gen), device=self.A.device)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=self.A.device)
        
        beta_leaves = prior[tree['leaves']] * B[:, x[tree['inv_map'][tree['leaves']]]].permute(1, 0, 2)
        beta_leaves = beta_leaves / beta_leaves.sum(dim=1, keepdim=True)
        beta[tree['leaves']] = beta_leaves

        for l in reversed(tree['levels']):
            # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
            beta_ch = beta[l[1]].unsqueeze(2)
            prior_l = prior[l[1]].unsqueeze(2)
            beta_uv = (A.unsqueeze(0) * beta_ch / prior_l).sum(1)
            t_beta[l[1]] = beta_uv
            
            # Computing beta on level = (\prod_ch beta_uv_ch) * prior_u * 
            beta_u = []
            u_idx = l[0].unique(sorted=False)
            for u in u_idx:
                ch_idx = (l[0] == u).nonzero().squeeze(1)
                beta_u.append(beta_uv[ch_idx].prod(0))

            beta_u = prior[u_idx] * B[:, x[tree['inv_map'][u_idx]]].permute(1, 0, 2) * torch.stack(beta_u)
            beta[u_idx] = beta_u / beta_u.sum(1, keepdim=True)
        
        return beta, t_beta


    def _downward(self, tree, n_gen, A, Pi, prior, beta, t_beta, C):
        eps = torch.zeros((tree['dim'], C, n_gen), device=self.A.device)
        t_eps = torch.zeros((tree['dim'], C, C, n_gen), device=self.A.device)
        eps[tree['roots']] = beta[tree['roots']]

        for l in tree['levels']:
            # Computing eps_{u, pa(u)}(i, j) = (beta_u(i) / (prior_u(i)) * \sum_{j} (eps_{pa(u)}(j)*A_ij t_beta_{pa(u), u}(j)))
            eps_pa = eps[l[0]].unsqueeze(1)
            t_beta_ch = t_beta[l[1]].unsqueeze(1)
            pa_factor = (eps_pa * A.unsqueeze(0)) / t_beta_ch 

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


    def _log_likelihood(self, x, tree, A, B, Pi, eps, t_eps):
        internal = torch.cat([l[0].unique(sorted=False) for l in tree['levels']])
        no_root = torch.cat([l[1].unique(sorted=False) for l in tree['levels']])
        all_nodes = torch.cat([internal, tree['leaves']])

        lhood_size = eps.size(0), eps.size(-1)
        likelihood = torch.zeros(lhood_size, device=self.A.device)

        # Likelihood Pi
        likelihood[tree['roots']] += (eps[tree['roots']] * Pi.log()).sum(1)
        
        # Likelihood A
        likelihood[no_root] += (t_eps[no_root] * A.log()).sum([1, 2])
        
        # Likelihood B
        all_nodes_mappings = tree['inv_map'][all_nodes]
        B_nodes = B[:, x[all_nodes_mappings]].permute(1, 0, 2)
        likelihood[all_nodes] += (eps[all_nodes] * B_nodes.log()).sum(1)
        
        return scatter(likelihood, tree['trees_ind'], dim=0)

