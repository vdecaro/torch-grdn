import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter

class TopDownHTMM(nn.Module):

    def __init__(self, n_gen, C, M, tree_dropout):
        super(TopDownHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=2.5))

        self.tree_dropout = tree_dropout
    

    def forward(self, x, trees, batch):
        sm_A, sm_B, sm_Pi = self._softmax_reparameterization(self.n_gen, self.A, self.B, self.Pi)

        prior = self._preliminary_downward(trees, sm_A, sm_Pi)
        log_likelihood, beta, t_beta = self._upward(x, trees, sm_A, sm_B, prior)
        if self.training:
            eps, t_eps = self._downward(trees, sm_A, sm_Pi, prior, beta, t_beta)
            self._compute_gradient(x, trees, batch, sm_A, sm_B, sm_Pi, eps, t_eps) 

        return log_likelihood


    def _softmax_reparameterization(self, n_gen, A, B, Pi):
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(n_gen):
            sm_A.append(F.softmax(A[:, :, i], dim=0))
            sm_B.append(F.softmax(B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(Pi[:, i], dim=0))

        return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


    def _preliminary_downward(self, tree, A, Pi):
        prior = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)

        prior[tree['roots']] = Pi

        for l in tree['levels']:
            prior_pa = prior[l[0]].unsqueeze(1)
            prior_l = (A.unsqueeze(0) * prior_pa).sum(2)
            prior[l[1]] = prior_l
        
        return prior
            

    def _upward(self, x, tree, A, B, prior):
        beta = prior * B[:, x[tree['inv_map']]].permute(1, 0, 2)
        beta = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        t_beta = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        log_likelihood = torch.zeros((tree['dim'], self.n_gen), device=self.A.device)

        beta_leaves_unnorm = beta[tree['leaves']]
        nu = beta_leaves_unnorm.sum(dim=1)

        beta[tree['leaves']] = beta_leaves_unnorm / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        for l in reversed(tree['levels']):
            # Computing beta_uv children = (A_ch @ beta_ch) / prior_pa
            beta_ch = beta[l[1]].unsqueeze(2)
            prior_l = prior[l[1]].unsqueeze(2)
            beta_uv = (A.unsqueeze(0) * beta_ch / prior_l).sum(1)
            t_beta[l[1]] = beta_uv
            
            # Computing beta on level = (\prod_ch beta_uv_ch) * prior_u * 
            pa_idx = l[0].unique(sorted=False)
            prev_beta = beta[pa_idx]
            beta = scatter(src=beta_uv, index=l[0], dim=0, out=beta, reduce='mul')
            beta_l_unnorm = prev_beta * beta[pa_idx]
            nu = beta_l_unnorm.sum(1)

            beta[pa_idx] = beta_l_unnorm / nu.unsqueeze(1)
            log_likelihood[pa_idx] = nu.log()

        return scatter(log_likelihood, tree['trees_ind'], dim=0), beta, t_beta


    def _downward(self, tree, A, Pi, prior, beta, t_beta):
        eps = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        t_eps = torch.zeros((tree['dim'], self.C, self.C, self.n_gen), device=self.A.device)
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
            eps_ch_unnorm = t_eps_ch.sum(2)
            eps[l[1]] = eps_ch_unnorm / eps_ch_unnorm.sum(1, keepdim=True)

        return eps.detach(), t_eps.detach()


    def _compute_gradient(self, x, tree, batch, A, B, Pi, eps, t_eps):
        internal = torch.cat([l[1].unique(sorted=False) for l in tree['levels']])

        # Likelihood B
        B_nodes = B[:, x[tree['inv_map']]].permute(1, 0, 2)
        exp_likelihood = (eps * B_nodes.log()).sum(1)

        # Likelihood A
        exp_likelihood[internal] += (t_eps[internal] * A.log().unsqueeze(0)).sum([1, 2])

        # Likelihood Pi
        exp_likelihood[tree['roots']] += (eps[tree['roots']] * Pi.log()).sum(1)
        
        bitmask = torch.rand_like(exp_likelihood, device=self.A.device) > self.tree_dropout
        exp_likelihood *= bitmask
        '''
        exp_likelihood = scatter(src=exp_likelihood, index=tree['trees_ind'], dim=0, reduce='sum')
        exp_likelihood = scatter(src=exp_likelihood, index=batch, dim=0, reduce='sum')
        neg_exp_likelihood = -exp_likelihood.mean(0).sum()
        '''
        neg_exp_likelihood = - exp_likelihood.sum() / (batch.max() + 1)
        neg_exp_likelihood.backward()
