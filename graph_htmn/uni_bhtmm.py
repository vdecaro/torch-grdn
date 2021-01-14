import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter


class UniformBottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, M, tree_dropout):
        super(UniformBottomUpHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.A = nn.Parameter(nn.init.normal_(torch.empty((C, C, n_gen)), std=2.5))
        self.B = nn.Parameter(nn.init.normal_(torch.empty((C, M, n_gen)), std=2.5))
        self.Pi = nn.Parameter(nn.init.normal_(torch.empty((C, n_gen)), std=2.5))
        
        self.tree_dropout = tree_dropout

    def forward(self, x, trees, batch):
        sm_A, sm_B, sm_Pi = self._softmax_reparameterization(self.A, self.B, self.Pi)

        log_likelihood, beta, t_beta = self._reversed_upward(x, trees, sm_A, sm_B, sm_Pi)
        if self.training:
            eps, t_eps = self._reversed_downward(trees, sm_A, sm_Pi, beta, t_beta)
            self._compute_gradient(x, trees, batch, sm_A, sm_B, sm_Pi, eps, t_eps)

        return log_likelihood


    def _softmax_reparameterization(self, A, B, Pi):
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(self.n_gen):
            sm_A.append(F.softmax(A[:, :, i], dim=0))
            sm_B.append(F.softmax(B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(Pi[:, i], dim=0))

        return torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)


    def _reversed_upward(self, x, tree, A, B, Pi):
        beta = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        t_beta = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        log_likelihood = torch.zeros((tree['dim'], self.n_gen), device=self.A.device)

        Pi_leaves = Pi.unsqueeze(0)
        leaves_idx = tree['inv_map'][tree['leaves']]
        B_leaves = B[:, x[leaves_idx]].permute(1, 0, 2)
        beta_leaves = Pi_leaves * B_leaves
        nu = beta_leaves.sum(dim=1)

        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        for l in reversed(tree['levels']):
            # Computing unnormalized beta_uv children = A_ch @ beta_ch
            beta_ch = beta[l[1]]
            t_beta_ch = (A.unsqueeze(0) * beta_ch.unsqueeze(1)).sum(2)
            t_beta = scatter(src=t_beta_ch, index=l[0], dim=0, out=t_beta, reduce='mean')

            u_idx = l[0].unique(sorted=False)
            B_l = B[:, x[tree['inv_map'][u_idx]]].permute(1, 0, 2)
            beta_l = B_l * t_beta[u_idx]
            nu = beta_l.sum(dim=1)

            beta[u_idx] = beta_l / nu.unsqueeze(1)
            log_likelihood[u_idx] = nu.log()

        return scatter(log_likelihood, tree['trees_ind'], dim=0), beta, t_beta

    def _reversed_downward(self, tree, A, Pi, beta, t_beta):
        eps = torch.zeros((tree['dim'], self.C, self.n_gen), device=self.A.device)
        t_eps = torch.zeros((tree['dim'], self.C, self.C, self.n_gen), device=self.A.device)

        eps[tree['roots']] = beta[tree['roots']]
        for l in tree['levels']:
            # Computing eps_{u, pa(u)}(i, j) = (eps_{pa(u)}(j)* A_ij * beta_u(i)) / (prior_u(i) * t_beta_{pa(u), u}(j))
            t_beta_pa = t_beta[l[0]].unsqueeze(2)
            eps_pa = eps[l[0]].unsqueeze(2)
            beta_ch = beta[l[1]].unsqueeze(1)
            eps_joint = (eps_pa * A.unsqueeze(0) * beta_ch) / t_beta_pa
            t_eps = scatter(src=eps_joint, index=l[0], dim=0, out=t_eps, reduce='mean')

            eps[l[1]] = eps_joint.sum(1)

        return eps.detach(), t_eps.detach()

    def _compute_gradient(self, x, tree, batch, A, B, Pi, eps, t_eps):
        internal = torch.cat([l[0].unique(sorted=False) for l in tree['levels']])
        
        # Likelihood B
        B_nodes = B[:, x[tree['inv_map']]].permute(1, 0, 2)
        exp_likelihood = (eps * B_nodes.log()).sum(1)

        # Likelihood A
        exp_likelihood[internal] += (t_eps[internal] * A.log().unsqueeze(0)).sum([1, 2])

        # Likelihood Pi
        exp_likelihood[tree['leaves']] += (eps[tree['leaves']] * Pi.unsqueeze(0).log()).sum(1)
        
        exp_likelihood = scatter(src=exp_likelihood, index=tree['trees_ind'], dim=0, reduce='sum')
        bitmask = torch.rand_like(exp_likelihood, device=self.A.device) > self.tree_dropout
        exp_likelihood *= bitmask
        exp_likelihood = scatter(src=exp_likelihood, index=batch, dim=0, reduce='mean')
        neg_exp_likelihood = - exp_likelihood.sum() / (batch.max() + 1)
        neg_exp_likelihood.backward()
