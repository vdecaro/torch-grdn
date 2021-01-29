import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter.scatter import scatter


class UniformBottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, M):
        super(UniformBottomUpHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.M = M

        self.A = nn.Parameter(nn.init.uniform_(torch.empty((C, C, n_gen))))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, n_gen))))


    def forward(self, x, trees):

        return ReversedUpwardDownward.apply(x, trees, self.A, self.B, self.Pi)


class ReversedUpwardDownward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, tree, lambda_A, lambda_B, lambda_Pi):
        # Softmax Reparameterization
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(lambda_A.size(-1)):
            sm_A.append(F.softmax(lambda_A[:, :, i], dim=0))
            sm_B.append(F.softmax(lambda_B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(lambda_Pi[:, i], dim=0))
        
        A, B, Pi = torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Upward recursion: init
        beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        # Upward Recursion: base case
        Pi_leaves = Pi.unsqueeze(0)
        leaves_idx = tree['inv_map'][tree['leaves']]
        B_leaves = B[:, x[leaves_idx]].permute(1, 0, 2)
        beta_leaves = Pi_leaves * B_leaves
        nu = beta_leaves.sum(dim=1)

        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        # Upward Recursion
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

        ctx.saved_input = x, tree
        ctx.save_for_backward(beta, t_beta, A, B, Pi)

        return scatter(log_likelihood, tree['trees_ind'], dim=0)

    @staticmethod
    def backward(ctx, log_likelihood):
        # Recovering saved tensors from forward
        x, tree = ctx.saved_input
        beta, t_beta, A, B, Pi = ctx.saved_tensors

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Creating parameter gradient tensors
        A_grad, B_grad, Pi_grad = torch.zeros_like(A), torch.zeros_like(B), torch.zeros_like(Pi)

        # Downward recursion: init
        eps = torch.zeros((tree['dim'], C, n_gen), device=device)
        out_deg = torch.zeros(tree['dim'], device=device)

        # Downward recursion: base case
        eps[tree['roots']] = beta[tree['roots']]

        # Downward recursion
        for l in tree['levels']:
            # Computing eps_{u, ch_i(u)}(i, j)
            out_deg = scatter(torch.ones_like(l[1], dtype=out_deg.dtype, device=device), dim=0, index=l[0], out=out_deg)
            t_beta_pa = t_beta[l[0]].unsqueeze(2)
            eps_pa = eps[l[0]].unsqueeze(2)
            beta_ch = beta[l[1]].unsqueeze(1)

            eps_joint = (eps_pa * A.unsqueeze(0) * beta_ch) / (t_beta_pa * out_deg[l[0]].view(-1, 1, 1, 1))
            
            eps_ch = eps_joint.sum(1)
            eps[l[1]] = eps_joint.sum(1)
            A_grad += (eps_joint - A.unsqueeze(0)*eps_ch.unsqueeze(1)).sum(0)

        eps_leaves = eps[tree['leaves']]
        Pi_grad = eps_leaves.sum(0) - tree['leaves'].size(0)*Pi
        
        eps_nodes = eps.permute(1, 0, 2)
        x_trees = x[tree['inv_map']]
        B_grad = scatter(eps_nodes - eps_nodes * B[:, x_trees],
                         index=x_trees,
                         dim=1,
                         out=B_grad)

        return None, None, A_grad, B_grad, Pi_grad
