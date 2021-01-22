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

        self.A = nn.Parameter(nn.init.uniform_(torch.empty((C, C, n_gen))))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, n_gen))))


    def forward(self, x, trees):

        return UpwardDownward.apply(x, trees, self.A, self.B, self.Pi)


class UpwardDownward(torch.autograd.Function):

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

        # Preliminary Downward recursion: init
        prior = torch.zeros((tree['dim'], C, n_gen), device=device)

        # Preliminary Downward recursion: base case
        prior[tree['roots']] = Pi

        # Preliminary Downward recursion
        for l in tree['levels']:
            prior_pa = prior[l[0]].unsqueeze(1)
            prior_l = (A.unsqueeze(0) * prior_pa).sum(2)
            prior[l[1]] = prior_l
        
        # Upward recursion: init
        beta = prior * B[:, x[tree['inv_map']]].permute(1, 0, 2)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        # Upward Recursion: base case
        beta_leaves_unnorm = beta[tree['leaves']]
        nu = beta_leaves_unnorm.sum(dim=1)

        beta[tree['leaves']] = beta_leaves_unnorm / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        # Upward Recursion
        for l in reversed(tree['levels']):
            # Computing beta_uv children
            beta_ch = beta[l[1]].unsqueeze(2)
            prior_l = prior[l[1]].unsqueeze(2)
            beta_uv = (A.unsqueeze(0) * beta_ch / prior_l).sum(1)
            t_beta[l[1]] = beta_uv
            
            # Computing beta on level
            pa_idx = l[0].unique(sorted=False)
            prev_beta = beta[pa_idx]
            beta = scatter(src=beta_uv, index=l[0], dim=0, out=beta, reduce='mul')
            beta_l_unnorm = prev_beta * beta[pa_idx]
            nu = beta_l_unnorm.sum(1)

            beta[pa_idx] = beta_l_unnorm / nu.unsqueeze(1)
            log_likelihood[pa_idx] = nu.log()

        ctx.saved_input = x, tree
        ctx.save_for_backward(prior, beta, t_beta, A, B, Pi)

        return scatter(log_likelihood, tree['trees_ind'], dim=0)

    @staticmethod
    def backward(ctx, log_likelihood):
        # Recovering saved tensors from forward
        x, tree = ctx.saved_input
        prior, beta, t_beta, A, B, Pi = ctx.saved_tensors

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Creating parameter gradient tensors
        A_grad, B_grad = torch.zeros_like(A), torch.zeros_like(B)

        # Downward recursion: init
        eps = torch.zeros((tree['dim'], C, n_gen), device=device)

        # Downward recursion: base case
        eps[tree['roots']] = beta[tree['roots']]

        for l in tree['levels']:
            # Computing eps_{u, pa(u)}(i, j) = (beta_u(i) / (prior_u(i)) * \sum_{j} (eps_{pa(u)}(j)*A_ij t_beta_{pa(u), u}(j)))
            eps_pa = eps[l[0]].unsqueeze(1)
            beta_ch = beta[l[1]].unsqueeze(2)
            eps_trans_pa = A.unsqueeze(0) * eps_pa
            
            t_beta_ch = t_beta[l[1]].unsqueeze(1)
            prior_ch = prior[l[1]].unsqueeze(2)

            eps_joint = (beta_ch * eps_trans_pa) / (prior_ch * t_beta_ch) 

            # Computing eps_u(i)
            eps_ch = eps_joint.sum(2)
            eps[l[1]] = eps_ch

            A_grad += (eps_joint - eps_trans_pa).sum(0)

        eps_roots = eps[tree['roots']]
        Pi_grad = eps_roots.sum(0) - tree['roots'].size(0)*Pi
        
        eps_nodes = eps.permute(1, 0, 2)
        B_grad = scatter(eps_nodes - eps_nodes * B[:, x[tree['inv_map']]],
                         index=x,
                         dim=1,
                         out=B_grad)

        return None, None, -A_grad, -B_grad, -Pi_grad
