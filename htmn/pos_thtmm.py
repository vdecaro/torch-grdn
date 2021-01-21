import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.scatter import scatter

class PositionalTopDownHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M):
        super(PositionalTopDownHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.uniform_(torch.empty((C, C, L, n_gen))))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, n_gen))))

    def forward(self, tree):
        log_likelihood = UpwardDownward.apply(tree, self.A, self.B, self.Pi)

        return log_likelihood


class UpwardDownward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tree, lambda_A, lambda_B, lambda_Pi):

        # Softmax Reparameterization
        sm_A, sm_B, sm_Pi = [], [], []
        for i in range(lambda_A.size(-1)):
            sm_A.append(torch.cat([F.softmax(lambda_A[:, :, j, i], dim=0).unsqueeze(2) for j in range(lambda_A.size(2))], dim=2))
            sm_B.append(F.softmax(lambda_B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(lambda_Pi[:, i], dim=0))

        A, B, Pi = torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1)

         # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Preliminary Downward Recursion: init
        roots = tree['levels'][0][0].unique(sorted=False)
        prior = torch.zeros((tree['dim'], C, n_gen), device=device)

        # Preliminary Downward Recursion: base case
        prior[roots] = Pi
        
        # Preliminary Downward Recursion
        for l in tree['levels']:
            pos_ch = tree['pos'][l[1]]
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            prior_pa = prior[l[0]].unsqueeze(1)
            prior_l = (A_ch * prior_pa).sum(2)
            prior[l[1]] = prior_l
        
        # Upward Recursion: init
        beta = prior * B[:, tree['x']].permute(1, 0, 2)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        # Upward Recursion: base case
        beta_leaves = beta[tree['leaves']]
        nu = beta_leaves.sum(1)

        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        # Upward recursion
        for l in reversed(tree['levels']):
            # Computing beta_uv children
            pos_ch = tree['pos'][l[1]]
            beta_ch = beta[l[1]].unsqueeze(2)
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            prior_l = prior[l[1]].unsqueeze(2)
            beta_uv = (A_ch * beta_ch / prior_l).sum(1)
            t_beta[l[1]] = beta_uv
            
            # Computing beta on level
            pa_idx = l[0].unique(sorted=False)
            prev_beta = beta[pa_idx]
            beta = scatter(src=beta_uv, index=l[0], dim=0, out=beta, reduce='mul')
            beta_l_unnorm = prev_beta * beta[pa_idx]
            nu = beta_l_unnorm.sum(1)

            beta[pa_idx] = beta_l_unnorm / nu.unsqueeze(1)
            log_likelihood[pa_idx] = nu.log()

        ctx.saved_input = tree
        ctx.save_for_backward(prior, beta, t_beta, A, B, Pi)

        return scatter(log_likelihood, tree['batch'], dim=0)

    @staticmethod
    def backward(ctx, log_likelihood):
        # Recovering saved tensors from forward
        tree = ctx.saved_input
        prior, beta, t_beta, A, B, Pi = ctx.saved_tensors

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Creating parameter gradient tensors
        A_grad, B_grad = torch.zeros_like(A), torch.zeros_like(B)

        eps = torch.zeros((tree['dim'], C, n_gen), device=device)

        roots = tree['levels'][0][0].unique(sorted=False)
        eps_roots = beta[roots]
        eps[roots] = eps_roots
        for l in tree['levels']:
            # Computing eps_{u, pa(u)}(i, j)
            eps_pa = eps[l[0]].unsqueeze(1)
            pos_ch = tree['pos'][l[1]]
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)
            t_beta_ch = t_beta[l[1]].unsqueeze(1)
            eps_trans_pa = eps_pa * A_ch
            pa_factor = eps_trans_pa / t_beta_ch 

            beta_ch = beta[l[1]].unsqueeze(2)
            prior_ch = prior[l[1]].unsqueeze(2)
            ch_factor = beta_ch / prior_ch

            eps_joint = ch_factor * pa_factor

            # Computing eps_u(i)
            num_eps_ch = eps_joint.sum(2)
            den_eps_ch = num_eps_ch.sum(1, keepdim=True)

            eps_ch = num_eps_ch / den_eps_ch
            eps[l[1]] = eps_ch

            # Accumulating gradient in grad_A and grad_SP
            A_grad = scatter((eps_joint - eps_trans_pa).permute(1, 2, 0, 3), 
                             index=pos_ch, 
                             dim=2, 
                             out=A_grad)

        B_grad = scatter(eps.permute(1, 0, 2),
                         index=tree['x'],
                         dim=1,
                         out=B_grad)
        B_grad -= eps.sum(0).unsqueeze(1) * B
        Pi_grad = eps_roots.sum(0) - roots.size(0) * Pi

        return None, -A_grad, -B_grad, -Pi_grad