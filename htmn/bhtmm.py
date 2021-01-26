import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.scatter import scatter

class BottomUpHTMM(nn.Module):

    def __init__(self, n_gen, C, L, M):
        super(BottomUpHTMM, self).__init__()
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.A = nn.Parameter(nn.init.uniform_(torch.empty((C, C, L, n_gen))))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty((C, M, n_gen))))
        self.Pi = nn.Parameter(nn.init.uniform_(torch.empty((C, L, n_gen))))
        self.SP = nn.Parameter(nn.init.uniform_(torch.empty((L, n_gen))))
    
    def forward(self, tree):

        return ReversedUpwardDownward.apply(tree, self.A, self.B, self.Pi, self.SP)


class ReversedUpwardDownward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tree, lambda_A, lambda_B, lambda_Pi, lambda_SP):
        # Softmax reparameterization
        sm_A, sm_B, sm_Pi, sm_SP = [], [], [], []
        for i in range(lambda_A.size(-1)):
            sm_A.append(torch.cat([F.softmax(lambda_A[:, :, j, i], dim=0).unsqueeze(2) for j in range(lambda_A.size(2))], dim=2))
            sm_B.append(F.softmax(lambda_B[:, :, i], dim=1))
            sm_Pi.append(F.softmax(lambda_Pi[:, :, i], dim=0))
            sm_SP.append(F.softmax(lambda_SP[:, i], dim=0))

        A, B, Pi, SP = torch.stack(sm_A, dim=-1), torch.stack(sm_B, dim=-1), torch.stack(sm_Pi, dim=-1), torch.stack(sm_SP, dim=-1)

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device
        
        # Upward recursion: init
        beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        t_beta = torch.zeros((tree['dim'], C, n_gen), device=device)
        log_likelihood = torch.zeros((tree['dim'], n_gen), device=device)

        # Upward recursion: base case
        pos_leaves = tree['pos'][tree['leaves']]
        Pi_leaves = Pi[:, pos_leaves]
        B_leaves = B[:, tree['x'][tree['leaves']]]
        beta_leaves = (Pi_leaves * B_leaves).permute(1, 0, 2)
        nu = beta_leaves.sum(dim=1)
        
        beta[tree['leaves']] = beta_leaves / nu.unsqueeze(1)
        log_likelihood[tree['leaves']] = nu.log()

        # Upward recursion
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

        ctx.saved_input = tree
        ctx.save_for_backward(beta, t_beta, A, B, Pi, SP)

        return scatter(log_likelihood, tree['batch'], dim=0)

    @staticmethod
    def backward(ctx, log_likelihood):
        # Recovering saved tensors from forward
        tree = ctx.saved_input
        beta, t_beta, A, B, Pi, SP = ctx.saved_tensors

        # Getting model info
        C, n_gen, device = A.size(0), A.size(-1), A.device

        # Creating parameter gradient tensors
        A_grad, B_grad, Pi_grad, SP_grad = torch.zeros_like(A), torch.zeros_like(B), torch.zeros_like(Pi), torch.zeros_like(SP)

        eps = torch.zeros((tree['dim'], C, n_gen), device=device)

        roots = tree['levels'][0][0].unique(sorted=False)
        eps[roots] = beta[roots]
        for l in tree['levels']:
            # Computing eps_{u, ch(u)}(i, j)
            pos_ch = tree['pos'][l[1]]
            SP_ch = SP[pos_ch]
            A_ch = A[:, :, pos_ch].permute(2, 0, 1, 3)

            trans_ch = SP_ch.unsqueeze(1).unsqueeze(1) * A_ch
            eps_pa = eps[l[0]].unsqueeze(2)
            beta_ch = beta[l[1]].unsqueeze(1)
            t_beta_pa = t_beta[l[0]].unsqueeze(2)

            # Computing eps_{ch(u)}
            eps_joint = (eps_pa * trans_ch * beta_ch) / t_beta_pa
            eps_ch = eps_joint.sum(1)
            eps[l[1]] = eps_ch

            # Accumulating gradient in grad_A and grad_SP
            SP_grad = scatter(eps_ch.sum(1) - SP_ch, 
                              index=pos_ch, 
                              dim=0, 
                              out=SP_grad)
            A_grad = scatter((eps_joint - A_ch*eps_ch.unsqueeze(1)).permute(1, 2, 0, 3), 
                             index=pos_ch, 
                             dim=2, 
                             out=A_grad)
        
        pos_leaves = tree['pos'][tree['leaves']]
        eps_leaves = eps[tree['leaves']]
        pi_leaves = Pi[:, pos_leaves]
        Pi_grad = scatter(eps_leaves.permute(1, 0, 2) - pi_leaves,
                          index=pos_leaves,
                          dim=1,
                          out=Pi_grad)
        
        B_grad = scatter(eps.permute(1, 0, 2) - eps.permute(1, 0, 2) * B[:, tree['x']],
                         index=tree['x'],
                         dim=1,
                         out=B_grad)

        return None, A_grad, B_grad, Pi_grad, SP_grad
