import torch
from math import factorial as fact


def contrastive_matrix(N_GEN):
    contrastive_units = fact(N_GEN) // (2*fact(N_GEN-2))
    contrastive_matrix = torch.zeros((N_GEN, contrastive_units))

    p = 0
    s = 1
    for i in range(contrastive_units):
        contrastive_matrix[p, i] = 1
        contrastive_matrix[s, i] = -1
        if s == N_GEN - 1:
            p = p + 1
            s = p
        s = s + 1
    return contrastive_matrix
