import prickle
import torch

@prickle.jit
def syrk_kernel(alpha, beta, C, A, N, M):
    prickle.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta[0]
            prickle.label('k')
            for k in range(M):
                C[i,j] += alpha[0] * A[i,k] * A[j,k]

def syrk(alpha, beta, C, A, opts):
    alpha = torch.tensor([alpha], dtype=A.dtype)
    beta = torch.tensor([beta], dtype=A.dtype)
    N, M = A.shape
    syrk_kernel(alpha, beta, C, A, N, M, opts=opts)
