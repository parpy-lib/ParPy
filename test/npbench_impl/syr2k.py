import prickle
import torch

@prickle.jit
def kernel_wrap(alpha, beta, C, A, B, N, M):
    prickle.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta
            prickle.label('k')
            for k in range(M):
                C[i,j] += A[j,k] * alpha * B[i,k] + B[j,k] * alpha * A[i,k]

def syr2k(alpha, beta, C, A, B, opts):
    N, M = A.shape
    kernel_wrap(alpha, beta, C, A, B, N, M, opts=opts)
