import prickle
import torch

@prickle.jit
def trmm_kernel(alpha, A, B, M, N):
    for i in range(M):
        prickle.label('j')
        for j in range(N):
            prickle.label('k')
            for k in range(i+1, M):
                B[i,j] += A[k,i] * B[k,j]
            B[i,j] = B[i,j] * alpha

def trmm(alpha, A, B, opts):
    M, N = B.shape
    trmm_kernel(alpha, A, B, M, N, opts=opts)
