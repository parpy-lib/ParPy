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

def syr2k(alpha, beta, C, A, B, opts, compile_only=False):
    N, M = A.shape
    if compile_only:
        args = [alpha, beta, C, A, B, N, M]
        return prickle.print_compiled(kernel_wrap, args, opts)
    kernel_wrap(alpha, beta, C, A, B, N, M, opts=opts)
