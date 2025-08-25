import parpy
import torch

@parpy.jit
def kernel_wrap(alpha, beta, C, A, B, N, M):
    parpy.label('i')
    for i in range(N):
        for j in range(i+1):
            C[i,j] *= beta
            parpy.label('k')
            for k in range(M):
                C[i,j] += A[j,k] * alpha * B[i,k] + B[j,k] * alpha * A[i,k]

def syr2k(alpha, beta, C, A, B, opts, compile_only=False):
    N, M = A.shape
    if compile_only:
        args = [alpha, beta, C, A, B, N, M]
        return parpy.print_compiled(kernel_wrap, args, opts)
    kernel_wrap(alpha, beta, C, A, B, N, M, opts=opts)
