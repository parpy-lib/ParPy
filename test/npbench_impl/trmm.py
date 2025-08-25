import parpy
import torch

@parpy.jit
def trmm_kernel(alpha, A, B, M, N):
    for i in range(M):
        parpy.label('j')
        for j in range(N):
            parpy.label('k')
            for k in range(i+1, M):
                B[i,j] += A[k,i] * B[k,j]
            B[i,j] = B[i,j] * alpha

def trmm(alpha, A, B, opts, compile_only=False):
    M, N = B.shape
    if compile_only:
        args = [alpha, A, B, M, N]
        return parpy.print_compiled(trmm_kernel, args, opts)
    trmm_kernel(alpha, A, B, M, N, opts=opts)
