import parpy
import torch

@parpy.jit
def parpy_kernel(alpha, beta, C, A, B, M, N):
    parpy.label('M')
    parpy.label('N')
    C[:,:] *= beta

    for i in range(M):
        parpy.label('N')
        for j in range(N):
            parpy.label('M')
            for k in range(i):
                C[k,j] += alpha * B[i,j] * A[i,k]

        parpy.label('N')
        for j in range(N):
            dot_sum = 0.0
            parpy.label('i_red')
            for k in range(i):
                dot_sum += B[k,j] * A[i,k]
            C[i,j] += alpha * B[i,j] * A[i,i] + alpha * dot_sum

def symm(alpha, beta, C, A, B, opts, compile_only=False):
    M, N = C.shape
    if compile_only:
        args = [alpha, beta, C, A, B, M, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(alpha, beta, C, A, B, M, N, opts=opts)
