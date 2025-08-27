import parpy
from parpy.operators import sqrt
import torch

@parpy.jit
def parpy_kernel(A, N):
    with parpy.gpu:
        A[0,0] = sqrt(A[0,0])
        for i in range(1, N):
            for j in range(i):
                s = 0.0
                parpy.label('k')
                for k in range(j):
                    s += A[i,k] * A[j,k]
                A[i,j] -= s
                A[i,j] = A[i,j] / A[j,j]
            s = 0.0
            for k in range(i):
                s += A[i,k] * A[i,k]
            A[i,i] -= s
            A[i,i] = sqrt(A[i,i])

def cholesky(A, opts, compile_only=False):
    N, _ = A.shape
    if compile_only:
        return parpy.print_compiled(parpy_kernel, [A, N], opts)
    parpy_kernel(A, N, opts=opts)
