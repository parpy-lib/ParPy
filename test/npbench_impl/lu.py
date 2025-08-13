import prickle
import torch

@prickle.jit
def lu_prickle(A, N):
    with prickle.gpu:
        for i in range(N):
            for j in range(i):
                s = 0.0
                prickle.label('k')
                for k in range(j):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s
                A[i,j] /= A[j,j]
            for j in range(i, N):
                s = 0.0
                prickle.label('k')
                for k in range(i):
                    s += A[i,k] * A[k,j]
                A[i,j] -= s

def lu(A, opts, compile_only=False):
    N, N = A.shape
    if compile_only:
        return prickle.print_compiled(lu_prickle, [A, N], opts)
    lu_prickle(A, N, opts=opts)
