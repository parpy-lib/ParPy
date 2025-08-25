import parpy
import torch


@parpy.jit
def ludcmp_kernel(A, b, x, y, N):
    with parpy.gpu:
        for i in range(N):
            for j in range(i):
                s = 0.0
                parpy.label('k')
                for k in range(j):
                    s += A[i,k] * A[k,j]
                A[i, j] -= s
                A[i, j] /= A[j, j]
            for j in range(i, N):
                s = 0.0
                parpy.label('k')
                for k in range(i):
                    s += A[i,k] * A[k,j]
                A[i, j] -= s
        for i in range(N):
            s = 0.0
            parpy.label('k')
            for k in range(i):
                s += A[i,k] * y[k]
            y[i] = b[i] - s
        for i in range(N-1, -1, -1):
            s = 0.0
            parpy.label('k')
            for k in range(i+1, N):
                s += A[i,k] * x[k]
            x[i] = (y[i] - s) / A[i,i]

def ludcmp(A, b, opts, compile_only=False):
    N, N = A.shape
    x = torch.zeros_like(b)
    y = torch.zeros_like(b)
    if compile_only:
        return parpy.print_compiled(ludcmp_kernel, [A, b, x, y, N], opts)
    ludcmp_kernel(A, b, x, y, N, opts=opts)
    return x, y
