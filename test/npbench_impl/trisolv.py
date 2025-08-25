import parpy
import torch

@parpy.jit
def parpy_kernel(L, x, b, N):
    parpy.label('N')
    for i in range(N):
        t = 0.0
        parpy.label('reduce')
        for k in range(i):
            t += L[i,k] * x[k]
        x[i] = (b[i] - t) / L[i,i]

def trisolv(L, x, b, opts, compile_only=False):
    N = x.shape[0]
    if compile_only:
        args = [L, x, b, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(L, x, b, N, opts=opts)
