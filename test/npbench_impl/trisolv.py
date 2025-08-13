import prickle
import torch

@prickle.jit
def prickle_kernel(L, x, b, N):
    prickle.label('N')
    for i in range(N):
        t = 0.0
        prickle.label('reduce')
        for k in range(i):
            t += L[i,k] * x[k]
        x[i] = (b[i] - t) / L[i,i]

def trisolv(L, x, b, opts, compile_only=False):
    N = x.shape[0]
    if compile_only:
        args = [L, x, b, N]
        return prickle.print_compiled(prickle_kernel, args, opts)
    prickle_kernel(L, x, b, N, opts=opts)
