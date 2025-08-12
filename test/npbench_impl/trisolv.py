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

def trisolv(L, x, b, opts):
    N = x.shape[0]
    prickle_kernel(L, x, b, N, opts=opts)
