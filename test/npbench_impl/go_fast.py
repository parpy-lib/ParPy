# https://numba.readthedocs.io/en/stable/user/5minguide.html

import prickle
import torch

@prickle.jit
def prickle_kernel(a, tmp, out, N):
    prickle.label('i')
    for i in range(N):
        tmp[0] += prickle.tanh(a[i,i])
    prickle.label('ix')
    prickle.label('j')
    out[:,:] = a[:,:] + tmp[0]

def go_fast(a, opts):
    N, N = a.shape
    tmp = torch.tensor([0.0], dtype=a.dtype)
    out = torch.empty_like(a)
    prickle_kernel(a, tmp, out, N, opts=opts)
    return out
