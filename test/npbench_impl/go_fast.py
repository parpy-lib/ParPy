# https://numba.readthedocs.io/en/stable/user/5minguide.html

import parpy
from parpy.operators import tanh
import torch

@parpy.jit
def parpy_kernel(a, tmp, out, N):
    parpy.label('i')
    for i in range(N):
        tmp[0] += tanh(a[i,i])
    parpy.label('ix')
    parpy.label('j')
    out[:,:] = a[:,:] + tmp[0]

def go_fast(a, opts, compile_only=False):
    N, N = a.shape
    tmp = torch.tensor([0.0], dtype=a.dtype)
    out = torch.empty_like(a)
    if compile_only:
        args = [a, tmp, out, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(a, tmp, out, N, opts=opts)
    return out
