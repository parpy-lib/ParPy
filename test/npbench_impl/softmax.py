import parpy
from parpy.operators import exp, max, sum
import torch

@parpy.jit
def softmax_wrap(x, out, N, H, SM):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        for j in range(H):
            parpy.label('k')
            for k in range(SM):
                parpy.label('l')
                m = max(x[i,j,k,:])

                parpy.label('l')
                out[i,j,k,:] = exp(x[i,j,k,:]-m)

                parpy.label('l')
                s = sum(out[i,j,k,:])

                parpy.label('l')
                out[i,j,k,:] /= s

# Numerically-stable version of softmax
def softmax(x, opts, compile_only=False):
    out = torch.zeros_like(x)
    N, H, SM, SM = x.shape
    if compile_only:
        args = [x, out, N, H, SM]
        return parpy.print_compiled(softmax_wrap, args, opts)
    softmax_wrap(x, out, N, H, SM, opts=opts)
    return out
