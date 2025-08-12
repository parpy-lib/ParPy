import prickle
import torch

@prickle.jit
def softmax_wrap(x, out, N, H, SM):
    prickle.label('i')
    for i in range(N):
        prickle.label('j')
        for j in range(H):
            prickle.label('k')
            for k in range(SM):
                prickle.label('l')
                m = prickle.max(x[i,j,k,:])

                prickle.label('l')
                out[i,j,k,:] = prickle.exp(x[i,j,k,:]-m)

                prickle.label('l')
                s = prickle.sum(out[i,j,k,:])

                prickle.label('l')
                out[i,j,k,:] /= s

# Numerically-stable version of softmax
def softmax(x, opts):
    out = torch.zeros_like(x)
    N, H, SM, SM = x.shape
    softmax_wrap(x, out, N, H, SM, opts=opts)
    return out
