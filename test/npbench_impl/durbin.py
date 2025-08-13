import prickle
import torch

@prickle.jit
def prickle_kernel(r, y, temp, N):
    with prickle.gpu:
        alpha = -r[0]
        beta = 1.0
        y[0] = -r[0]
        for k in range(1, N):
            beta *= 1.0 - alpha * alpha
            t = 0.0
            prickle.label('k_red')
            for i in range(k):
                t += r[k-i-1] * y[i]
            alpha = -(r[k] + t) / beta
            prickle.label('k')
            for i in range(k):
                temp[i] = alpha * y[k-i-1]
            prickle.label('k')
            for i in range(k):
                y[i] += temp[i]
            y[k] = alpha

def durbin(r, opts, compile_only=False):
    y = torch.empty_like(r)
    temp = torch.empty_like(y)
    N, = r.shape
    if compile_only:
        args = [r, y, temp, N]
        return prickle.print_compiled(prickle_kernel, args, opts)
    prickle_kernel(r, y, temp, N, opts=opts)
    return y
