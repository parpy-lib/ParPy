import parpy
import torch

@parpy.jit
def parpy_kernel(r, y, temp, N):
    with parpy.gpu:
        alpha = -r[0]
        beta = 1.0
        y[0] = -r[0]
        for k in range(1, N):
            beta *= 1.0 - alpha * alpha
            t = 0.0
            parpy.label('k_red')
            for i in range(k):
                t += r[k-i-1] * y[i]
            alpha = -(r[k] + t) / beta
            parpy.label('k')
            for i in range(k):
                temp[i] = alpha * y[k-i-1]
            parpy.label('k')
            for i in range(k):
                y[i] += temp[i]
            y[k] = alpha

def durbin(r, opts, compile_only=False):
    y = torch.empty_like(r)
    temp = torch.empty_like(y)
    N, = r.shape
    if compile_only:
        args = [r, y, temp, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(r, y, temp, N, opts=opts)
    return y
