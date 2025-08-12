import prickle
import torch

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

@prickle.jit
def softmax_kernel(x, out, N, M):
    prickle.label('i')
    for i in range(N):
        prickle.label('j')
        maxv = prickle.max(x[i,:])
        prickle.label('j')
        out[i,:] = prickle.exp(x[i,:] - maxv)
        prickle.label('j')
        s = prickle.sum(out[i,:])
        prickle.label('j')
        out[i,:] /= s

def softmax(x, opts):
    N, M = x.shape
    out = torch.empty_like(x)
    softmax_kernel(x, out, N, M, opts=opts)
    return out

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3, opts):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3, opts)  # Softmax call can be omitted if necessary
    return x
