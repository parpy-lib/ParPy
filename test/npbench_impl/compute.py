import prickle
import torch

@prickle.jit
def compute_helper(array_1, array_2, N, M, a, b, c, out):
    prickle.label('i')
    for i in range(N):
        prickle.label('j')
        for j in range(M):
            clamped = prickle.min(prickle.max(array_1[i,j], prickle.int64(2)), prickle.int64(10))
            out[i,j] = clamped * a + array_2[i,j] * b + c

def compute(array_1, array_2, a, b, c, opts):
    out = torch.zeros_like(array_1)
    N, M = array_1.shape
    compute_helper(array_1, array_2, N, M, a, b, c, out, opts=opts)
    return out
