import parpy
from parpy.operators import int64, max, min
import torch

@parpy.jit
def compute_helper(array_1, array_2, N, M, a, b, c, out):
    parpy.label('i')
    for i in range(N):
        parpy.label('j')
        for j in range(M):
            clamped = min(max(array_1[i,j], int64(2)), int64(10))
            out[i,j] = clamped * a + array_2[i,j] * b + c

def compute(array_1, array_2, a, b, c, opts, compile_only=False):
    out = torch.zeros_like(array_1)
    N, M = array_1.shape
    if compile_only:
        args = [array_1, array_2, N, M, a, b, c, out]
        return parpy.print_compiled(compute_helper, args, opts)
    compute_helper(array_1, array_2, N, M, a, b, c, out, opts=opts)
    return out
