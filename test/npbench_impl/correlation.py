import parpy
from parpy.operators import sum
import torch

@parpy.jit
def parpy_kernel(corr, data, M):
    parpy.label('i')
    for i in range(M-1):
        parpy.label('j')
        for j in range(i+1, M):
            corr[i, j] = sum(data[:, i] * data[:, j])
            corr[j, i] = corr[i, j]

def correlation(M, float_n, data, opts, compile_only=False):
    mean = torch.mean(data, axis=0)
    stddev = torch.std(data, unbiased=False, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= torch.sqrt(float_n) * stddev
    corr = torch.eye(M, dtype=data.dtype)
    if compile_only:
        args = [corr, data, M]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(corr, data, M, opts=opts)
    return corr
