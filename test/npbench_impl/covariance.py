import parpy
import torch


@parpy.jit
def covariance_parpy(cov, data, float_n, M):
    parpy.label('i')
    for i in range(M):
        parpy.label('j')
        for j in range(i, M):
            s = parpy.sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def covariance(M, float_n, data, opts, compile_only=False):
    mean = torch.mean(data, axis=0)
    data -= mean
    cov = torch.zeros((M, M), dtype=data.dtype)
    if compile_only:
        args = [cov, data, float_n, M]
        return parpy.print_compiled(covariance_parpy, args, opts)
    covariance_parpy(cov, data, float_n, M, opts=opts)
    return cov
