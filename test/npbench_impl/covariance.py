import prickle
import torch


@prickle.jit
def covariance_prickle(cov, data, float_n, M):
    prickle.label('i')
    for i in range(M):
        prickle.label('j')
        for j in range(i, M):
            s = prickle.sum(data[:, i] * data[:, j])
            cov[i, j] = s / (float_n - 1.0)
            cov[j, i] = cov[i, j]

def covariance(M, float_n, data, opts):
    mean = torch.mean(data, axis=0)
    data -= mean
    cov = torch.zeros((M, M), dtype=data.dtype, device=data.device)
    covariance_prickle(cov, data, float_n, M, opts=opts)
    return cov
