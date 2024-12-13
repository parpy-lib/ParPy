import parir
from parir.parir import ParKind
import torch

torch.manual_seed(1234)

@parir.jit
def sum_rows(x, out, N, M):
    for i in range(N):
        s = 0.0
        for j in range(M):
            s = s + x[i * M + j]
        out[i] = s

def sum_wrap(x, parallelize=None):
    N, M = x.shape
    xflat = x.flatten()
    out = torch.empty(N, dtype=x.dtype)
    sum_rows(xflat, out, N, M, parallelize=parallelize)
    return out

def test_sum_outer_parallel():
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    expected = sum_wrap(x)

    # Only parallelize over the outer loop
    par = { "i" : [ParKind.GpuBlocks(N)], }
    actual = sum_wrap(x, parallelize=par)
    assert torch.allclose(expected, actual, atol=1e-5)

def test_sum_inner_and_outer_parallel():
    N = 100
    M = 500
    x = torch.randn((N, M), dtype=torch.float32)
    expected = sum_wrap(x)

    # Run both the outer and the inner loops in parallel, by performing a
    # parallel reduction within each thread block.
    par = { "i": [ParKind.GpuBlocks(N)], "j": [ParKind.GpuThreads(128)] }
    actual = sum_wrap(x, parallelize=par)
    assert torch.allclose(expected, actual, atol=1e-5)
