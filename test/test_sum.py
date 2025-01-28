import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def sum_rows(x, out, N, M):
    for i in range(N):
        s = 0.0
        for j in range(M):
            s = s + x[i * M + j]
        out[i] = s

def sum_wrap(x, p=None):
    N, M = x.shape
    xflat = x.flatten()
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    sum_rows(xflat, out, N, M, parallelize=p, cache=False)
    return out

def compare_sum(N, M, p):
    x = torch.randn((N, M), dtype=torch.float32)
    # Run sequentially on CPU and compare result against parallelized version
    expected = sum_wrap(x)
    actual = sum_wrap(x.cuda(), p).cpu()
    torch.cuda.synchronize()
    assert torch.allclose(expected, actual, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sum_outer_parallel_gpu():
    N = 100
    M = 50
    p = { "i" : [ParKind.GpuThreads(N)] }
    compare_sum(N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sum_inner_and_outer_parallel_gpu():
    N = 100
    M = 50
    p = {
        "i": [ParKind.GpuThreads(N)],
        "j": [ParKind.GpuThreads(128)]
    }
    compare_sum(N, M, p)
