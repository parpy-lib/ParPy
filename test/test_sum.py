import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def sum_rows(x, out, N, M):
    for i in range(N):
        out[i] = 0.0
        for j in range(M):
            out[i] = out[i] + x[i, j]

def sum_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    sum_rows(x, out, N, M, parallelize=p)
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
    p = {'i': [ParKind.GpuThreads(N)]}
    compare_sum(N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sum_inner_and_outer_parallel_gpu():
    N = 100
    M = 50
    p = {
        "i": [ParKind.GpuThreads(N)],
        "j": [ParKind.GpuThreads(128), ParKind.GpuReduction()]
    }
    compare_sum(N, M, p)

def test_sum_compiles():
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty(N, dtype=torch.float32)
    p = {'i': [ParKind.GpuThreads(N)]}
    s1 = parir.print_compiled(sum_rows, [x, out, N, M], p)
    assert len(s1) != 0

    p = {
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(128), ParKind.GpuReduction()]
    }
    s2 = parir.print_compiled(sum_rows, [x, out, N, M], p)
    assert len(s2) != 0
