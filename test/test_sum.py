import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def sum_rows(x, out, N):
    parir.label("outer")
    for i in range(N):
        parir.label("inner")
        out[i] = parir.sum(x[i,:])

def sum_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    if p is None:
        sum_rows(x, out, N, seq=True)
    else:
        sum_rows(x, out, N, parallelize=p, cache=False)
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
    p = {'outer': [parir.threads(N)]}
    compare_sum(N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sum_inner_and_outer_parallel_gpu():
    N = 100
    M = 50
    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(128), parir.reduce()]
    }
    compare_sum(N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sum_multi_block_reduction_fails():
    N = 100
    M = 2048
    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(M), parir.reduce()]
    }
    with pytest.raises(RuntimeError) as e_info:
        compare_sum(N, M, p)
    assert e_info.match(r".*1024 threads.*")

def test_sum_compiles():
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty(N, dtype=torch.float32)
    p = {'outer': [parir.threads(N)]}
    s1 = parir.print_compiled(sum_rows, [x, out, N], p)
    assert len(s1) != 0

    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(128), parir.reduce()]
    }
    s2 = parir.print_compiled(sum_rows, [x, out, N], p)
    assert len(s2) != 0
