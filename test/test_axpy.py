import parir
from parir.parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def axpy(a, x, y, out, N):
    for i in range(N):
        out[i] = a * x[i] + y[i]

def axpy_wrap(a, x, y, parallelize=None):
    N = len(x)
    out = torch.empty_like(x)
    axpy(a, x, y, out, N, parallelize=parallelize)
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_axpy_gpu():
    N = 100
    a = float(torch.randn(1, dtype=torch.float32)[0])
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    expected = axpy_wrap(a, x, y)

    # Run each iteration of the 'i' loop on a separate GPU thread.
    parir.clear_cache()
    par = { "i": [ParKind.GpuThreads(128)] }
    actual = axpy_wrap(a, x.cuda(), y.cuda(), par).cpu()
    torch.cuda.synchronize()
    assert torch.allclose(expected, actual, atol=1e-5)
