import parir
from parir import ParKind, ParSpec
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def copy(x, y, N):
    for i in range(N):
        y[i] = x[i]

def copy_wrap(x, p=None):
    N, = x.shape
    y = torch.empty_like(x)
    copy(x, y, N, parallelize=p, cache=False)
    return y

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_gpu():
    x = torch.randn(10, dtype=torch.float32)
    y1 = copy_wrap(x)
    p = { "i" : ParSpec(ParKind.GpuThreads(1024)) }
    y2 = copy_wrap(x.cuda(), p).cpu()
    torch.cuda.synchronize()
    assert torch.allclose(y1, y2, atol=1e-5)
