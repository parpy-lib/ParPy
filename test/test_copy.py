import parir
from parir.parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def copy(x, y, N):
    for i in range(N):
        y[i] = x[i]

def copy_wrap(x, parallelize=None):
    N, = x.shape
    y = torch.empty_like(x)
    copy(x, y, N, parallelize=parallelize)
    return y

def test_copy_gpu():
    x = torch.randn(10, dtype=torch.float32)
    y1 = copy_wrap(x)
    p = { "i" : [ParKind.GpuThreads(1024)] }
    if torch.cuda.is_available():
        y2 = copy_wrap(x.cuda(), parallelize=p).cpu()
        torch.cuda.synchronize()
        assert torch.allclose(y1, y2, atol=1e-5)
    else:
        with pytest.raises(AssertionError):
            _ = copy_wrap(x.cuda(), parallelize=p).cpu()
