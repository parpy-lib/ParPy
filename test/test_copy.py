import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def copy(x, y):
    parir.label('i')
    y[:] = x[:]

def copy_wrap(x, p=None):
    y = torch.empty_like(x)
    if p is None:
        copy(x, y, seq=True)
    else:
        copy(x, y, parallelize=p, cache=False)
    return y

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_gpu():
    x = torch.randn(10, dtype=torch.float32)
    y1 = copy_wrap(x)
    p = {'i': [parir.threads(1024)]}
    y2 = copy_wrap(x.cuda(), p).cpu()
    torch.cuda.synchronize()
    assert torch.allclose(y1, y2, atol=1e-5)

def test_copy_compiles():
    x = torch.randn(10, dtype=torch.float32)
    y = torch.empty_like(x)
    p = {'i': [parir.threads(1024)]}
    s = parir.print_compiled(copy, [x, y], p)
    assert len(s) != 0
