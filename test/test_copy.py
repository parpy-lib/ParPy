import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def copy(x, y):
    parir.label('i')
    y[:] = x[:]

def copy_wrap(x, p):
    y = torch.empty_like(x)
    copy(x, y, parallelize=p, cache=False)
    return y

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_gpu():
    x = torch.randn(10, dtype=torch.float32, device='cuda')
    p = {'i': parir.threads(1024)}
    y = copy_wrap(x, p)
    assert torch.allclose(x, y)

def test_copy_compiles():
    x = torch.randn(10, dtype=torch.float32)
    y = torch.empty_like(x)
    p = {'i': parir.threads(1024)}
    s = parir.print_compiled(copy, [x, y], p)
    assert len(s) != 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_run_compiled_string():
    x = torch.randn(10, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    p = {'i': parir.threads(1024)}
    code = parir.print_compiled(copy, [x, y], p)
    fn = parir.compile_string(copy.__name__, code)
    fn(x, y)
    assert torch.allclose(x, y)
