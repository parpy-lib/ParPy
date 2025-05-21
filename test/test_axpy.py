import parir
import pytest
import torch

from common import *

torch.manual_seed(1234)

@parir.jit
def axpy(a, x, y, out, N):
    parir.label('i')
    out[:] = a * x[:] + y[:]

def axpy_wrap(a, x, y, N, p=None):
    out = torch.empty_like(x)
    if p is None:
        axpy(a, x, y, out, N, opts=seq_opts())
    else:
        axpy(a, x, y, out, N, opts=par_opts(p))
    return out

def axpy_test_data():
    N = 100
    a = torch.randn(1, dtype=torch.float32)[0]
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return N, a, x, y

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_axpy_gpu():
    N, a, x, y = axpy_test_data()
    expected = axpy_wrap(a, x, y, N)

    # Run each iteration of the 'i' loop on a separate GPU thread.
    p = {'i': parir.threads(128)}
    actual = axpy_wrap(a.cuda(), x.cuda(), y.cuda(), N, p).cpu()
    torch.cuda.synchronize()
    assert torch.allclose(expected, actual, atol=1e-5)

def test_axpy_compile_fails_no_parallelism():
    N, a, x, y = axpy_test_data()
    out = torch.empty_like(x)
    with pytest.raises(RuntimeError) as e_info:
        parir.print_compiled(axpy, [a, x, y, out, N])
    assert e_info.match(r".*does not contain any parallelism.*")

def test_axpy_compiles_with_parallelism():
    N, a, x, y = axpy_test_data()
    out = torch.empty_like(x)
    p = {'i': parir.threads(128)}
    s = parir.print_compiled(axpy, [a, x, y, out, N], par_opts(p))
    assert len(s) != 0
