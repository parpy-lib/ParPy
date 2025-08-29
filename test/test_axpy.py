import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

def axpy(a, x, y, out, N):
    parpy.label('i')
    out[:] = a * x[:] + y[:]

def axpy_wrap(a, x, y, N, opts=None):
    out = torch.zeros_like(x)
    if opts is None:
        axpy(a, x, y, out, N)
    else:
        parpy.jit(axpy)(a, x, y, out, N, opts=opts)
    return out

def axpy_test_data():
    N = 100
    a = torch.randn(1, dtype=torch.float32)[0]
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return N, a, x, y

@pytest.mark.parametrize('backend', compiler_backends)
def test_axpy_gpu(backend):
    def helper():
        N, a, x, y = axpy_test_data()
        expected = axpy_wrap(a, x, y, N, None)
        p = {'i': parpy.threads(128)}
        actual = axpy_wrap(a, x, y, N, par_opts(backend, p))
        assert torch.allclose(expected, actual, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_axpy_compile_fails_no_parallelism(backend):
    N, a, x, y = axpy_test_data()
    out = torch.zeros_like(x)
    with pytest.raises(RuntimeError) as e_info:
        parpy.print_compiled(axpy, [a, x, y, out, N], par_opts(backend, {}))
    assert e_info.match(r".*does not contain any parallelism.*")

@pytest.mark.parametrize('backend', compiler_backends)
def test_axpy_compiles_with_parallelism(backend):
    N, a, x, y = axpy_test_data()
    out = torch.zeros_like(x)
    p = {'i': parpy.threads(128)}
    s = parpy.print_compiled(axpy, [a, x, y, out, N], par_opts(backend, p))
    assert len(s) != 0
