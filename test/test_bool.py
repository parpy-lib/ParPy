import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

def store_gt(x, y, out, N):
    parpy.label('i')
    out[:] = x[:] < y[:]

def reduce_and(x, out):
    with parpy.gpu:
        out[0] = out[0] and x[0]

def bool_test_data():
    N = 100
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return x, y, N

def bool_wrap(x, y, opts=None):
    N, = x.shape
    tmp = torch.zeros(N, dtype=torch.bool)
    if opts is None:
        store_gt(x, y, tmp, N)
    else:
        parpy.jit(store_gt)(x, y, tmp, N, opts=opts)
    out = torch.zeros(1, dtype=torch.bool)
    if opts is None:
        reduce_and(tmp, out)
    else:
        parpy.jit(reduce_and)(tmp, out, opts=opts)
    return out

@pytest.mark.parametrize('backend', compiler_backends)
def test_bool_gpu(backend):
    def helper():
        x, y, N = bool_test_data()
        expected = bool_wrap(x, y)
        p = {'i': parpy.threads(N)}
        actual = bool_wrap(x, y, par_opts(backend, p))
        assert torch.allclose(expected, actual, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_bool_compiles(backend):
    x, y, N = bool_test_data()
    tmp = torch.zeros_like(x, dtype=torch.bool)
    p = {'i': parpy.threads(64)}
    s = parpy.print_compiled(store_gt, [x, y, tmp, N], par_opts(backend, p))
    assert len(s) != 0
    res = torch.zeros(1, dtype=torch.bool)
    s = parpy.print_compiled(reduce_and, [tmp, res], par_opts(backend, {}))
    assert len(s) != 0
