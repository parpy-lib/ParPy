import parir
import pytest
import torch

from common import *

torch.manual_seed(1234)

@parir.jit
def store_gt(x, y, out, N):
    parir.label('i')
    out[:] = x[:] < y[:]

@parir.jit
def reduce_and(x, out):
    with parir.gpu:
        out[0] = out[0] and x[0]

def bool_test_data():
    N = 100
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return x, y, N

def bool_wrap(x, y, opts):
    N, = x.shape
    tmp = torch.empty(N, dtype=torch.bool, device=x.device)
    store_gt(x, y, tmp, N, opts=opts)
    out = torch.empty(1, dtype=torch.bool, device=x.device)
    reduce_and(tmp, out, opts=opts)
    return out

@pytest.mark.parametrize('backend', compiler_backends)
def test_bool_gpu(backend):
    def helper():
        x, y, N = bool_test_data()
        expected = bool_wrap(x, y, seq_opts(backend))
        p = {'i': parir.threads(N)}
        actual = bool_wrap(x, y, par_opts(backend, p))
        assert torch.allclose(expected, actual, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_bool_compiles(backend):
    x, y, N = bool_test_data()
    tmp = torch.empty_like(x, dtype=torch.bool)
    p = {'i': parir.threads(64)}
    s = parir.print_compiled(store_gt, [x, y, tmp, N], par_opts(backend, p))
    assert len(s) != 0

    res = torch.empty(1, dtype=torch.bool)
    s = parir.print_compiled(reduce_and, [tmp, res], seq_opts(backend))
    assert len(s) != 0
