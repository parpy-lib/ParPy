import parir
import pytest
import torch

from common import *

torch.manual_seed(1234)

@parir.jit
def copy(x, y):
    parir.label('i')
    y[:] = x[:]

def copy_wrap(x, opts):
    y = torch.empty_like(x)
    copy(x, y, opts=opts)
    return y

@pytest.mark.parametrize('backend', compiler_backends)
def test_copy_gpu(backend):
    def helper():
        x = torch.randn(10, dtype=torch.float32)
        p = {'i': parir.threads(1024)}
        y = copy_wrap(x, par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_copy_compiles(backend):
    x = torch.randn(10, dtype=torch.float32)
    y = torch.empty_like(x)
    p = {'i': parir.threads(1024)}
    s = parir.print_compiled(copy, [x, y], par_opts(backend, p))
    assert len(s) != 0

@pytest.mark.parametrize('backend', compiler_backends)
def test_copy_run_compiled_string(backend):
    def helper():
        x = torch.randn(10, dtype=torch.float32)
        y = torch.empty_like(x)
        p = {'i': parir.threads(1024)}
        code = parir.print_compiled(copy, [x, y], par_opts(backend, p))
        fn = parir.compile_string(copy.__name__, code, par_opts(backend, p))
        fn(x, y)
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)
