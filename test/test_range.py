import prickle
import pytest
import torch

from common import *

torch.manual_seed(1234)

@prickle.jit
def upper_bound_range(x, N):
    prickle.label('i')
    for i in range(N):
        x[i] = i

@prickle.jit
def no_step_range(x, N):
    prickle.label('i')
    for i in range(1, N):
        x[i] = i

@prickle.jit
def step_range(x, N):
    prickle.label('i')
    for i in range(1, N, 2):
        x[i] = i

@prickle.jit
def negative_step_range(x, N):
    prickle.label('i')
    for i in range(N-1, -1, -1):
        x[i] = i

def range_helper(fn, backend, compile_only):
    N = 100
    x = torch.zeros((N,), dtype=torch.int64)
    p = {'i': prickle.threads(32)}
    if compile_only:
        s = prickle.print_compiled(fn, [x, N], par_opts(backend, p))
        assert len(s) != 0
    else:
        x_device = x.detach().clone()
        upper_bound_range(x_device, N, opts=par_opts(backend, p))
        upper_bound_range(x, N, opts=seq_opts(backend))
        assert torch.allclose(x, x_device)

range_funs = [upper_bound_range, no_step_range, step_range, negative_step_range]

@pytest.mark.parametrize('fn', range_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_range_gpu(fn, backend):
    run_if_backend_is_enabled(backend, lambda: range_helper(fn, backend, False))

@pytest.mark.parametrize('fn', range_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_range_compile(fn, backend):
    run_if_backend_is_enabled(backend, lambda: range_helper(fn, backend, True))

def test_zero_step_fails():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def zero_step(x, N):
            for i in range(0, N, 0):
                x[i] = i
    assert e_info.match(r"Range step size must be non-zero")
