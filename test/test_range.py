import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

def upper_bound_range(x, N):
    parpy.label('i')
    for i in range(N):
        x[i] = i

def no_step_range(x, N):
    parpy.label('i')
    for i in range(1, N):
        x[i] = i

def step_range(x, N):
    parpy.label('i')
    for i in range(1, N, 2):
        x[i] = i

def negative_step_range(x, N):
    parpy.label('i')
    for i in range(N-1, -1, -1):
        x[i] = i

def range_helper(fn, backend, compile_only):
    N = 100
    x = torch.zeros((N,), dtype=torch.int64)
    p = {'i': parpy.threads(32)}
    if compile_only:
        s = parpy.print_compiled(fn, [x, N], par_opts(backend, p))
        assert len(s) != 0
    else:
        x_device = x.detach().clone()
        parpy.jit(fn)(x_device, N, opts=par_opts(backend, p))
        fn(x, N)
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
        @parpy.jit
        def zero_step(x, N):
            for i in range(0, N, 0):
                x[i] = i
    assert e_info.match(r"step size must be non-zero")

def single_ranges(x, N):
    parpy.label('i')
    for i in parpy.builtin.ranges(N):
        x[0, i] = i

def multiloop_ranges(x, N):
    parpy.label('i')
    for i, j in parpy.builtin.ranges(N, N):
        x[i, j] = i + j

def triple_ranges(x, N):
    parpy.label('i')
    for i, j, k in parpy.builtin.ranges(N, N, N):
        if k == 0:
            x[i, j] = i + j

def bounds_ranges(x, N):
    parpy.label('i')
    for i, j in parpy.builtin.ranges((1, N-1), (1, N-1)):
        x[i, j] = i + j

def bounds_and_steps_ranges(x, N):
    parpy.label('i')
    for i, j in parpy.builtin.ranges((1, N-1, 2), N):
        x[i, j] = i + j

ranges_funs = [
    single_ranges,
    multiloop_ranges,
    triple_ranges,
    bounds_ranges,
    bounds_and_steps_ranges,
]

def parpy_ranges_helper(fn, backend, compile_only):
    N = 10
    x = torch.zeros((N, N), dtype=torch.int32)
    p = {'i': parpy.threads(32)}
    if compile_only:
        s = parpy.print_compiled(fn, [x, N], par_opts(backend, p))
        assert len(s) != 0
    else:
        x_dev = x.detach().clone()
        parpy.jit(fn)(x_dev, N, opts=par_opts(backend, p))
        fn(x, N)
        assert torch.allclose(x, x_dev)

@pytest.mark.parametrize('fn', ranges_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_parpy_ranges_run(fn, backend):
    run_if_backend_is_enabled(backend, lambda: parpy_ranges_helper(fn, backend, False))

@pytest.mark.parametrize('fn', ranges_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_parpy_ranges_compile(fn, backend):
    run_if_backend_is_enabled(backend, lambda: parpy_ranges_helper(fn, backend, True))

def test_parpy_dependent_ranges_fails():
    def dependent_ranges(x, N):
        for i, j in parpy.builtin.ranges(N, (i+1, N)):
            x[i, j] = i + j
    N = 10
    x = torch.zeros((N, N), dtype=torch.int32)
    with pytest.raises(RuntimeError) as e_info:
        parpy.jit(dependent_ranges)(x, N)
    assert e_info.match(r"dependent range bounds")
