import prickle
import pytest
import torch

from common import *

@prickle.jit
def prickle_add(x: prickle.float32, y: prickle.float32):
    return x + y

@prickle.jit
def prickle_mul(x: prickle.float32, y: prickle.float32):
    return x * y

@prickle.jit
def prickle_add_mul(x: prickle.float32, y: prickle.float32, z: prickle.float32):
    return prickle_mul(prickle_add(x, y), z)

@prickle.jit
def prickle_add_direct(x, y, N):
    prickle.label('N')
    y[:] = prickle_add(x[:], y[:])

@prickle.jit
def prickle_add_mul_nested(x, y, N):
    prickle.label('N')
    y[:] = prickle_add_mul(x[:], y[:], x[:])

@prickle.jit
def prickle_sum_call(x, y):
    with prickle.gpu:
        y[0] = prickle.sum(x[:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_direct_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': prickle.threads(10)}
        prickle_add_direct(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': prickle.threads(10)}
        prickle_add_mul_nested(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x**2, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_sum_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros(1)
        p = {'N': prickle.threads(10)}
        prickle_sum_call(x, y, opts=par_opts(backend, p))
        assert torch.allclose(torch.sum(x), y)
    run_if_backend_is_enabled(backend, helper)

@prickle.jit
def add_inplace(x, y, M):
    prickle.label("1d")
    y[:] += x[:]

@prickle.jit
def add_2d_inplace(x, y, N, M):
    prickle.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)

@prickle.jit
def add_2d_inplace_x2(x, y, z, w, N, M):
    prickle.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)
        add_inplace(z[i], w[i], M)

@prickle.jit
def add_3d_inplace(x, y, N, M, K):
    prickle.label('3d')
    for i in range(N):
        add_2d_inplace(x[i], y[i], M, K)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        p = {'2d': prickle.threads(10), '1d': prickle.threads(15)}
        add_2d_inplace(x, y, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_different_types(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        z = torch.randint(0, 10, (10, 15), dtype=torch.int32)
        w = torch.zeros_like(z)
        p = {'2d': prickle.threads(10), '1d': prickle.threads(15)}
        add_2d_inplace_x2(x, y, z, w, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
        assert torch.allclose(z, w)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_dependency(backend):
    def helper():
        x = torch.randn(10, 20, 30)
        y = torch.zeros_like(x)
        p = {
            '3d': prickle.threads(10),
            '2d': prickle.threads(20),
            '1d': prickle.threads(30)
        }
        add_3d_inplace(x, y, 10, 20, 30, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

def test_call_non_decorated_function_fails():
    # This function is intentionally not decorated with '@prickle.jit'
    def non_decorated_add(x, y, M):
        for i in range(M):
            y[i] += x[i]
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def add_2d(x, y, N, M):
            prickle.label('N')
            for i in range(N):
                non_decorated_add(x[i], y[i], M)
    assert e_info.match(r".*unknown function non_decorated_add.*")

def test_recursive_call_fails():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def reset(x, i):
            with prickle.gpu:
                if i > 0:
                    x[i] = 0.0
                    reset(x, i-1)
    assert e_info.match(r".*unknown function reset.*")
