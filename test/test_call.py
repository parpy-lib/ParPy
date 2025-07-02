import parir
import pytest
import torch

from common import *

@parir.jit
def parir_add(x: parir.float32, y: parir.float32):
    return x + y

@parir.jit
def parir_mul(x: parir.float32, y: parir.float32):
    return x * y

@parir.jit
def parir_add_mul(x: parir.float32, y: parir.float32, z: parir.float32):
    return parir_mul(parir_add(x, y), z)

@parir.jit
def parir_add_direct(x, y, N):
    parir.label('N')
    y[:] = parir_add(x[:], y[:])

@parir.jit
def parir_add_mul_nested(x, y, N):
    parir.label('N')
    y[:] = parir_add_mul(x[:], y[:], x[:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_direct_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': parir.threads(10)}
        parir_add_direct(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': parir.threads(10)}
        parir_add_mul_nested(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x**2, y)
    run_if_backend_is_enabled(backend, helper)

@parir.jit
def add_inplace(x, y, M):
    parir.label("1d")
    y[:] += x[:]

@parir.jit
def add_2d_inplace(x, y, N, M):
    parir.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)

@parir.jit
def add_2d_inplace_x2(x, y, z, w, N, M):
    parir.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)
        add_inplace(z[i], w[i], M)

@parir.jit
def add_3d_inplace(x, y, N, M, K):
    parir.label('3d')
    for i in range(N):
        add_2d_inplace(x[i], y[i], M, K)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        p = {'2d': parir.threads(10), '1d': parir.threads(15)}
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
        p = {'2d': parir.threads(10), '1d': parir.threads(15)}
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
            '3d': parir.threads(10),
            '2d': parir.threads(20),
            '1d': parir.threads(30)
        }
        add_3d_inplace(x, y, 10, 20, 30, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

def test_call_non_decorated_function_fails():
    # This function is intentionally not decorated with '@parir.jit'
    def non_decorated_add(x, y, M):
        for i in range(M):
            y[i] += x[i]
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def add_2d(x, y, N, M):
            parir.label('N')
            for i in range(N):
                non_decorated_add(x[i], y[i], M)
    assert e_info.match(r".*unknown function non_decorated_add.*")

def test_recursive_call_fails():
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def reset(x, i):
            with parir.gpu:
                if i > 0:
                    x[i] = 0.0
                    reset(x, i-1)
    assert e_info.match(r".*unknown function reset.*")
