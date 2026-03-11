import parpy
from parpy.types import *
import pytest
import torch

from common import *

T = type_var()
M = shape_var()
N = shape_var()
K = shape_var()
R = shape_var()

@parpy.callback
def gemm(alpha, beta, A, B, C):
    A_t, B_t, C_t = A.torch(), B.torch(), C.torch()
    C_t = alpha * A_t @ B_t + beta * C_t
    C.copy_from(C_t)

@parpy.jit
def ones_init(C: buffer(T, [M, N])):
    for i in range(M):
        for j in range(N):
            C[i,j] = parpy.builtin.convert(1, T)

@parpy.jit
def gemm_parpy(
        alpha: T,
        beta: T,
        A: buffer(T, [R, M, K]),
        B: buffer(T, [R, K, N]),
        C: buffer(T, [R, M, N])
):
    for i in range(R):
        parpy.builtin.inline(ones_init(C[i]))
        gemm(alpha, beta, A[i], B[i], C[i])

@pytest.mark.parametrize('backend', compiler_backends)
def test_gemm_callback(backend):
    def helper():
        alpha = torch.tensor(1.5, dtype=torch.float32)
        beta = torch.tensor(2.5, dtype=torch.float32)
        M, N, K, R = 32, 64, 128, 10
        A = torch.randn(R, M, K, dtype=torch.float32)
        B = torch.randn(R, K, N, dtype=torch.float32)
        C = torch.randn(R, M, N, dtype=torch.float32)
        p = {
            'M': parpy.threads(M),
            'N': parpy.threads(N),
            'R': parpy.threads(R),
        }
        opts = par_opts(backend, p)
        gemm_parpy(alpha, beta, A, B, C, opts=opts)
        expected = alpha * A @ B + beta * torch.ones(R, M, N, dtype=torch.float32)
        assert torch.allclose(C, expected, atol=1e-3)

        # Also test running this for the native Triton backend
        if backend == parpy.CompileBackend.Triton:
            opts.triton_native = True
            gemm_parpy(alpha, beta, A, B, C, opts=opts)
            assert torch.allclose(C, expected, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

@parpy.jit
def second_entry(a: parpy.types.buffer(T, [N])):
    for i in range(N):
        a[i] += parpy.builtin.convert(1, T)

@parpy.callback
def rec_callback(a):
    opts = par_opts(a.backend(), {'N': parpy.threads(32)})
    second_entry(a, opts=opts)

@parpy.jit
def first_entry(a: parpy.types.buffer(T, [M, N])):
    for i in range(M):
        rec_callback(a[i])

@pytest.mark.parametrize('backend', compiler_backends)
def test_callback_performs_jit(backend):
    def helper():
        a = torch.randn(20, 10, dtype=torch.float32)
        b = a.detach().clone()
        opts = par_opts(backend, {'M': parpy.threads(32)})
        first_entry(b, opts=opts)
        assert torch.allclose(a + 1.0, b)

        # Also test running this for the native Triton backend
        if backend == parpy.CompileBackend.Triton:
            b = a.detach().clone()
            opts.triton_native = True
            first_entry(b, opts=opts)
            assert torch.allclose(a + 1.0, b)
    run_if_backend_is_enabled(backend, helper)

@parpy.callback
def odd_callback(a, x):
    a_t = a.torch()
    a_t[1,:] += x
    # We need to copy if we run this on Metal
    if a.backend == parpy.CompileBackend.Metal:
        a.copy_from(a_t)

@parpy.callback
def even_callback(a, x):
    a_t = a.torch()
    a_t[0,:] += x
    # We need to copy if we run this on Metal
    if a.backend == parpy.CompileBackend.Metal:
        a.copy_from(a_t)

@parpy.jit
def cond_write(a, x):
    with parpy.gpu:
        a[0] = a[0]
    if x & 1 == 0:
        odd_callback(a, x)
    else:
        even_callback(a, x)

@pytest.mark.parametrize('backend', compiler_backends)
def test_conditional_callback(backend):
    def helper():
        a = torch.zeros(2, 10, dtype=torch.float32)
        opts = par_opts(backend, {})
        cond_write(a, 0, opts=opts)
        cond_write(a, 1, opts=opts)

        # Also test running this for the native Triton backend
        if backend == parpy.CompileBackend.Triton:
            opts.triton_native = True
            cond_write(a, 0, opts=opts)
            cond_write(a, 1, opts=opts)
    run_if_backend_is_enabled(backend, helper)
