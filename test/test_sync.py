import numpy as np
import parpy
import pytest

from common import *

np.random.seed(1234)

@parpy.jit
def sum_rows(x, N, out):
    parpy.label('N')
    for i in range(N):
        parpy.label('M')
        out[i] = parpy.operators.sum(x[i,:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_multi_block(backend):
    def helper():
        N, M = 10, 20
        x = np.random.randn(N, M).astype(np.float32)

        # Parallelize reduction within a single block (n = 1024)
        out1 = np.zeros(N, dtype=x.dtype)
        p1 = {'N': parpy.threads(N), 'M': parpy.threads(1024)}
        sum_rows(x, N, out1, opts=par_opts(backend, p1))

        # Parallelize reduction across multiple blocks (n > 1024)
        out2 = np.zeros_like(out1)
        p2 = {'N': parpy.threads(N), 'M': parpy.threads(2048)}
        sum_rows(x, N, out2, opts=par_opts(backend, p2))

        assert np.allclose(out1, out2, atol=1e-6)
    run_if_backend_is_enabled(backend, helper)

@parpy.jit
def normalize(x, N):
    parpy.label('N')
    for i in range(N):
        parpy.label('M_1')
        s = parpy.operators.sum(x[i,:])
        parpy.label('M_2')
        x[i,:] /= s

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.parametrize('backend', compiler_backends)
def test_varying_parallelism(backend):
    def helper():
        N, M = 10, 20
        x = np.randn(N, M).astype(np.float32)
        p = {
            'N': parpy.threads(N),
            'M_1': parpy.threads(128),
            'M_2': parpy.threads(64)
        }
        normalize(x, N, opts=par_opts(backend, p))
        assert np.allclose(np.sum(x, axis=-1), np.ones(N))
    run_if_backend_is_enabled(backend, helper)

def sum_exp_3d(x, N, M, out):
    parpy.label('N')
    for i in range(N):
        parpy.label('M')
        for j in range(M):
            parpy.label('K_1')
            x[i,j,:] = parpy.operators.exp(x[i,j,:])
        parpy.label('M')
        for j in range(M):
            parpy.label('K_2')
            out[i,j] = parpy.operators.sum(x[i,j,:])

def sum_exp_3d_wrap(backend, par):
    N, M, K = 10, 20, 30
    x = np.random.randn(N, M, K).astype(np.float32)
    x_2 = np.copy(x)
    out = np.zeros((N, M), dtype=x.dtype)
    parpy.jit(sum_exp_3d)(x, N, M, out, opts=par_opts(backend, par))
    ref_out = np.zeros_like(out)
    sum_exp_3d(x_2, N, M, ref_out)
    assert np.allclose(out, ref_out, atol=1e-5)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_imbalanced_parallelism(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'M': parpy.threads(20),
            'K_1': parpy.threads(32),
            'K_2': parpy.threads(64),
        }
        sum_exp_3d_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.parametrize('backend', compiler_backends)
def test_imbalanced_parallelism_in_sequential_for(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'K_1': parpy.threads(32),
            'K_2': parpy.threads(64),
        }
        sum_exp_3d_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)
