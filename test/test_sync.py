import parir
import pytest
import torch

from common import *

@parir.jit
def sum_rows(x, N, out):
    parir.label('N')
    for i in range(N):
        parir.label('M')
        out[i] = parir.sum(x[i,:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_multi_block(backend):
    def helper():
        N, M = 10, 20
        x = torch.randn(N, M, dtype=torch.float32)

        # Parallelize reduction within a single block (n = 1024)
        out1 = torch.zeros(N, dtype=x.dtype)
        p1 = {'N': parir.threads(N), 'M': parir.threads(1024)}
        sum_rows(x, N, out1, opts=par_opts(backend, p1))

        # Parallelize reduction across multiple blocks (n > 1024)
        out2 = torch.zeros_like(out1)
        p2 = {'N': parir.threads(N), 'M': parir.threads(2048)}
        sum_rows(x, N, out2, opts=par_opts(backend, p2))

        assert torch.allclose(out1, out2, atol=1e-6)
    run_if_backend_is_enabled(backend, helper)

@parir.jit
def normalize(x, N):
    parir.label('N')
    for i in range(N):
        parir.label('M_1')
        s = parir.sum(x[i,:])
        parir.label('M_2')
        x[i,:] /= s

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.parametrize('backend', compiler_backends)
def test_varying_parallelism(backend):
    def helper():
        N, M = 10, 20
        x = torch.randn(N, M, dtype=torch.float32)
        p = {
            'N': parir.threads(N),
            'M_1': parir.threads(128),
            'M_2': parir.threads(64)
        }
        normalize(x, N, opts=par_opts(backend, p))
        assert torch.allclose(torch.sum(x, axis=-1), torch.ones(N))
    run_if_backend_is_enabled(backend, helper)

@parir.jit
def sum_exp_3d(x, N, M, out):
    parir.label('N')
    for i in range(N):
        parir.label('M')
        for j in range(M):
            parir.label('K_1')
            x[i,j,:] = parir.exp(x[i,j,:])
        parir.label('M')
        for j in range(M):
            parir.label('K_2')
            out[i,j] = parir.sum(x[i,j,:])

def sum_exp_3d_wrap(backend, par):
    N, M, K = 10, 20, 30
    x = torch.randn(N, M, K, dtype=torch.float32)
    x_2 = x.detach().clone()
    out = torch.zeros(N, M, dtype=x.dtype)
    sum_exp_3d(x, N, M, out, opts=par_opts(backend, par))
    ref_out = torch.zeros_like(out)
    sum_exp_3d(x_2, N, M, ref_out, opts=seq_opts(backend))
    assert torch.allclose(out, ref_out, atol=1e-5)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_imbalanced_parallelism(backend):
    def helper():
        p = {
            'N': parir.threads(10),
            'M': parir.threads(20),
            'K_1': parir.threads(32),
            'K_2': parir.threads(64),
        }
        sum_exp_3d_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.parametrize('backend', compiler_backends)
def test_imbalanced_parallelism_in_sequential_for(backend):
    def helper():
        p = {
            'N': parir.threads(10),
            'K_1': parir.threads(32),
            'K_2': parir.threads(64),
        }
        sum_exp_3d_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)
