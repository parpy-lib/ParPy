import numpy as np
import parir
import pytest
import torch

from common import *

@parir.jit
def syrk(alpha, beta, C, A, N, M):
    parir.label('i')
    for i in range(N):
        parir.label('j')
        for j in range(i+1):
            # NOTE: In the generated code, only one thread should write to this
            # memory location to avoid a concurrency bug.
            C[i,j] = C[i,j] * beta
            for k in range(M):
                C[i,j] = C[i,j] + alpha * A[i,k] * A[j,k]

def syrk_data():
    M = 20
    N = 30
    alpha = 1.5
    beta = 1.2
    C = torch.from_numpy(
        np.fromfunction(lambda i,j: ((i * j + 2) % N) / M, (N, N), dtype=np.float32)
    )
    A = torch.from_numpy(
        np.fromfunction(lambda i,j: ((i * j + 2) % N) / N, (N, M), dtype=np.float32)
    )
    return alpha, beta, C, A, N, M

def syrk_par_spec(N, nthreads):
    return {
        'i': parir.threads(N),
        'k': parir.threads(nthreads).reduce()
    }

def syrk_run_par(backend, nthreads):
    def helper():
        alpha, beta, C, A, N, M = syrk_data()

        # Run sequentially to produce a reference solution
        C_ref = C.clone()
        syrk(alpha, beta, C_ref, A, N, M, opts=seq_opts(backend))

        # Run in parallel using the specified number of threads and validate the
        # result
        p = syrk_par_spec(N, nthreads)
        C_device = C.clone()
        syrk(alpha, beta, C_device, A, N, M, opts=par_opts(backend, p))

        assert torch.allclose(C_ref, C_device), f"Run failed using {nthreads} threads"
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk_parallel_64(backend):
    syrk_run_par(backend, 64)

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk_parallel_128(backend):
    syrk_run_par(backend, 128)

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk_parallel_256(backend):
    syrk_run_par(backend, 256)

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk_parallel_512(backend):
    syrk_run_par(backend, 512)

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk_parallel_1024(backend):
    syrk_run_par(backend, 1024)
