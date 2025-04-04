import numpy as np
import parir
import pytest
import torch

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
        np.fromfunction(lambda i,j: ((i * j + 2) % N) / M, (N, N), dtype=np.float64)
    )
    A = torch.from_numpy(
        np.fromfunction(lambda i,j: ((i * j + 2) % N) / N, (N, M), dtype=np.float64)
    )
    return alpha, beta, C, A, N, M

def syrk_par_spec(N, nthreads):
    return {
        'i': parir.threads(N),
        'k': parir.threads(nthreads).reduce()
    }

def syrk_run_par(nthreads):
    alpha, beta, C, A, N, M = syrk_data()

    # Run sequentially to produce a reference solution
    C_ref = C.clone()
    syrk(alpha, beta, C_ref, A, N, M, seq=True)

    # Run in parallel using the specified number of threads and validate the
    # result
    p = syrk_par_spec(N, nthreads)
    C_cu = C.clone().cuda()
    syrk(alpha, beta, C_cu, A.cuda(), N, M, parallelize=p, cache=False)

    assert torch.allclose(C_ref, C_cu.cpu()), f"Run failed using {nthreads} threads"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_syrk_parallel_64():
    syrk_run_par(64)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_syrk_parallel_128():
    syrk_run_par(128)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_syrk_parallel_256():
    syrk_run_par(256)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_syrk_parallel_512():
    syrk_run_par(512)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_syrk_parallel_1024():
    syrk_run_par(1024)
