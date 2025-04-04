import parir
import pytest
import torch

@parir.jit
def sum_rows(x, N, out):
    parir.label('N')
    for i in range(N):
        parir.label('M')
        out[i] = parir.sum(x[i,:])

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_reduce_multi_block():
    N, M = 10, 20
    x = torch.randn(N, M, dtype=torch.float32, device='cuda')

    # Parallelize reduction within a single block (n = 1024)
    out1 = torch.empty(N, dtype=x.dtype, device=x.device)
    p1 = {'N': parir.threads(N), 'M': parir.threads(1024)}
    sum_rows(x, N, out1, parallelize=p1, cache=False)

    # Parallelize reduction across multiple blocks (n > 1024)
    out2 = torch.empty_like(out1)
    p2 = {'N': parir.threads(N), 'M': parir.threads(2048)}
    sum_rows(x, N, out2, parallelize=p2, cache=False)

    assert torch.allclose(out1, out2, atol=1e-6)

@parir.jit
def normalize(x, N):
    parir.label('N')
    for i in range(N):
        parir.label('M_1')
        s = parir.sum(x[i,:])
        parir.label('M_2')
        x[i,:] /= s

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_varying_parallelism():
    N, M = 10, 20
    x = torch.randn(N, M, dtype=torch.float32, device='cuda')
    p = {
        'N': parir.threads(N),
        'M_1': parir.threads(128),
        'M_2': parir.threads(64)
    }
    normalize(x, N, parallelize=p, cache=False)
    assert torch.allclose(torch.sum(x, axis=-1), torch.ones(N))

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

def sum_exp_3d_wrap(par):
    N, M, K = 10, 20, 30
    x = torch.randn(N, M, K, dtype=torch.float32, device='cuda')
    x_2 = x.detach().clone()
    out = torch.empty(N, M, dtype=x.dtype, device=x.device)
    sum_exp_3d(x, N, M, out, parallelize=par, cache=False)
    ref_out = torch.empty_like(out)
    sum_exp_3d(x_2, N, M, ref_out, seq=True)
    assert torch.allclose(out, ref_out, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_nested_imbalanced_parallelism():
    p = {
        'N': parir.threads(10),
        'M': parir.threads(20),
        'K_1': parir.threads(32),
        'K_2': parir.threads(64),
    }
    sum_exp_3d_wrap(p)

@pytest.mark.skip("Compiler does not currently support this kind of imbalanced parallelism")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_imbalanced_parallelism_in_sequential_for():
    p = {
        'N': parir.threads(10),
        'K_1': parir.threads(32),
        'K_2': parir.threads(64),
    }
    sum_exp_3d_wrap(p)
