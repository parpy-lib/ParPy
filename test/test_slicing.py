import parir
import pytest
import torch

@parir.jit
def add_slices(x, y, out, N):
    parir.label('N')
    out[:N] = x[:N] + y[:N]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_add_slices():
    x = torch.randn(10, dtype=torch.float32, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add_slices(x, y, out, 10, parallelize={'N': [parir.threads(10)]})
    assert torch.allclose(x + y, out)

@parir.jit
def add_slices_2d(x, y, out, N, M):
    parir.label('N')
    parir.label('M')
    out[:N, :M] = x[:N, :M] + y[:N, :M]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_add_slices_2d():
    x = torch.randn(10, 20, dtype=torch.float32, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    p = {'N': [parir.threads(10)], 'M': [parir.threads(20)]}
    add_slices_2d(x, y, out, 10, 20, parallelize=p)
    assert torch.allclose(x + y, out)

@parir.jit
def mul_discontinuous_2d(x, y, out, N, M):
    parir.label('N')
    parir.label('M')
    out[:N, :M] = x[:N, 0, :M] * y[0, :N, :M]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_mul_discontinuous_2d():
    x = torch.randn(10, 1, 20, dtype=torch.float32, device='cuda')
    y = torch.randn(1, 10, 20, dtype=torch.float32, device='cuda')
    out = torch.empty(10, 20, dtype=torch.float32, device='cuda')
    p = {'N': [parir.threads(10)], 'M': [parir.threads(20)]}
    mul_discontinuous_2d(x, y, out, 10, 20, parallelize=p)
    assert torch.allclose(x[:,0,:] * y[0,:,:], out)

@parir.jit
def matmul_slice(a, b, c, M, N, K):
    parir.label('N')
    for i in range(N):
        parir.label('M')
        for j in range(M):
            parir.label('K')
            c[i,j] = sum(a[i,:K] * b[:K,j])

@pytest.mark.skip("Reduction on slices not supported")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_matmul_slicing():
    M, N, K = 10, 15, 20
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    c = torch.empty((M, N), dtype=torch.float16, device='cuda')
    p = {
        'N': [parir.threads(N)], 'M': [parir.threads(M)], 'K': [parir.threads(K), parir.reduce()]
    }
    matmul_slice(a, b, c, M, N, K, parallelize=p)
    assert torch.allclose(a @ b, c, atol=1e-5)

def test_slice_materialization_fail():
    @parir.jit
    def slice_materialize(x, N):
        with parir.gpu:
            y = x[1:N]
    x = torch.randn(10, device='cuda')
    with pytest.raises(RuntimeError):
        slice_materialize(x, 10)
