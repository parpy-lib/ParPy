import parir
import pytest
import torch

@parir.jit
def add_slices(x, y, out, N):
    with parir.gpu:
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
    with parir.gpu:
        out[:N, :M] = x[:N, :M] + y[:N, :M]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_add_slices_2d():
    x = torch.randn(10, 20, dtype=torch.float32, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add_slices_2d(x, y, out, 10, 20)
    assert torch.allclose(x + y, out)

@parir.jit
def mul_discontinuous_2d(x, y, out, N, M):
    with parir.gpu:
        out[:N, :M] = x[:N, 0, :M] * y[0, :N, :M]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_mul_discontinuous_2d():
    x = torch.randn(10, 1, 20, dtype=torch.float32, device='cuda')
    y = torch.randn(1, 10, 20, dtype=torch.float32, device='cuda')
    out = torch.empty(10, 20, dtype=torch.float32, device='cuda')
    mul_discontinuous_2d(x, y, out, 10, 20)
    assert torch.allclose(x[:,0,:] * y[0,:,:], out)

@parir.jit
def matmul_slice(a, b, c, M, N, K):
    # TODO: Add label support for multi-dimensional slices
    with parir.gpu:
        c[:M, :N] = sum(a[:M, :K] * b[:K, :N])

@pytest.mark.skip("Reduction on slices not supported")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_matmul_slicing():
    M, N, K = 10, 15, 20
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b = torch.randn((K, N), dtype=torch.float16, device='cuda')
    c = torch.empty((M, N), dtype=torch.float16, device='cuda')
    matmul_slice(a, b, c, M, N, K)
    assert torch.allclose(a @ b, c, atol=1e-5)
