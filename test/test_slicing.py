import parir
import pytest
import torch

@parir.jit
def add_slices(x, y, out):
    parir.label('N')
    out[:] = x[:] + y[:]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_add_slices():
    x = torch.randn(10, dtype=torch.float32, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add_slices(x, y, out, parallelize={'N': [parir.threads(10)]}, cache=False)
    assert torch.allclose(x + y, out)

@parir.jit
def add_slices_2d(x, y, out):
    parir.label('N')
    parir.label('M')
    out[:, :] = x[:, :] + y[:, :]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_add_slices_2d():
    x = torch.randn(10, 20, dtype=torch.float32, device='cuda')
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    p = {'N': [parir.threads(10)], 'M': [parir.threads(20)]}
    add_slices_2d(x, y, out, parallelize=p, cache=False)
    assert torch.allclose(x + y, out)

@parir.jit
def mul_discontinuous_2d(x, y, out):
    parir.label('N')
    parir.label('M')
    out[:, :] = x[:, 0, :] * y[0, :, :]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_parir_mul_discontinuous_2d():
    x = torch.randn(10, 1, 20, dtype=torch.float32, device='cuda')
    y = torch.randn(1, 10, 20, dtype=torch.float32, device='cuda')
    out = torch.empty(10, 20, dtype=torch.float32, device='cuda')
    p = {'N': [parir.threads(10)], 'M': [parir.threads(20)]}
    mul_discontinuous_2d(x, y, out, parallelize=p, cache=False)
    assert torch.allclose(x[:,0,:] * y[0,:,:], out)

@parir.jit
def matmul_slice(a, b, c, M, N):
    for i in range(M):
        parir.label('N')
        for j in range(N):
            parir.label('K')
            c[i,j] = parir.sum(a[i,:] * b[:,j], axis=0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_matmul_slicing():
    M, N, K = 10, 15, 20
    a = torch.randn((M, K), dtype=torch.float64, device='cuda')
    b = torch.randn((K, N), dtype=torch.float64, device='cuda')
    c = torch.empty((M, N), dtype=torch.float64, device='cuda')
    p = {
        'N': [parir.threads(N)], 'M': [parir.threads(M)],
        'K': [parir.threads(K), parir.reduce()]
    }
    matmul_slice(a, b, c, M, N, parallelize=p, cache=False)
    assert torch.allclose(a @ b, c, atol=1e-5)

def test_slice_assign_fail():
    @parir.jit
    def slice_materialize(x):
        with parir.gpu:
            y = x[1:]
    x = torch.randn(10, device='cuda')
    with pytest.raises(RuntimeError) as e_info:
        slice_materialize(x)
    assert e_info.match(r"Slice expression cannot be assigned to fresh variable y.*")

@parir.jit
def jacobi_1d(A, B, nsteps):
    for t in range(1, nsteps):
        parir.label('N')
        B[1:-1] = (A[:-2] + A[1:-1] + A[2:]) / 3.0
        parir.label('N')
        A[1:-1] = (B[:-2] + B[1:-1] + B[2:]) / 3.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_slice_offset():
    N = 10
    A = torch.randn((N,), dtype=torch.float64, device='cuda')
    B = torch.randn_like(A)

    A_seq = A.detach().clone()
    B_seq = B.detach().clone()
    jacobi_1d(A_seq, B_seq, N, seq=True)

    p = {'N': [parir.threads(N)]}
    jacobi_1d(A, B, N, cache=False, parallelize=p)

    assert torch.allclose(A, A_seq, atol=1e-5)
    assert torch.allclose(B, B_seq, atol=1e-5)
