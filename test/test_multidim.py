import parir
from parir import ParKind
import pytest
import torch

@parir.jit
def copy_2d(dst, src, N, M):
    for i in range(N):
        for j in range(M):
            dst[i,j] = src[i,j]

@parir.jit
def copy_3d(dst, src, N, M, K):
    for i in range(N):
        for j in range(M):
            for k in range(K):
                dst[i,j,k] = src[i,j,k]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_two_dims():
    N, M = 20, 30
    x = torch.randn((N, M), dtype=torch.float32)
    y = torch.empty_like(x)
    copy_2d(y, x, N, M)

    x_cu = x.cuda()
    y_cu = torch.empty_like(x_cu)
    p = {
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(M)]
    }
    copy_2d(y_cu, x_cu, N, M, parallelize=p)

    assert torch.allclose(y, y_cu.cpu())

@pytest.mark.skip("Indexing is currently broken when using more than two dimensions")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_three_dims():
    N, M, K = 20, 30, 40
    x = torch.randn((N, M, K), dtype=torch.float32)
    y = torch.empty_like(x)
    copy_3d(y, x, N, M, K)

    x_cu = x.cuda()
    y_cu = torch.empty_like(x_cu)
    p = {
        'i': [ParKind.GpuThreads(N)],
        'j': [ParKind.GpuThreads(M)],
        'k': [ParKind.GpuThreads(K)]
    }
    copy_3d(y_cu, x_cu, N, M, K, parallelize=p)

    assert torch.allclose(y, y_cu.cpu())
