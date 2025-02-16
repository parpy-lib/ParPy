import parir
import pytest
import torch

@parir.jit
def add_inplace(x, y, M):
    parir.label("1d")
    for i in range(M):
        y[i] += x[i]

@parir.jit
def add_2d_inplace(x, y, N, M):
    parir.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)

@parir.jit
def add_2d_inplace_x2(x, y, z, w, N, M):
    parir.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)
        add_inplace(z[i], w[i], M)

@parir.jit
def add_3d_inplace(x, y, N, M, K):
    parir.label('3d')
    for i in range(N):
        add_2d_inplace(x[i], y[i], M, K)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_call():
    x = torch.randn(10, 15, device='cuda')
    y = torch.zeros_like(x)
    p = {'2d': [parir.threads(10)], '1d': [parir.threads(15)]}
    add_2d_inplace(x, y, 10, 15, parallelize=p, cache=False)
    assert torch.allclose(x, y)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_call_different_types():
    x = torch.randn(10, 15, device='cuda')
    y = torch.zeros_like(x)
    z = torch.randint(0, 10, (10, 15), dtype=torch.int32, device='cuda')
    w = torch.zeros_like(z)
    p = {'2d': [parir.threads(10)], '1d': [parir.threads(15)]}
    add_2d_inplace_x2(x, y, z, w, 10, 15, parallelize=p, cache=False)
    assert torch.allclose(x, y)
    assert torch.allclose(z, w)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_nested_call_dependency():
    x = torch.randn(10, 20, 30, device='cuda')
    y = torch.zeros_like(x)
    p = {
        '3d': [parir.threads(10)],
        '2d': [parir.threads(20)],
        '1d': [parir.threads(30)]
    }
    add_3d_inplace(x, y, 10, 20, 30, parallelize=p, cache=False)
    assert torch.allclose(x, y)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_call_non_decorated_function_fails():
    # This function is intentionally not decorated with '@parir.jit'
    def add(x, y, M):
        for i in range(M):
            y[i] += x[i]
    with pytest.raises(RuntimeError):
        @parir.jit
        def add_2d(x, y, N, M):
            parir.label('N')
            for i in range(N):
                add(x[i], y[i], M)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_recursive_call_fails():
    with pytest.raises(RuntimeError):
        @parir.jit
        def reset(x, i):
            with parir.gpu:
                if i > 0:
                    x[i] = 0.0
                    reset(x, i-1)
