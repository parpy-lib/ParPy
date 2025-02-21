import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def store_gt(x, y, out, N):
    parir.label('i')
    for i in range(N):
        out[i] = x[i] < y[i]

@parir.jit
def reduce_and(x, out):
    with parir.gpu:
        out[0] = out[0] and x[0]

def bool_test_data():
    N = 100
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return x, y, N

def bool_wrap(x, y, p=None):
    N, = x.shape
    tmp = torch.empty(N, dtype=torch.bool, device=x.device)
    if p is None:
        store_gt(x, y, tmp, N, seq=True)
    else:
        store_gt(x, y, tmp, N, parallelize=p, cache=False)
    out = torch.empty(1, dtype=torch.bool, device=x.device)
    if p is None:
        reduce_and(tmp, out, seq=True)
    else:
        reduce_and(tmp, out)
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_bool_gpu():
    x, y, N = bool_test_data()
    expected = bool_wrap(x, y)

    p = {'i': [parir.threads(N)]}
    actual = bool_wrap(x.cuda(), y.cuda(), p).cpu()
    assert torch.allclose(expected, actual, atol=1e-5)

def test_bool_compiles():
    x, y, N = bool_test_data()
    tmp = torch.empty_like(x, dtype=torch.bool)
    p = {'i': [parir.threads(64)]}
    s = parir.print_compiled(store_gt, [x, y, tmp, N], p)
    assert len(s) != 0

    res = torch.empty(1, dtype=torch.bool)
    s = parir.print_compiled(reduce_and, [tmp, res])
    assert len(s) != 0
