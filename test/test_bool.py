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
def reduce_and(x, out, N):
    parir.label('j')
    for j in range(N):
        out[0] = out[0] and x[j]

def bool_test_data():
    N = 100
    x = torch.randn(N, dtype=torch.float32)
    y = torch.randn(N, dtype=torch.float32)
    return x, y, N

def bool_wrap(x, y, N, p=None):
    tmp = torch.empty(N, dtype=torch.bool, device=x.device)
    store_gt(x, y, tmp, N, parallelize=p, cache=False)
    out = torch.empty(1, dtype=torch.bool, device=x.device)
    reduce_and(tmp, out, N)
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_bool_gpu():
    x, y, N = bool_test_data()
    expected = bool_wrap(x, y, N)

    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(64), parir.reduce()]
    }
    actual = bool_wrap(x.cuda(), y.cuda(), N, p).cpu()
    assert torch.allclose(expected, actual, atol=1e-5)

def test_bool_compiles():
    x, y, N = bool_test_data()
    tmp = torch.empty_like(x, dtype=torch.bool)
    p = {'i': [parir.threads(64)]}
    s = parir.print_compiled(store_gt, [x, y, tmp, N], p)
    assert len(s) != 0

    p = {'j': [parir.threads(64), parir.reduce()]}
    res = torch.empty(1, dtype=torch.bool)
    s = parir.print_compiled(reduce_and, [tmp, res, N], p)
    assert len(s) != 0
