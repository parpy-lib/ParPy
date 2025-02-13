import numpy as np
import parir
import pytest
import torch

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def softmax(x, N, M, out):
    # We have N independent instances we want to do softmax on
    parir.label('i')
    for i in range(N):
        m = parir.float32(-parir.inf)
        parir.label('j1')
        for j in range(M):
            m = max(m, x[i, j])

        parir.label('j2')
        for j in range(M):
            out[i, j] = parir.exp(x[i, j] - m)

        s = parir.float32(0.0)
        parir.label('j3')
        for j in range(M):
            s = s + out[i, j]

        parir.label('j4')
        for j in range(M):
            out[i, j] = out[i, j] / s


def softmax_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty_like(x)
    if p is None:
        softmax(x, N, M, out, seq=True)
    else:
        softmax(x, N, M, out, parallelize=p, cache=False)
    return out

def compare_softmax(p):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32, device='cuda')
    y1 = torch.softmax(x, 1)
    y2 = softmax_wrap(x, p)
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_seq_reduce():
    p = { "i" : [parir.threads(256)] }
    compare_softmax(p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_gpu():
    p = {
        "i" : [parir.threads(256)],
        "j1": [parir.threads(128), parir.reduce()],
        "j2": [parir.threads(128)],
        "j3": [parir.threads(128), parir.reduce()],
        "j4": [parir.threads(128)]
    }
    compare_softmax(p)

def test_softmax_compiles():
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty_like(x)
    p = {
        "i" : [parir.threads(256)],
        "j1": [parir.threads(128), parir.reduce()],
        "j2": [parir.threads(128)],
        "j3": [parir.threads(128), parir.reduce()],
        "j4": [parir.threads(128)]
    }
    s = parir.print_compiled(softmax, [x, N, M, out], p)
    assert len(s) != 0
