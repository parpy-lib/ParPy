import numpy as np
import parir
import pytest
import torch

from common import *

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def softmax(x, N, M, out):
    # We have N independent instances we want to do softmax on
    parir.label('N')
    for i in range(N):
        parir.label('M')
        m = parir.max(x[i,:])

        parir.label('M')
        out[i,:] = parir.exp(x[i,:] - m)

        parir.label('M')
        s = parir.sum(out[i,:])

        parir.label('M')
        out[i,:] = out[i,:] / s


def softmax_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty_like(x)
    if p is None:
        softmax(x, N, M, out, opts=seq_opts())
    else:
        softmax(x, N, M, out, opts=par_opts(p))
    return out

def compare_softmax(p):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32, device='cuda')
    y1 = torch.softmax(x, 1)
    y2 = softmax_wrap(x, p)
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_seq_reduce():
    p = { "N" : parir.threads(256) }
    compare_softmax(p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_gpu():
    p = {
        "N" : parir.threads(256),
        "M": parir.threads(128),
    }
    compare_softmax(p)

def test_softmax_compiles():
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty_like(x)
    p = {
        "N" : parir.threads(256),
        "M": parir.threads(128),
    }
    s = parir.print_compiled(softmax, [x, N, M, out], par_opts(p))
    assert len(s) != 0
