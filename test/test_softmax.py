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


def softmax_wrap(x, opts):
    N, M = x.shape
    out = torch.empty_like(x)
    softmax(x, N, M, out, opts=opts)
    return out

def compare_softmax(opts):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    y1 = torch.softmax(x, 1)
    y2 = softmax_wrap(x, opts)
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax_seq_reduce(backend):
    p = { "N" : parir.threads(256) }
    run_if_backend_is_enabled(backend, lambda: compare_softmax(par_opts(backend, p)))

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax_gpu(backend):
    p = {
        "N" : parir.threads(256),
        "M": parir.threads(128),
    }
    run_if_backend_is_enabled(backend, lambda: compare_softmax(par_opts(backend, p)))

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax_compiles(backend):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty_like(x)
    p = {
        "N" : parir.threads(256),
        "M": parir.threads(128),
    }
    s = parir.print_compiled(softmax, [x, N, M, out], par_opts(backend, p))
    assert len(s) != 0
