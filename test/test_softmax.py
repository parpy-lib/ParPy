import numpy as np
import prickle
import pytest
import torch

from common import *

torch.manual_seed(1234)
np.random.seed(1234)

@prickle.jit
def softmax(x, N, M, out):
    # We have N independent instances we want to do softmax on
    prickle.label('N')
    for i in range(N):
        prickle.label('M')
        m = prickle.max(x[i,:])

        prickle.label('M')
        out[i,:] = prickle.exp(x[i,:] - m)

        prickle.label('M')
        s = prickle.sum(out[i,:])

        prickle.label('M')
        out[i,:] = out[i,:] / s


def softmax_wrap(x, opts):
    N, M = x.shape
    out = torch.zeros_like(x)
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
    p = { "N" : prickle.threads(256) }
    run_if_backend_is_enabled(backend, lambda: compare_softmax(par_opts(backend, p)))

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax_gpu(backend):
    p = {
        "N" : prickle.threads(256),
        "M": prickle.threads(128),
    }
    run_if_backend_is_enabled(backend, lambda: compare_softmax(par_opts(backend, p)))

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax_compiles(backend):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.zeros_like(x)
    p = {
        "N" : prickle.threads(256),
        "M": prickle.threads(128),
    }
    s = prickle.print_compiled(softmax, [x, N, M, out], par_opts(backend, p))
    assert len(s) != 0
