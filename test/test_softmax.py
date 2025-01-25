import numpy as np
import parir
from parir import ParKind, ParSpec
import pytest
import torch

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def softmax(x, N, M, out):
    # We have N independent instances we want to do softmax on
    for i in range(N):
        m = -parir.inf
        for j in range(M):
            m = max(m, x[i, j])

        for j in range(M):
            out[i, j] = x[i, j] - m

        s = 0.0
        for j in range(M):
            s = s + out[i, j]

        for j in range(M):
            out[i, j] = out[i, j] / s


def softmax_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty_like(x)
    softmax(x, N, M, out, parallelize=p, cache=False)
    return out

def compare_softmax(p):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32, device='cuda')
    y1 = torch.nn.softmax(x, dim=-1)
    y2 = softmax_wrap(x, p)
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skip(reason="Uses unsupported language features")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_seq_reduce():
    p = { "i" : ParSpec(ParKind.GpuThreads(N)) }
    compare_softmax(p)

@pytest.mark.skip(reason="Uses unsupported language features")
@pytest.mark.skip(reason="Parallel reductions are not supported")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_gpu():
    p = {
        "i" : ParSpec(ParKind.GpuThreads(N)),
        "j" : ParSpec(ParKind.GpuThreads(128))
    }
    compare_softmax(p)
