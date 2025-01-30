import numpy as np
import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def softmax(x, N, M, out):
    # We have N independent instances we want to do softmax on
    for i in range(N):
        m = parir.float32(-parir.inf)
        for j1 in range(M):
            m = max(m, x[i, j1])

        for j2 in range(M):
            out[i, j2] = parir.exp(x[i, j2] - m)

        s = parir.float32(0.0)
        for j3 in range(M):
            s = s + out[i, j3]

        for j4 in range(M):
            out[i, j4] = out[i, j4] / s


def softmax_wrap(x, p=None):
    N, M = x.shape
    out = torch.empty_like(x)
    softmax(x, N, M, out, parallelize=p)
    return out

def compare_softmax(p):
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32, device='cuda')
    y1 = torch.softmax(x, 1)
    y2 = softmax_wrap(x, p)
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_seq_reduce():
    p = { "i" : [ParKind.GpuThreads(256)] }
    compare_softmax(p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_softmax_gpu():
    p = {
        "i" : [ParKind.GpuThreads(256)],
        "j1": [ParKind.GpuThreads(128), ParKind.GpuReduction()],
        "j2": [ParKind.GpuThreads(128)],
        "j3": [ParKind.GpuThreads(128), ParKind.GpuReduction()],
        "j4": [ParKind.GpuThreads(128)]
    }
    compare_softmax(p)

def test_softmax_compiles():
    N, M = 256, 512
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty_like(x)
    p = {
        "i" : [ParKind.GpuThreads(256)],
        "j1": [ParKind.GpuThreads(128), ParKind.GpuReduction()],
        "j2": [ParKind.GpuThreads(128)],
        "j3": [ParKind.GpuThreads(128), ParKind.GpuReduction()],
        "j4": [ParKind.GpuThreads(128)]
    }
    s = parir.print_compiled(softmax, [x, N, M, out], p)
    assert len(s) != 0
