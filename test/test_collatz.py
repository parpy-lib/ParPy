import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def collatz(out, N):
    for i in range(1, N+1):
        v = i
        count = 0
        while v > 1:
            if v % 2 == 0:
                v = v / 2
            else:
                v = 3 * v + 1
            count = count + 1
        out[i] = count

def collatz_wrap(N, device='cpu', p=None):
    out = torch.zeros(N+1, dtype=torch.int32, device=device)
    collatz(out, N, parallelize=p, cache=False)
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_collatz_gpu():
    N = 1000
    expected = collatz_wrap(N)
    p = {'i': [ParKind.GpuThreads(256)]}
    actual = collatz_wrap(N, 'cuda', p=p)
    assert torch.allclose(expected, actual.cpu())

def test_collatz_compiles_with_parallelism():
    N = 1000
    out = torch.zeros(N+1, dtype=torch.int32)
    p = {'i': [ParKind.GpuThreads(128)]}
    s = parir.print_compiled(collatz, [out, N], p)
    assert len(s) != 0
