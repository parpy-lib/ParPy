import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def copy(dst, src, N, M):
    for i in range(N):
        for j in range(M):
            dst[i,j] = src[i,j]

def generate_copy_test_data(device):
    N, M = 100, 100
    x = torch.randn((N, M), dtype=torch.float64, device=device)
    y = torch.empty_like(x)
    p = {'i': [ParKind.GpuThreads(N)], 'j': [ParKind.GpuThreads(M)]}
    return y, x, N, M, p

def test_copy_print_compiled():
    y, x, N, M, p = generate_copy_test_data('cpu')
    s = parir.print_compiled(copy, [y, x, N, M], p)
    assert len(s) != 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_run_compiled_string():
    y, x, N, M, p = generate_copy_test_data('cuda')
    s = parir.print_compiled(copy, [y, x, N, M], p)
    fn = parir.compile_string(copy.__name__, s, cache=False)
    fn(y, x, N, M)
    assert torch.allclose(x, y)
