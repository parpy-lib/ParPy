import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def upper_bound_range(x, N):
    for i in range(N):
        x[i] = i

@parir.jit
def no_step_range(x, N):
    for i in range(1, N):
        x[i] = i

@parir.jit
def step_range(x, N):
    for i in range(1, N, 2):
        x[i] = i

@parir.jit
def negative_step_range(x, N):
    for i in range(N-1, -1, -1):
        x[i] = i

def range_helper(fn, compile_only):
    N = 100
    x = torch.zeros((N,), dtype=torch.int64)
    p = {'i': [ParKind.GpuThreads(32)]}
    if compile_only:
        s = parir.print_compiled(fn, [x, N], p)
        assert len(s) != 0
    else:
        x_cu = x.detach().clone().cuda()
        upper_bound_range(x_cu, N, parallelize=p, cache=False)
        upper_bound_range(x, N)
        assert torch.allclose(x, x_cu.cpu())

range_funs = [upper_bound_range, no_step_range, step_range, negative_step_range]

@pytest.mark.parametrize('fn', range_funs)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_range_gpu(fn):
    range_helper(fn, False)

@pytest.mark.parametrize('fn', range_funs)
def test_range_compile(fn):
    range_helper(fn, True)

def test_zero_step_fails():
    with pytest.raises(RuntimeError):
        @parir.jit
        def zero_step(x, N):
            for i in range(0, N, 0):
                x[i] = i
