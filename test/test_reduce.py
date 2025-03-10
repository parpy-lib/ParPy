import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def sum_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.sum(x[i,:])

@parir.jit
def prod_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.prod(x[i,:])

@parir.jit
def max_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.max(x[i,:])

@parir.jit
def min_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.min(x[i,:])

@parir.jit
def any_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.any(x[i,:])

@parir.jit
def all_rows(x, out, N):
    parir.label('outer')
    for i in range(N):
        parir.label('inner')
        out[i] = parir.all(x[i,:])

@parir.jit
def sum_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.sum(x[:,:], axis=1)

@parir.jit
def prod_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.prod(x[:,:], axis=1)

@parir.jit
def max_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.max(x[:,:], axis=1)

@parir.jit
def min_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.min(x[:,:], axis=1)

@parir.jit
def any_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.any(x[:,:], axis=1)

@parir.jit
def all_axis(x, out, N):
    parir.label('outer')
    parir.label('inner')
    out[:] = parir.all(x[:,:], axis=1)

@parir.jit
def sum_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.sum(x[:,:])

@parir.jit
def prod_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.prod(x[:,:])

@parir.jit
def max_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.max(x[:,:])

@parir.jit
def min_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.min(x[:,:])

@parir.jit
def any_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.any(x[:,:])

@parir.jit
def all_2d(x, out, N):
    with parir.gpu:
        parir.label('outer')
        out[0] = parir.all(x[:,:])

def reduce_wrap(reduce_fn, x, p=None):
    N, M = x.shape
    out = torch.zeros(N, dtype=x.dtype, device=x.device)
    if p is None:
        reduce_fn(x, out, N, seq=True)
    else:
        reduce_fn(x, out, N, parallelize=p, cache=False)
    return out

def compare_reduce(reduce_fn, bool_type, N, M, p):
    x = torch.randn((N, M), dtype=torch.float32)
    if bool_type:
        x = x > 0.5
    expected = reduce_wrap(reduce_fn, x)
    actual = reduce_wrap(reduce_fn, x.cuda(), p).cpu()
    if bool_type:
        assert torch.allclose(expected, actual)
    else:
        assert torch.allclose(expected, actual, atol=1e-5)

reduce_funs = [
    (sum_rows, False),
    (prod_rows, False),
    (max_rows, False),
    (min_rows, False),
    (any_rows, True),
    (all_rows, True),
    (sum_axis, False),
    (prod_axis, False),
    (max_axis, False),
    (min_axis, False),
    (any_axis, True),
    (all_axis, True),
    (sum_2d, False),
    (prod_2d, False),
    (max_2d, False),
    (min_2d, False),
    (any_2d, True),
    (all_2d, True),
]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize('fun_data', reduce_funs)
def test_reduce_outer_parallel_gpu(fun_data):
    N = 100
    M = 50
    p = {'outer': [parir.threads(N)]}
    fn, bool_type = fun_data
    compare_reduce(fn, bool_type, N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize('fun_data', reduce_funs)
def test_reduce_inner_and_outer_parallel_gpu(fun_data):
    N = 100
    M = 50
    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(128)]
    }
    fn, bool_type = fun_data
    compare_reduce(fn, bool_type, N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize('fun_data', reduce_funs)
def test_irregular_reduction(fun_data):
    # We request use of 83 threads for the innermost loop, which is not evenly
    # divisible by 32. The compiler should adjust it upward to the next number
    # divisible by 32 or warp-level intrinsics will misbehave.
    N = 100
    M = 83
    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(M)]
    }
    fn, bool_type = fun_data
    compare_reduce(fn, bool_type, N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize('fun_data', reduce_funs)
def test_multi_block_reduction(fun_data):
    N = 100
    M = 2048
    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(M)]
    }
    fn, bool_type = fun_data
    compare_reduce(fn, bool_type, N, M, p)

@pytest.mark.parametrize('fun_data', reduce_funs)
def test_reduction_compiles(fun_data):
    fn, bool_type = fun_data
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    if bool_type:
        x = x < 0.5
    out = torch.empty(N, dtype=x.dtype)
    p = {'outer': [parir.threads(N)]}
    s1 = parir.print_compiled(fn, [x, out, N], p)
    assert len(s1) != 0

    p = {
        'outer': [parir.threads(N)],
        'inner': [parir.threads(128)]
    }
    s2 = parir.print_compiled(fn, [x, out, N], p)
    assert len(s2) != 0
