import parir
import pytest
import re
import torch

from common import *

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

def reduce_wrap(reduce_fn, x, opts):
    N, M = x.shape
    out = torch.zeros(N, dtype=x.dtype)
    reduce_fn(x, out, N, opts=opts)
    return out

def compare_reduce(reduce_fn, N, M, opts):
    x = torch.randn((N, M), dtype=torch.float32)
    expected = reduce_wrap(reduce_fn, x, opts)
    actual = reduce_wrap(reduce_fn, x, opts)
    assert torch.allclose(expected, actual, atol=1e-4), f"{expected}\n{actual}"

reduce_funs = [
    sum_rows,
    prod_rows,
    max_rows,
    min_rows,
    sum_axis,
    prod_axis,
    max_axis,
    min_axis,
    sum_2d,
    prod_2d,
    max_2d,
    min_2d,
]
multi_dim_reduce_funs = set([sum_2d, prod_2d, max_2d, min_2d])

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_outer_parallel_gpu(fn, backend):
    def helper():
        N = 100
        M = 50
        p = {'outer': parir.threads(N)}
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_inner_and_outer_parallel_gpu(fn, backend):
    def helper():
        N = 100
        M = 50
        p = {
            'outer': parir.threads(N),
            'inner': parir.threads(128)
        }
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_irregular_reduction(fn, backend):
    # We request use of 83 threads for the innermost loop, which is not evenly
    # divisible by 32. The compiler should adjust it upward to the next number
    # divisible by 32 or warp-level intrinsics will misbehave.
    def helper():
        N = 100
        M = 83
        p = {
            'outer': parir.threads(N),
            'inner': parir.threads(M)
        }
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_multi_block_reduction(fn, backend):
    # Request more than 1024 threads, so that the compiler generates the
    # multi-block reduction approach. In addition, we request the number of
    # threads per block as 512.
    def helper():
        N = 100
        M = 2048
        p = {
            'outer': parir.threads(N),
            'inner': parir.threads(M).tpb(512)
        }
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduction_compiles(fn, backend):
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty(N, dtype=x.dtype)
    p = {'outer': parir.threads(N)}
    s1 = parir.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parir.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 1, 1\), dim3\(128, 1, 1\)>>>\(.*\);"
        elif backend == parir.CompileBackend.Metal:
            pat = r"parir_metal::launch_kernel\(.*1, 1, 1, 128, 1, 1\);"
        else:
            pat = ""
        assert re.search(pat, s1, re.DOTALL) is not None
    else:
        assert len(s1) != 0

    p = {
        'outer': parir.threads(N),
        'inner': parir.threads(128)
    }
    s2 = parir.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parir.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 100, 1\), dim3\(128, 1, 1\)>>>\(.*\);"
        elif backend == parir.CompileBackend.Metal:
            pat = r"parir_metal::launch_kernel\(.*1, 100, 1, 128, 1, 1\);"
        else:
            pat = ""
        assert re.search(pat, s2, re.DOTALL) is not None
    else:
        assert len(s2) != 0

    p = {
        'outer': parir.threads(N),
        'inner': parir.threads(1024).tpb(128)
    }
    s3 = parir.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parir.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 8, 100\), dim3\(128, 1, 1\)>>>\(.*\);"
        elif backend == parir.CompileBackend.Metal:
            pat = r"parir_metal::launch_kernel\(.*1, 8, 100, 128, 1, 1\);"
        else:
            pat = ""
        assert re.search(pat, s3, re.DOTALL) is not None
    else:
        assert len(s3) != 0

# Tests using a custom step size.
@parir.jit
def odd_entries_sum(x, y, N, M):
    parir.label('N')
    for i in range(N):
        y[i] = 0.0
        parir.label('M')
        for j in range(1, M, 2):
            y[i] += x[i, j]

def odd_entries_wrap(backend, p):
    N = 10
    M = 4096
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.empty(N, dtype=x.dtype)
    odd_entries_sum(x, out, N, M, opts=par_opts(backend, p))
    out_seq = torch.empty_like(out)
    odd_entries_sum(x, out_seq, N, M, opts=seq_opts(backend))
    assert torch.allclose(out, out_seq, atol=1e-4)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_single_block(backend):
    def helper():
        p = {
            'N': parir.threads(10),
            'M': parir.threads(32).reduce()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_multiblock(backend):
    def helper():
        p = {
            'N': parir.threads(10),
            'M': parir.threads(2048).reduce()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)
