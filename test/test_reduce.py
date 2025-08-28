import parpy
import pytest
import re
import subprocess
import tempfile
import torch

from common import *

torch.manual_seed(1234)

@parpy.jit
def sum_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.operators.sum(x[i,:])

@parpy.jit
def prod_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.operators.prod(x[i,:])

@parpy.jit
def max_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.operators.max(x[i,:])

@parpy.jit
def min_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.operators.min(x[i,:])

@parpy.jit
def sum_axis(x, out, N):
    parpy.label('outer')
    parpy.label('inner')
    out[:] = parpy.operators.sum(x[:,:], axis=1)

@parpy.jit
def prod_axis(x, out, N):
    parpy.label('outer')
    parpy.label('inner')
    out[:] = parpy.operators.prod(x[:,:], axis=1)

@parpy.jit
def max_axis(x, out, N):
    parpy.label('outer')
    parpy.label('inner')
    out[:] = parpy.operators.max(x[:,:], axis=1)

@parpy.jit
def min_axis(x, out, N):
    parpy.label('outer')
    parpy.label('inner')
    out[:] = parpy.operators.min(x[:,:], axis=1)

@parpy.jit
def sum_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.operators.sum(x[:,:])

@parpy.jit
def prod_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.operators.prod(x[:,:])

@parpy.jit
def max_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.operators.max(x[:,:])

@parpy.jit
def min_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.operators.min(x[:,:])

def reduce_wrap(reduce_fn, x, opts):
    N, M = x.shape
    out = torch.zeros(N, dtype=x.dtype)
    reduce_fn(x, out, N, opts=opts)
    return out

def compare_reduce(reduce_fn, N, M, opts):
    x = torch.randn((N, M), dtype=torch.float32)
    opts.seq = True
    expected = reduce_wrap(reduce_fn, x, opts)
    opts.seq = False
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
        p = {'outer': parpy.threads(N)}
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_inner_and_outer_parallel_gpu(fn, backend):
    def helper():
        N = 100
        M = 50
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(128)
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
            'outer': parpy.threads(N),
            'inner': parpy.threads(M)
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
            'outer': parpy.threads(N),
            'inner': parpy.threads(M).tpb(512)
        }
        compare_reduce(fn, N, M, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_clustered_reduction(fn, backend):
    def helper():
        N = 100
        M = 2048
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(M).tpb(512)
        }
        opts = par_opts(backend, p)
        opts.use_cuda_thread_block_clusters = True
        compare_reduce(fn, N, M, opts)
    run_if_clusters_are_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_extended_clustered_reduction(fn, backend):
    def helper():
        N = 100
        M = 8192
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(M).tpb(512)
        }
        opts = par_opts(backend, p)
        opts.use_cuda_thread_block_clusters = True
        opts.max_thread_blocks_per_cluster = 16
        compare_reduce(fn, N, M, opts)
    run_if_clusters_are_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduction_codegen(fn, backend):
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.zeros(N, dtype=x.dtype)
    p = {'outer': parpy.threads(N)}
    s1 = parpy.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parpy.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 1, 1\), dim3\(128, 1, 1\).*>>>\(.*\);"
        elif backend == parpy.CompileBackend.Metal:
            pat = r"parpy_metal::launch_kernel\(.*1, 1, 1, 128, 1, 1\).*"
        else:
            pat = ""
        assert re.search(pat, s1, re.DOTALL) is not None
    else:
        assert len(s1) != 0

    p = {
        'outer': parpy.threads(N),
        'inner': parpy.threads(128)
    }
    s2 = parpy.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parpy.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 100, 1\), dim3\(128, 1, 1\).*>>>\(.*\);"
        elif backend == parpy.CompileBackend.Metal:
            pat = r"parpy_metal::launch_kernel\(.*1, 100, 1, 128, 1, 1\).*"
        else:
            pat = ""
        assert re.search(pat, s2, re.DOTALL) is not None
    else:
        assert len(s2) != 0

    p = {
        'outer': parpy.threads(N),
        'inner': parpy.threads(1024).tpb(128)
    }
    s3 = parpy.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parpy.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 8, 100\), dim3\(128, 1, 1\).*>>>\(.*\);"
        elif backend == parpy.CompileBackend.Metal:
            pat = r"parpy_metal::launch_kernel\(.*1, 8, 100, 128, 1, 1\).*"
        else:
            pat = ""
        assert re.search(pat, s3, re.DOTALL) is not None
    else:
        assert len(s3) != 0

@pytest.mark.parametrize('fn', reduce_funs)
def test_clustered_reduction_codegen_in_cuda(fn):
    N = 100
    M = 50
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.zeros(N, dtype=x.dtype)
    p = {
        'outer': parpy.threads(N),
        'inner': parpy.threads(4096).tpb(512)
    }
    opts = par_opts(parpy.CompileBackend.Cuda, p)
    opts.use_cuda_thread_block_clusters = True
    s = parpy.print_compiled(fn, [x, out, N], opts)
    if not fn in multi_dim_reduce_funs:
        pat = r".*<<<dim3\(8, 100, 1\), dim3\(512, 1, 1\).*>>>\(.*\);"
        assert re.search(pat, s, re.DOTALL) is not None
        pat = r".*__cluster_dims__\(8, 1, 1\).*"
        assert re.search(pat, s, re.DOTALL) is not None
        # This attribute should only be inserted when we use more than 8 thread
        # blocks per cluster.
        pat = r".*cudaFuncSetAttribute.*"
        assert re.search(pat, s, re.DOTALL) is None
    else:
        assert len(s) != 0

    p = {
        'outer': parpy.threads(N),
        'inner': parpy.threads(4096).tpb(256)
    }
    opts.parallelize = p
    s = parpy.print_compiled(fn, [x, out, N], opts)
    if not fn in multi_dim_reduce_funs:
        # In this situation, where the kernel has 16 blocks, the compiler will
        # not use clusters unless the user explicitly sets the maximum number
        # of thread blocks (see the next example).
        pat = r".*__cluster_dims__\(16, 1, 1\).*"
        assert re.search(pat, s, re.DOTALL) is None
    else:
        assert len(s) != 0

    p = {
        'outer': parpy.threads(N),
        'inner': parpy.threads(4096).tpb(256)
    }
    opts.parallelize = p
    opts.max_thread_blocks_per_cluster = 16
    s = parpy.print_compiled(fn, [x, out, N], opts)
    if not fn in multi_dim_reduce_funs:
        pat = r".*<<<dim3\(16, 100, 1\), dim3\(256, 1, 1\).*>>>\(.*\);"
        assert re.search(pat, s, re.DOTALL) is not None
        pat = r".*__cluster_dims__\(16, 1, 1\).*"
        assert re.search(pat, s, re.DOTALL) is not None
        pat = r".*cudaFuncSetAttribute.*"
        assert re.search(pat, s, re.DOTALL) is not None
    else:
        assert len(s) != 0

@pytest.mark.parametrize('fn', reduce_funs)
def test_clustered_reduction_compiles_in_cuda(fn):
    def helper():
        N = 100
        M = 50
        x = torch.randn((N, M), dtype=torch.float32)
        out = torch.zeros(N, dtype=x.dtype)
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(4096).tpb(512)
        }
        opts = par_opts(parpy.CompileBackend.Cuda, p)
        opts.use_cuda_thread_block_clusters = True
        fn(x, out, N, opts=opts)
    run_if_clusters_are_enabled(parpy.CompileBackend.Cuda, helper)

# Tests using a custom step size.
@parpy.jit
def odd_entries_sum(x, y, N, M):
    parpy.label('N')
    for i in range(N):
        y[i] = 0.0
        parpy.label('M')
        for j in range(1, M, 2):
            y[i] += x[i, j]

def odd_entries_wrap(backend, p):
    N = 10
    M = 4096
    x = torch.randn((N, M), dtype=torch.float32)
    out = torch.zeros(N, dtype=x.dtype)
    odd_entries_sum(x, out, N, M, opts=par_opts(backend, p))
    out_seq = torch.zeros_like(out)
    odd_entries_sum(x, out_seq, N, M, opts=seq_opts(backend))
    assert torch.allclose(out, out_seq, atol=1e-4)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_single_block(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'M': parpy.threads(32).reduce()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_multiblock(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'M': parpy.threads(2048).reduce()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)
