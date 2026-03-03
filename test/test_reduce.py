import numpy as np
import parpy
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

def sum_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.reduce.sum(x[i,:])

def prod_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.reduce.prod(x[i,:])

def max_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.reduce.max(x[i,:])

def min_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        parpy.label('inner')
        out[i] = parpy.reduce.min(x[i,:])

def sum_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.reduce.sum(x[:,:])

def prod_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.reduce.prod(x[:,:])

def max_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.reduce.max(x[:,:])

def min_2d(x, out, N):
    parpy.label('outer')
    out[0] = parpy.reduce.min(x[:,:])

def reduce_wrap(reduce_fn, x, opts=None):
    N, M = x.shape
    out = np.zeros(N, dtype=np.float32)
    if opts is None:
        reduce_fn(x, out, N)
    else:
        parpy.jit(reduce_fn)(x, out, N, opts=opts)
    return out

def compare_reduce(reduce_fn, N, M, ty, opts):
    if opts.backend == parpy.CompileBackend.Metal and ty == np.float64:
        pytest.skip("64-bit floats are not supported in Metal")
    x = np.random.randn(N, M).astype(ty)
    expected = reduce_wrap(reduce_fn, x)
    actual = reduce_wrap(reduce_fn, x, opts)
    atol = 0.05 if ty == np.float16 else 1e-4
    rtol = 1e-2 if ty == np.float16 else 1e-4
    assert np.allclose(expected, actual, atol=atol, rtol=rtol), f"{expected}\n{actual}"

reduce_funs = [
    sum_rows,
    prod_rows,
    max_rows,
    min_rows,
    sum_2d,
    prod_2d,
    max_2d,
    min_2d,
]
multi_dim_reduce_funs = set([sum_2d, prod_2d, max_2d, min_2d])

dtypes = [np.float16, np.float32, np.float64]

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_reduce_outer_parallel_gpu(fn, backend, ty):
    def helper():
        N = 100
        M = 50
        p = {'outer': parpy.threads(N)}
        compare_reduce(fn, N, M, ty, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_reduce_inner_and_outer_parallel_gpu(fn, backend, ty):
    def helper():
        N = 100
        M = 50
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(128)
        }
        compare_reduce(fn, N, M, ty, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_irregular_reduction(fn, backend, ty):
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
        compare_reduce(fn, N, M, ty, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_multi_block_reduction(fn, backend, ty):
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
        compare_reduce(fn, N, M, ty, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_clustered_reduction(fn, backend, ty):
    def helper():
        N = 100
        M = 2048
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(M).tpb(512)
        }
        opts = par_opts(backend, p)
        opts.use_cuda_thread_block_clusters = True
        compare_reduce(fn, N, M, ty, opts)
    run_if_clusters_are_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_extended_clustered_reduction(fn, backend, ty):
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
        compare_reduce(fn, N, M, ty, opts)
    run_if_clusters_are_enabled(backend, helper)

@pytest.mark.parametrize('fn', reduce_funs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_reduction_codegen(fn, backend):
    N = 100
    M = 50
    x = np.random.randn(N, M).astype(np.float32)
    out = np.zeros((N,), dtype=x.dtype)
    p = {'outer': parpy.threads(N)}
    s1 = parpy.print_compiled(fn, [x, out, N], par_opts(backend, p))
    if not fn in multi_dim_reduce_funs:
        if backend == parpy.CompileBackend.Cuda:
            pat = r".*<<<dim3\(1, 1, 1\), dim3\(128, 1, 1\).*>>>\(.*\);"
        elif backend == parpy.CompileBackend.Metal:
            pat = r"parpy_metal::launch_kernel\(.*1, 1, 1, 128, 1, 1, .*\).*"
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
            pat = r"parpy_metal::launch_kernel\(.*1, 100, 1, 128, 1, 1, .*\).*"
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
            pat = r"parpy_metal::launch_kernel\(.*1, 8, 100, 128, 1, 1, .*\).*"
        else:
            pat = ""
        assert re.search(pat, s3, re.DOTALL) is not None
    else:
        assert len(s3) != 0

@pytest.mark.parametrize('fn', reduce_funs)
def test_clustered_reduction_codegen_in_cuda(fn):
    N = 100
    M = 50
    x = np.random.randn(N, M).astype(np.float32)
    out = np.zeros((N,), dtype=x.dtype)
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
        pat = r".*cudaFuncAttributeNonPortableClusterSizeAllowed.*"
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
        pat = r".*cudaFuncAttributeNonPortableClusterSizeAllowed.*"
        assert re.search(pat, s, re.DOTALL) is not None
    else:
        assert len(s) != 0

@pytest.mark.parametrize('fn', reduce_funs)
def test_clustered_reduction_compiles_in_cuda(fn):
    def helper():
        N = 100
        M = 50
        x = np.random.randn(N, M).astype(np.float32)
        out = np.zeros((N,), dtype=x.dtype)
        p = {
            'outer': parpy.threads(N),
            'inner': parpy.threads(4096).tpb(512)
        }
        opts = par_opts(parpy.CompileBackend.Cuda, p)
        opts.use_cuda_thread_block_clusters = True
        fn(x, out, N, opts=opts)
    run_if_clusters_are_enabled(parpy.CompileBackend.Cuda, helper)

# Tests using a custom step size.
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
    x = np.random.randn(N, M).astype(np.float32)
    out = np.zeros((N,), dtype=x.dtype)
    parpy.jit(odd_entries_sum)(x, out, N, M, opts=par_opts(backend, p))
    out_seq = np.zeros_like(out)
    odd_entries_sum(x, out_seq, N, M)
    assert np.allclose(out, out_seq, atol=1e-4)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_single_block(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'M': parpy.threads(32).par_reduction()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_odd_entries_multiblock(backend):
    def helper():
        p = {
            'N': parpy.threads(10),
            'M': parpy.threads(2048).par_reduction()
        }
        odd_entries_wrap(backend, p)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_reduce_mixed_parallel_loops(backend):
    def helper():
        @parpy.jit
        def softmax_mixed_parallelism(x, out, nrows, ncols):
            parpy.label('outer')
            for i in range(nrows):
                m = parpy.builtin.convert(0.0, parpy.types.F32)
                for j in range(ncols):
                    m = parpy.builtin.maximum(m, x[i, j])
                parpy.label('inner_map')
                out[i, :] = parpy.math.exp(x[i, :] - m)
                s = parpy.builtin.convert(0.0, parpy.types.F32)
                for j in range(ncols):
                    s = s + out[i, j]
                parpy.label('inner_map')
                out[i, :] /= s
        nrows = 10
        ncols = 52
        x = torch.randn(nrows, ncols, dtype=torch.float32)
        out = torch.empty_like(x)
        opts = par_opts(backend, {
            'outer': parpy.threads(nrows),
            'inner_map': parpy.threads(32),
        })
        softmax_mixed_parallelism(x, out, nrows, ncols, opts=opts)
        assert torch.allclose(out, torch.softmax(x, axis=1), atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('ty', dtypes)
def test_reduce_varied_input_size(backend, ty):
    def helper():
        @parpy.jit
        def fn(x, out, N):
            with parpy.gpu:
                parpy.label('N')
                for i in range(N[0]):
                    out[0] += x[i]
        N = np.full((1,), 10, dtype=np.int64)
        x = np.random.randn(10).astype(ty)
        out = np.zeros((1,), dtype=np.float32)
        if backend == parpy.CompileBackend.Metal and ty == np.float64:
            pytest.skip("64-bit floats are not supported in Metal")
        fn(x, out, N, opts=par_opts(backend, {'N': parpy.threads(10).par_reduction()}))
        assert np.allclose(out, np.sum(x), atol=1e-3)
    run_if_backend_is_enabled(backend, helper)
