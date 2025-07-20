import numpy as np
import prickle
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

@prickle.jit
def col_sum(x, y, N):
    prickle.label('N')
    for i in range(N):
        y[i] = prickle.sum(x[i, :])

# The 'cuda_fp16.h' header should only be included when we are using 16-bit
# floats.
def includes_cuda_fp16(code):
    pat = r""".*include.*<cuda_fp16.h>.*"""
    return re.search(pat, code, re.DOTALL) is not None

# The 'cooperative_groups.h' header should only be included when the user
# has enabled the use of thread block clusters via an option.
def includes_cooperative_groups(code):
    pat = r""".*include.*<cooperative_groups.h>.*"""
    return re.search(pat, code, re.DOTALL) is not None

def gen_code(ty, opts):
    N = 20
    M = 10
    x = np.random.randn(N, M).astype(ty)
    y = np.zeros((N,)).astype(ty)
    opts.backend = prickle.CompileBackend.Cuda
    opts.parallelize = {'N': prickle.threads(N)}
    return prickle.print_compiled(col_sum, [x, y, N], opts)

def test_cuda_no_extra_includes():
    code = gen_code(np.float32, prickle.CompileOptions())
    assert not includes_cuda_fp16(code)
    assert not includes_cooperative_groups(code)

def test_cuda_16_bit_float_includes():
    code = gen_code(np.float16, prickle.CompileOptions())
    assert includes_cuda_fp16(code)
    assert not includes_cooperative_groups(code)

def test_cuda_clusters_enabled_includes():
    opts = prickle.CompileOptions()
    opts.use_cuda_thread_block_clusters = True
    code = gen_code(np.float32, opts)
    assert not includes_cuda_fp16(code)
    assert includes_cooperative_groups(code)

def records_cuda_graph(code):
    pat = r""".*cudaStreamBeginCapture.*cudaStreamEndCapture.*cudaGraphLaunch.*"""
    return re.search(pat, code, re.DOTALL) is not None

def test_cuda_graphs_included():
    opts = prickle.CompileOptions()
    opts.use_cuda_graphs = True
    code = gen_code(np.float32, opts)
    assert records_cuda_graph(code)

def test_cuda_graphs_compiles():
    def helper():
        opts = prickle.CompileOptions()
        opts.use_cuda_graphs = True
        code = gen_code(np.float32, opts)
        fn = prickle.compile_string("col_sum", code, opts)
        assert fn is not None
    run_if_backend_is_enabled(prickle.CompileBackend.Cuda, helper)

def test_cuda_graph_runs_correctly():
    def helper():
        opts = prickle.CompileOptions()
        opts.use_cuda_graphs = True
        code = gen_code(np.float32, opts)
        fn = prickle.compile_string("col_sum", code, opts)

        # Run the program sequentially and in parallel using CUDA graphs and
        # ensure we get the same result.
        N = 20
        M = 10
        x = np.random.randn(N, M).astype(np.float32)
        y1 = np.zeros((N,)).astype(np.float32)
        opts.seq = True
        col_sum(x, y1, 20, opts=opts)
        opts.seq = False
        y2 = np.zeros((N,)).astype(np.float32)
        fn(x, y2, 20)
        assert np.allclose(y1, y2)
    run_if_backend_is_enabled(prickle.CompileBackend.Cuda, helper)
