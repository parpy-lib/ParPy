import numpy as np
import prickle
import pytest
import re

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
