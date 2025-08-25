import numpy as np
import parpy
import parpy.types as types
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

backend = parpy.CompileBackend.Cuda

@parpy.jit
def col_sum(x, y, N):
    parpy.label('N')
    for i in range(N):
        y[i] = parpy.sum(x[i, :])

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
    opts.backend = backend
    opts.parallelize = {'N': parpy.threads(N)}
    return parpy.print_compiled(col_sum, [x, y, N], opts)

def test_cuda_no_extra_includes():
    code = gen_code(np.float32, parpy.CompileOptions())
    assert not includes_cuda_fp16(code)
    assert not includes_cooperative_groups(code)

def test_cuda_16_bit_float_includes():
    code = gen_code(np.float16, parpy.CompileOptions())
    assert includes_cuda_fp16(code)
    assert not includes_cooperative_groups(code)

def test_cuda_clusters_enabled_includes():
    opts = parpy.CompileOptions()
    opts.use_cuda_thread_block_clusters = True
    code = gen_code(np.float32, opts)
    assert not includes_cuda_fp16(code)
    assert includes_cooperative_groups(code)

def records_cuda_graph(code):
    pat = r""".*cudaStreamBeginCapture.*cudaStreamEndCapture.*cudaGraphLaunch.*"""
    return re.search(pat, code, re.DOTALL) is not None

def test_cuda_graphs_included():
    opts = parpy.CompileOptions()
    opts.use_cuda_graphs = True
    code = gen_code(np.float32, opts)
    assert records_cuda_graph(code)

def test_cuda_graphs_compiles():
    def helper():
        opts = parpy.CompileOptions()
        opts.use_cuda_graphs = True
        code = gen_code(np.float32, opts)
        fn = parpy.compile_string("col_sum", code, opts)
        assert fn is not None
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def test_cuda_graph_runs_correctly():
    def helper():
        opts = parpy.CompileOptions()
        opts.use_cuda_graphs = True
        code = gen_code(np.float32, opts)
        fn = parpy.compile_string("col_sum", code, opts)

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
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def test_cuda_catch_runtime_error():
    def helper():
        code = """
#include "parpy_cuda.h"
extern "C"
int32_t f() {
    float *y;
    parpy_cuda_check_error(cudaMalloc(&y, -1));
    return 0;
}
        """
        fn = parpy.compile_string("f", code, parpy.CompileOptions())
        with pytest.raises(RuntimeError) as e_info:
            fn()
        assert e_info.match(r"out of memory")
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def clamp_helper(ext_id):
    params = [("x", types.F32), ("lo", types.F32), ("hi", types.F32)]
    res_ty = types.F32

    @parpy.external(ext_id, backend, parpy.Target.Device, header="<cuda_utils.h>")
    def clamp(x: types.F32, lo: types.F32, hi: types.F32) -> types.F32:
        if x < lo: return lo
        if x > hi: return hi
        return x

    # Clear the function cache to ensure we do not refer to the 'clamp_many'
    # defined as part of another test run.
    parpy.clear_cache()

    @parpy.jit
    def clamp_many(out, x):
        parpy.label('N')
        out[:] = clamp(x[:], 0.0, 10.0)
    x = torch.randn(10, dtype=torch.float32)
    out = torch.zeros_like(x)
    opts = par_opts(backend, {'N': parpy.threads(10)})
    opts.includes += ["test/code"]
    clamp_many(out, x, opts=opts)
    assert all(out >= 0.0) and all(out <= 10.0)

def test_call_user_defined_external_cuda():
    def helper():
        clamp_helper("clamp")
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def test_call_non_existent_external_cuda():
    def helper():
        with pytest.raises(RuntimeError, match="Compilation of generated CUDA code failed"):
            clamp_helper("clamp_non_existent")
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def test_call_invalid_decl_external_cuda():
    def helper():
        with pytest.raises(RuntimeError, match="Compilation of generated CUDA code failed"):
            clamp_helper("clamp_non_device")
    run_if_backend_is_enabled(parpy.CompileBackend.Cuda, helper)

def test_call_external_array_op_cuda():
    def helper():
        @parpy.external("sum_row_ext", backend, parpy.Target.Device, header="<cuda_utils.h>")
        def sum_row(x: types.pointer(types.F64), n: types.I64) -> types.F64:
            s = 0.0
            for i in range(n):
                s += x[i]
            return s

        @parpy.jit
        def sum_ext_seq(x, y, N, M):
            parpy.label('N')
            for i in range(N):
                y[i] = sum_row(x[i], M)
        x = torch.randn(10, 20, dtype=torch.float64)
        y = torch.zeros(10, dtype=torch.float64)
        opts = par_opts(backend, {'N': parpy.threads(10)})
        opts.includes += ['test/code']
        sum_ext_seq(x, y, 10, 20, opts=opts)
        assert torch.allclose(y, torch.sum(x, dim=1))
    run_if_backend_is_enabled(backend, helper)

def test_call_external_inconsistent_shapes_cuda():
    def helper():
        @parpy.external("sum_row_ext", backend, parpy.Target.Device, header="<cuda_utils.h>")
        def sum_row(x: types.pointer(types.F64), n: types.I64) -> types.F64:
            s = 0.0
            for i in range(n):
                s += x[i]
            return s

        with pytest.raises(TypeError, match="incompatible types"):
            @parpy.jit
            def sum_ext_seq(out, x, y):
                with parpy.gpu:
                    out[0] = sum_row(x, 10) + sum_row(y, 20)
            x = torch.randn(10, dtype=torch.float64)
            y = torch.randn(20, dtype=torch.float64)
            out = torch.zeros(1, dtype=torch.float64)
            opts = par_opts(backend, {})
            opts.includes += ['test/code']
            sum_ext_seq(out, x, y, opts=opts)
            assert torch.allclose(out, torch.sum(x) + torch.sum(y))
    run_if_backend_is_enabled(backend, helper)

def test_host_external_cuda_fails():
    with pytest.raises(RuntimeError, match="Host externals are not supported in the CUDA backend"):
        @parpy.external("dummy", backend, parpy.Target.Host)
        def cu_id(x: types.I64) -> types.I64:
            return x

def test_block_parallel_external_cuda():
    def helper():
        @parpy.external(
                "warp_sum", backend, parpy.Target.Device,
                header="<cuda_utils.h>", parallelize=parpy.threads(32))
        def warp_sum_custom(x: types.pointer(types.F32)) -> types.F32:
            return np.sum(x)

        @parpy.jit
        def sum_rows_using_ext(out, x, N):
            parpy.label('N')
            for i in range(N):
                out[i] = warp_sum_custom(x[i])
        x = torch.randn(10, 32, dtype=torch.float32)
        out = torch.empty(10, dtype=torch.float32)
        opts = par_opts(backend, {'N': parpy.threads(10)})
        opts.includes += ['test/code']
        sum_rows_using_ext(out, x, 10, opts=opts)
        assert torch.allclose(out, torch.sum(x, dim=1))
    run_if_backend_is_enabled(backend, helper)
