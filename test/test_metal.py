import numpy as np
import parpy
import pytest
import subprocess
import tempfile

from common import *

np.random.seed(1234)

@parpy.jit
def sum_elems_per_row(x, y, N, M):
    for i in range(N):
        y[i] = 0.0
        parpy.label('M')
        for j in range(M):
            y[i] += x[i,j]

@pytest.mark.parametrize('backend', compiler_backends)
def test_metal_no_parallelism(backend):
    N = 10
    M = 20
    x = np.random.randn(N, M).astype(np.float32)
    y = np.ndarray(N).astype(np.float32)
    p = {'M': parpy.threads(M).reduce()}
    opts = par_opts(backend, p)
    if backend == parpy.CompileBackend.Metal:
        if parpy.backend.is_enabled(backend):
            sum_elems_per_row(x, y, N, M, opts=opts)
            assert np.allclose(np.sum(x, axis=1), y, atol=1e-5)
        return
    with pytest.raises(RuntimeError) as e_info:
        code = parpy.print_compiled(sum_elems_per_row, [x, y, N, M], opts)
    assert e_info.match(r".*Assignments are not allowed outside parallel code.*")

def test_metal_catch_runtime_error():
    backend = parpy.CompileBackend.Metal
    def helper():
        code = """
#include "parpy_metal.h"
extern "C"
int32_t f() {
    MTL::Buffer *buf;
    parpy_metal_check_error(parpy_metal::alloc(&buf, -1));
    return 0;
}
        """
        with pytest.raises(RuntimeError) as e_info:
            opts = par_opts(backend, {})
            fn = parpy.compile_string("f", code, opts)
            fn()
        assert e_info.match(r"Buffer allocation failed")
    run_if_backend_is_enabled(backend, helper)

def test_metal_host_external():
    backend = parpy.CompileBackend.Metal
    def helper():
        import parpy.types as types
        @parpy.external("add_metal_host", backend, parpy.Target.Host, header="<metal_utils.h>")
        def add_metal(x: types.F32, y: types.F32) -> types.F32:
            return x + y

        @parpy.jit
        def set_to_sum(x, a, b):
            c = add_metal(a, b)
            parpy.label('N')
            x[:] = c
        x = torch.zeros(10, dtype=torch.float32)
        a = 1.0
        b = 2.0
        opts = par_opts(backend, {'N': parpy.threads(10)})
        opts.includes += ['test/code']
        set_to_sum(x, a, b, opts=opts)
        assert torch.allclose(x, torch.tensor(3.0))
    run_if_backend_is_enabled(backend, helper)

def test_metal_parallel_host_external_fails():
    backend = parpy.CompileBackend.Metal
    with pytest.raises(RuntimeError) as e_info:
        import parpy.types as types
        @parpy.external(
            "dummy", backend, parpy.Target.Host,
            parallelize=parpy.threads(1024)
        )
        def dummy(x: types.I64) -> types.I64:
            return x
    assert e_info.match("Host externals cannot be parallel")
