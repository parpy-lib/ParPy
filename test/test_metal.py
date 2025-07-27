import numpy as np
import prickle
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

@prickle.jit
def sum_elems_per_row(x, y, N):
    for i in range(N):
        prickle.label('M')
        y[i] = prickle.sum(x[i, :])

@pytest.mark.parametrize('backend', compiler_backends)
def test_metal_no_parallelism(backend):
    N = 10
    M = 20
    x = np.random.randn(N, M).astype(np.float32)
    y = np.ndarray(N).astype(np.float32)
    p = {'M': prickle.threads(M)}
    opts = par_opts(backend, p)
    if backend == prickle.CompileBackend.Metal:
        if prickle.backend.is_enabled(backend):
            sum_elems_per_row(x, y, N, opts=opts)
            assert np.allclose(np.sum(x, axis=1), y, atol=1e-5)
        return
    with pytest.raises(RuntimeError) as e_info:
        code = prickle.print_compiled(sum_elems_per_row, [x, y, N], opts)
    assert e_info.match(r".*Assignments are not allowed outside parallel code.*")

def test_metal_catch_runtime_error():
    backend = prickle.CompileBackend.Metal
    def helper():
        code = """
#include "prickle_metal.h"
extern "C"
int32_t f() {
    MTL::Buffer *buf;
    prickle_metal_check_error(prickle_metal::alloc(&buf, -1));
    return 0;
}
        """
        with pytest.raises(RuntimeError) as e_info:
            opts = par_opts(backend, {})
            fn = prickle.compile_string("f", code, opts)
            fn()
        assert e_info.match(r"Buffer allocation failed")
    run_if_backend_is_enabled(backend, helper)
