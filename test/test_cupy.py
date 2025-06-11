# Tests that CuPy arrays can be used interchangeably with Torch tensors.

import importlib
import parir
import pytest

from common import *

@parir.jit
def add(a, b, c, N):
    parir.label('N')
    c[:] = a[:] + b[:]

@pytest.mark.skipif(importlib.util.find_spec('cupy') is None, reason="This test requires CuPy")
@pytest.mark.parametrize('backend', compiler_backends)
def test_call_cupy(backend):
    def helper():
        import cupy
        a = cupy.random.randn(10)
        b = cupy.random.randn(10)
        c = cupy.ndarray(10)
        add(a, b, c, 10, opts=par_opts(backend, {'N': parir.threads(10)}))
        assert cupy.allclose(a + b, c)
    run_if_backend_is_enabled(backend, helper)
