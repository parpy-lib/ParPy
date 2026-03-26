import numpy as np
import parpy
import parpy.types as types
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

backend = parpy.CompileBackend.Triton

# The function below is split up by the inter-block transformation, and
# the value of t1 is stored in a temporary allocation to carry over
# between kernels. The temporary allocation is indexed by the loop
# variable (of the GPU context loop). This leads to a shape mismatch,
# because 't1' gets a blocked type while 'z[0]' is scalar, because of
# the constant index.
#
# The compiler should resolve this by replacing the constant index with
# the loop variable, which is valid as long as the loop only runs once.
# Subsequent functions use variations to ensure the compiler resolves this in a
# general way.
@parpy.jit
def f1(x, y, z, N):
    with parpy.gpu:
        t1 = 1.0
        parpy.label('N')
        for i in range(N):
            y[i] = parpy.reduce.sum(x[i,:])
        z[0] = t1

@parpy.jit
def f2(x, y, z, N):
    parpy.label('K')
    for k in range(1):
        t1 = 1.0
        parpy.label('N')
        for i in range(N):
            y[i] = parpy.reduce.sum(x[i,:])
        z[0] = t1

@parpy.jit
def f3(x, y, z, N):
    with parpy.gpu:
        t1 = 1.0
        parpy.label('N')
        for i in range(N):
            y[i] = parpy.reduce.sum(x[i,:])
        z[1] = t1

@parpy.jit
def f4(x, y, z, N):
    parpy.label('K')
    for k in range(2, 3):
        t1 = 1.0
        parpy.label('N')
        for i in range(N):
            y[i] = parpy.reduce.sum(x[i,:])
        z[1] = t1

funcs = [f1, f2, f3, f4]

@pytest.mark.parametrize('fn', funcs)
def test_inter_block_shape_mismatch(fn):
    def helper():
        N = 60
        M = 10
        x = np.random.randn(N, M).astype(np.float32)
        y = np.empty(N, dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        opts = par_opts(backend, {
            'K': parpy.threads(1),
            'N': parpy.threads(N).tpb(32),
        })
        fn(x, y, z, N, opts=opts)
        assert np.allclose(y, np.sum(x, axis=1))
        assert z[0] + z[1] == 1.0
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_multi_writes_to_same_location(backend):
    def helper():
        @parpy.jit
        def f(x, N):
            parpy.label('N')
            for i in range(N):
                x[0] += i
        opts = par_opts(backend, {'N': parpy.threads(32).par_reduction()})
        x = np.zeros(1, dtype=np.int32)
        f(x, 10, opts=opts)
        assert np.allclose(x, sum(range(10)))
        opts = par_opts(backend, {'N': parpy.threads(32)})
        if backend == parpy.CompileBackend.Triton:
            with pytest.raises(RuntimeError) as e_info:
                f(x, 10, opts=opts)
            assert e_info.match(r"Try marking.*as a parallel reduction")
        else:
            f(x, 10, opts=opts)
    run_if_backend_is_enabled(backend, helper)
