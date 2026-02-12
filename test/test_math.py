import numpy as np
import parpy
import torch

from common import *

@parpy.jit
def parpy_abs(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.abs(x[0])

@parpy.jit
def parpy_cos(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.cos(x[0])

@parpy.jit
def parpy_exp(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.exp(x[0])

@parpy.jit
def parpy_log(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.log(x[0])

@parpy.jit
def parpy_sin(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.sin(x[0])

@parpy.jit
def parpy_sqrt(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.sqrt(x[0])

@parpy.jit
def parpy_tanh(dst, x):
    with parpy.gpu:
        dst[0] = parpy.math.tanh(x[0])

unary_tests = [
    (parpy_abs, np.abs),
    (parpy_cos, np.cos),
    (parpy_exp, np.exp),
    (parpy_log, np.log),
    (parpy_sin, np.sin),
    (parpy_sqrt, np.sqrt),
    (parpy_tanh, np.tanh),
]

data_types = [
    parpy.types.I8,
    parpy.types.I16,
    parpy.types.I32,
    parpy.types.I64,
    parpy.types.U8,
    parpy.types.U16,
    parpy.types.U32,
    parpy.types.U64,
    parpy.types.F16,
    parpy.types.F32,
    parpy.types.F64,
]

def unop_should_fail(backend, fn, dtype, running):
    if fn == parpy_abs:
        if backend == parpy.CompileBackend.Cuda:
            return False
        elif backend == parpy.CompileBackend.Metal:
            return dtype == parpy.types.F64
        elif backend == parpy.CompileBackend.Triton:
            return False
    elif running and fn == parpy_tanh and backend == parpy.CompileBackend.Cuda:
        # The tanh operation for half-precision numbers seems to have been
        # added in CUDA 12.8. Therefore, if we are running the generated code,
        # this should fail if using a less recent version.
        major, minor = get_cuda_version()
        if major > 12 or major == 12 and minor >= 8:
            return not dtype in [parpy.types.F16, parpy.types.F32, parpy.types.F64]
        else:
            return not dtype in [parpy.types.F32, parpy.types.F64]
    elif fn == parpy_tanh and backend == parpy.CompileBackend.Triton:
        return True
    else:
        if backend == parpy.CompileBackend.Cuda:
            return not dtype in [parpy.types.F16, parpy.types.F32, parpy.types.F64]
        elif backend == parpy.CompileBackend.Metal:
            return not dtype in [parpy.types.F16, parpy.types.F32]
        elif backend == parpy.CompileBackend.Triton:
            return not dtype in [parpy.types.F16, parpy.types.F32, parpy.types.F64]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', unary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_run_unary_operation(backend, test_data, dtype):
    def helper():
        parpy_fn, seq_fn = test_data
        x = np.ndarray((1,), dtype=dtype.to_numpy())
        x[0] = 0.5
        dst = np.zeros((1,), dtype=dtype.to_numpy())
        opts = par_opts(backend, {})
        if unop_should_fail(backend, parpy_fn, dtype, True):
            with pytest.raises(RuntimeError):
                parpy_fn(dst, x, opts=opts)
        else:
            parpy_fn(dst, x, opts=opts)
            expected = seq_fn(x)
            assert np.allclose(dst, expected, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', unary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_compile_unary_operation(backend, test_data, dtype):
    parpy_fn, _ = test_data
    x = np.ndarray((1,), dtype=dtype.to_numpy())
    x[0] = 0.5
    dst = np.zeros((1,), dtype=dtype.to_numpy())
    opts = par_opts(backend, {})
    if unop_should_fail(backend, parpy_fn, dtype, False):
        with pytest.raises(RuntimeError):
            parpy.print_compiled(parpy_fn, [dst, x], opts)
    else:
        code = parpy.print_compiled(parpy_fn, [dst, x], opts)
        assert len(code) != 0

@parpy.jit
def parpy_atan2(dst, x, y):
    with parpy.gpu:
        dst[0] = parpy.math.atan2(x[0], y[0])

binary_tests = [
    (parpy_atan2, np.arctan2),
]

def binop_should_fail(backend, fn, dtype):
    if fn == parpy_atan2:
        if backend == parpy.CompileBackend.Cuda:
            return not dtype in [parpy.types.F32, parpy.types.F64]
        elif backend == parpy.CompileBackend.Metal:
            return not dtype in [parpy.types.F16, parpy.types.F32]
        elif backend == parpy.CompileBackend.Triton:
            return True
    else:
        if backend == parpy.CompileBackend.Cuda:
            return False
        elif backend == parpy.CompileBackend.Metal:
            return dtype == parpy.types.F64
        elif backend == parpy.CompileBackend.Triton:
            return False

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', binary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_run_binary_operation(backend, test_data, dtype):
    def helper():
        parpy_fn, seq_fn = test_data
        x = np.ndarray((1,), dtype=dtype.to_numpy())
        x[0] = 0.5
        y = np.ndarray((1,), dtype=dtype.to_numpy())
        y[0] = 0.5
        dst = np.zeros((1,), dtype=dtype.to_numpy())
        opts = par_opts(backend, {})
        if binop_should_fail(backend, parpy_fn, dtype):
            with pytest.raises(RuntimeError):
                parpy_fn(dst, x, y, opts=opts)
        else:
            parpy_fn(dst, x, y, opts=opts)
            expected = seq_fn(x, y)
            assert np.allclose(dst, expected, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', binary_tests)
@pytest.mark.parametrize('dtype', data_types)
def test_compile_binary_operation(backend, test_data, dtype):
    parpy_fn, _ = test_data
    x = np.ndarray((1,), dtype=dtype.to_numpy())
    x[0] = 0.5
    y = np.ndarray((1,), dtype=dtype.to_numpy())
    y[0] = 0.5
    dst = np.zeros((1,), dtype=dtype.to_numpy())
    opts = par_opts(backend, {})
    if binop_should_fail(backend, parpy_fn, dtype):
        with pytest.raises(RuntimeError):
            parpy.print_compiled(parpy_fn, [dst, x, y], opts)
    else:
        code = parpy.print_compiled(parpy_fn, [dst, x, y], opts)
        assert len(code) != 0
