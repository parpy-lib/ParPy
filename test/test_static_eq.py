import numpy as np
import parpy
import pytest

from common import *

np.random.seed(1234)

@parpy.jit
def parpy_backend_set_value(x):
    with parpy.gpu:
        if parpy.builtin.static_backend_eq(parpy.CompileBackend.Cuda):
            x[0] = 0
        elif parpy.builtin.static_backend_eq(parpy.CompileBackend.Metal):
            x[0] = 1
        elif parpy.builtin.static_backend_eq(parpy.CompileBackend.Triton):
            x[0] = 2
        else:
            parpy.builtin.static_fail("Function only supports CUDA, Metal, and Triton")

@pytest.mark.parametrize('backend', compiler_backends)
def test_backend_set_value(backend):
    def helper():
        x = np.ndarray((1,), dtype=np.int32)
        parpy_backend_set_value(x, opts=par_opts(backend, {}))
        if backend == parpy.CompileBackend.Cuda:
            assert x[0] == 0
        elif backend == parpy.CompileBackend.Metal:
            assert x[0] == 1
        elif backend == parpy.CompileBackend.Triton:
            assert x[0] == 2
        else:
            pytest.failwith(f"Unsupported backend {backend}")
    run_if_backend_is_enabled(backend, helper)

sz = parpy.types.type_var()
N = parpy.types.shape_var()

@parpy.jit
def parpy_type_set_value(x: parpy.types.buffer(sz, [N])):
    with parpy.gpu:
        if parpy.builtin.static_types_eq(sz, parpy.types.Bool):
            x[0] = False
        elif parpy.builtin.static_types_eq(sz, parpy.types.I8):
            x[0] = 0
        elif parpy.builtin.static_types_eq(sz, parpy.types.I16):
            x[0] = 1
        elif parpy.builtin.static_types_eq(sz, parpy.types.I32):
            x[0] = 2
        elif parpy.builtin.static_types_eq(sz, parpy.types.I64):
            x[0] = 3
        elif parpy.builtin.static_types_eq(sz, parpy.types.U8):
            x[0] = parpy.builtin.convert(4, sz)
        elif parpy.builtin.static_types_eq(sz, parpy.types.U16):
            x[0] = parpy.builtin.convert(5, sz)
        elif parpy.builtin.static_types_eq(sz, parpy.types.U32):
            x[0] = parpy.builtin.convert(6, sz)
        elif parpy.builtin.static_types_eq(sz, parpy.types.U64):
            x[0] = parpy.builtin.convert(7, sz)
        elif parpy.builtin.static_types_eq(sz, parpy.types.F16):
            x[0] = 8.0
        elif parpy.builtin.static_types_eq(sz, parpy.types.F32):
            x[0] = 9.0
        elif parpy.builtin.static_types_eq(sz, parpy.types.F64) and \
                not parpy.builtin.static_backend_eq(parpy.CompileBackend.Metal):
            x[0] = 10.0
        else:
            parpy.builtin.static_fail("Function not supported for type")

cases = [
    (parpy.types.Bool, False),
    (parpy.types.I8, 0),
    (parpy.types.I16, 1),
    (parpy.types.I32, 2),
    (parpy.types.I64, 3),
    (parpy.types.U8, 4),
    (parpy.types.U16, 5),
    (parpy.types.U32, 6),
    (parpy.types.U64, 7),
    (parpy.types.F16, 8.0),
    (parpy.types.F32, 9.0),
    (parpy.types.F64, 10.0),
]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_case', cases)
def test_type_set_value(backend, test_case):
    def helper():
        sz, expected = test_case
        x = np.zeros((1,), sz.to_numpy())
        opts = par_opts(backend, {})
        if backend == parpy.CompileBackend.Metal and sz == parpy.types.F64:
            with pytest.raises(RuntimeError) as e_info:
                parpy_type_set_value(x, opts=opts)
            assert e_info.match("Function not supported for type")
        else:
            parpy_type_set_value(x, opts=opts)
            assert x[0] == expected
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_case', cases)
def test_type_set_value_compiles(backend, test_case):
    sz, _ = test_case
    x = np.zeros((1,), sz.to_numpy())
    opts = par_opts(backend, {})
    if backend == parpy.CompileBackend.Metal and sz == parpy.types.F64:
        with pytest.raises(RuntimeError) as e_info:
            parpy.print_compiled(parpy_type_set_value, [x], opts)
        assert e_info.match("Function not supported for type")
    else:
        code = parpy.print_compiled(parpy_type_set_value, [x], opts)
        assert len(code) != 0
