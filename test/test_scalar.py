import parpy
import pytest
from parpy.parpy import ElemSize, ScalarSizes
import re
import torch

from common import *

T = parpy.types.type_var()

@parpy.jit
def simple_fun(x: T):
    with parpy.gpu:
        y = parpy.builtin.convert(x, T)

def scalar_sizes(backend):
    opts = parpy.CompileOptions()
    opts.backend = backend
    return ScalarSizes(opts)

def expected_int_type(backend):
    return scalar_sizes(backend).int

def expected_float_type(backend):
    return scalar_sizes(backend).float

def expected_string(sz, backend):
    triton = parpy.CompileBackend.Triton
    match sz:
        case ElemSize.Bool:
            return "int1" if backend == triton else "bool"
        case ElemSize.I8:
            return "int8"
        case ElemSize.I16:
            return "int16"
        case ElemSize.I32:
            return "int32"
        case ElemSize.I64:
            return "int64"
        case ElemSize.U8:
            return "uint8"
        case ElemSize.U16:
            return "uint16"
        case ElemSize.U32:
            return "uint32"
        case ElemSize.U64:
            return "uint64"
        case ElemSize.F16:
            return "float16" if backend == triton else "half"
        case ElemSize.F32:
            return "float32" if backend == triton else "float"
        case ElemSize.F64:
            return "float64" if backend == triton else "double"
        case _:
            raise RuntimeError(f"Unsupported element size {sz}")

def assert_code_contains(code, sz, backend):
    pat = expected_string(sz, backend)
    assert re.search(pat, code, re.DOTALL) is not None

@pytest.mark.parametrize('backend', compiler_backends)
def test_int_type_default(backend):
    opts = par_opts(backend, {})
    code = parpy.print_compiled(simple_fun, [1], opts)
    expected = expected_int_type(backend)
    assert_code_contains(code, expected, backend)

@pytest.mark.parametrize('backend', compiler_backends)
def test_float_type_default(backend):
    opts = par_opts(backend, {})
    code = parpy.print_compiled(simple_fun, [1.0], opts)
    expected = expected_float_type(backend)
    assert_code_contains(code, expected, backend)

int_types = [
    ElemSize.I8, ElemSize.I16, ElemSize.I32, ElemSize.I64,
    ElemSize.U8, ElemSize.U16, ElemSize.U32, ElemSize.U64,
]
float_types = [ElemSize.F16, ElemSize.F32, ElemSize.F64]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('int_ty', int_types)
def test_int_type_forced(backend, int_ty):
    opts = par_opts(backend, {})
    opts.force_int_size = int_ty
    code = parpy.print_compiled(simple_fun, [1], opts)
    assert_code_contains(code, int_ty, backend)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('float_ty', float_types)
def test_float_type_forced(backend, float_ty):
    opts = par_opts(backend, {})
    opts.force_float_size = float_ty
    if backend == parpy.CompileBackend.Metal and float_ty == ElemSize.F64:
        with pytest.raises(TypeError) as e_info:
            code = parpy.print_compiled(simple_fun, [1.0], opts)
        assert e_info.match(r"does not support.*double-precision float.*")
    else:
        code = parpy.print_compiled(simple_fun, [1.0], opts)
        assert_code_contains(code, float_ty, backend)
