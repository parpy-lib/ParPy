import parpy
import pytest
from parpy.parpy import ElemSize, ScalarSizes
import re
import torch

from common import *

@parpy.jit
def simple_fun(x):
    with parpy.gpu:
        y = x

def scalar_sizes(backend):
    opts = parpy.CompileOptions()
    opts.backend = backend
    return ScalarSizes(opts)

def expected_int_type(backend):
    return scalar_sizes(backend).int

def expected_float_type(backend):
    return scalar_sizes(backend).float

def expected_string(sz):
    match sz:
        case ElemSize.Bool:
            return "bool"
        case ElemSize.I8:
            return "int8_t"
        case ElemSize.I16:
            return "int16_t"
        case ElemSize.I32:
            return "int32_t"
        case ElemSize.I64:
            return "int64_t"
        case ElemSize.U8:
            return "uint8_t"
        case ElemSize.U16:
            return "uint16_t"
        case ElemSize.U32:
            return "uint32_t"
        case ElemSize.U64:
            return "uint64_t"
        case ElemSize.F16:
            return "half"
        case ElemSize.F32:
            return "float"
        case ElemSize.F64:
            return "double"
        case _:
            raise RuntimeError(f"Unsupported element size {sz}")

def assert_code_contains(code, sz):
    pat = expected_string(sz)
    assert re.search(pat, code, re.DOTALL) is not None

@pytest.mark.parametrize('backend', compiler_backends)
def test_int_type_default(backend):
    opts = par_opts(backend, {})
    code = parpy.print_compiled(simple_fun, [1], opts)
    expected = expected_int_type(backend)
    assert_code_contains(code, expected)

@pytest.mark.parametrize('backend', compiler_backends)
def test_float_type_default(backend):
    opts = par_opts(backend, {})
    code = parpy.print_compiled(simple_fun, [1.0], opts)
    expected = expected_float_type(backend)
    assert_code_contains(code, expected)

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
    assert_code_contains(code, int_ty)

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
        assert_code_contains(code, float_ty)
