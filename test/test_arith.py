from enum import Enum
import math
import parpy
import pytest
import numpy as np

from common import *

@parpy.jit
def parpy_add(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] + b[0]

@parpy.jit
def parpy_sub(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] - b[0]

@parpy.jit
def parpy_mul(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] * b[0]

@parpy.jit
def parpy_div(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] / b[0]

@parpy.jit
def parpy_div_int(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] // b[0]

@parpy.jit
def parpy_rem(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] % b[0]

@parpy.jit
def parpy_pow(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] ** b[0]

@parpy.jit
def parpy_aug_ops(dst, a, b):
    with parpy.gpu:
        dst[0] += a[0]
        dst[0] -= b[0]

@parpy.jit
def parpy_bit_and(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] & b[0]

@parpy.jit
def parpy_bit_or(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] | b[0]

@parpy.jit
def parpy_bit_xor(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] ^ b[0]

@parpy.jit
def parpy_bit_shl(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] << b[0]

@parpy.jit
def parpy_bit_shr(dst, a, b):
    with parpy.gpu:
        dst[0] = a[0] >> b[0]

arith_funs = [
    parpy_add, parpy_sub, parpy_mul, parpy_div, parpy_div_int, parpy_rem,
    parpy_pow, parpy_aug_ops,
]
arith_tests = [
    (parpy_add, lambda x, y: x + y),
    (parpy_sub, lambda x, y: x - y),
    (parpy_mul, lambda x, y: x * y),
    (parpy_div_int, lambda x, y: x // y),
    (parpy_div, lambda x, y: x / y),
    (parpy_rem, lambda x, y: x % y),
    (parpy_pow, lambda x, y: x ** y),
    (parpy_aug_ops, lambda x,y: x - y)
]
bitwise_funs = [
    parpy_bit_and, parpy_bit_or, parpy_bit_xor, parpy_bit_shl, parpy_bit_shr
]
bitwise_tests = [
    (parpy_bit_and, np.bitwise_and),
    (parpy_bit_or, np.bitwise_or),
    (parpy_bit_xor, np.bitwise_xor),
    (parpy_bit_shl, np.left_shift),
    (parpy_bit_shr, np.right_shift),
]
tests = arith_tests + bitwise_tests

signed_int_tys = [parpy.types.I8, parpy.types.I16, parpy.types.I32, parpy.types.I64]
unsigned_int_tys = [parpy.types.U8, parpy.types.U16, parpy.types.U32, parpy.types.U64]
int_tys = signed_int_tys + unsigned_int_tys
float_tys = [parpy.types.F16, parpy.types.F32, parpy.types.F64]
arith_tys = int_tys + float_tys

def is_float_dtype(dtype):
    return dtype in float_tys

def is_untyped_dtype(dtype):
    return dtype in unsigned_int_tys

def is_invalid_div_or_rem_call(fn, ldtype, rdtype):
    return ((fn.__name__ == "parpy_div_int" or fn.__name__ == "parpy_rem") and
        (is_float_dtype(ldtype) or is_float_dtype(rdtype)))

def op_supported_types(backend, op):
    if op in [parpy_add, parpy_sub, parpy_mul, parpy_aug_ops]:
        return int_tys + float_tys
    elif op == parpy_div:
        return float_tys
    elif op == parpy_pow:
        return float_tys
    elif op in [parpy_div_int, parpy_rem] + bitwise_funs:
        return int_tys
    else:
        raise RuntimeError(f"Unclassified function: {fn.__name__} in op_should_fail")

def op_expected_error(backend, op, ldtype, rdtype):
    # If either argument is a type not supported by the operation, we expect
    # the call to result in a type error.
    accepted_types = op_supported_types(backend, op)
    if not (ldtype in accepted_types and rdtype in accepted_types):
        return TypeError

    # Metal does not support 64-bit floats
    if backend == parpy.CompileBackend.Metal and \
            (ldtype == parpy.types.F64 or rdtype == parpy.types.F64):
        return TypeError

    # CUDA does not support the 'pow' operation on 16-bit floats. Note that if
    # one operand is a larger floating-point type, the 16-bit float is coerced
    # to the larger float size.
    if backend == parpy.CompileBackend.Cuda and \
            op == parpy_pow and \
            (ldtype == parpy.types.F16 and rdtype == parpy.types.F16):
        return TypeError

    # The pow operation is not supported in Triton.
    if backend == parpy.CompileBackend.Triton and op == parpy_pow:
        return RuntimeError

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', arith_tests)
@pytest.mark.parametrize('dtype', arith_tys)
def test_run_binop(backend, test_data, dtype):
    def helper():
        parpy_fn, seq_fn = test_data
        x = np.ndarray((1,), dtype=dtype.to_numpy())
        y = np.ndarray((1,), dtype=dtype.to_numpy())
        x[0] = 13
        y[0] = 4
        dst = np.zeros((1,), dtype=dtype.to_numpy())
        opts = par_opts(backend, {})
        err = op_expected_error(backend, parpy_fn, dtype, dtype)
        if err is not None:
            with pytest.raises(err):
                parpy_fn(dst, x, y, opts=opts)
        else:
            parpy_fn(dst, x, y, opts=opts)
            expected = seq_fn(x, y)
            assert np.allclose(dst, expected, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', arith_tests)
@pytest.mark.parametrize('dtype', arith_tys)
def test_compile_binop(backend, test_data, dtype):
    parpy_fn, seq_fn = test_data
    x = np.ndarray((1,), dtype=dtype.to_numpy())
    y = np.ndarray((1,), dtype=dtype.to_numpy())
    x[0] = 13
    y[0] = 4
    dst = np.zeros((1,), dtype=dtype.to_numpy())
    opts = par_opts(backend, {})
    err = op_expected_error(backend, parpy_fn, dtype, dtype)
    if err is not None:
        with pytest.raises(err):
            parpy.print_compiled(parpy_fn, [dst, x, y], opts)
    else:
        code = parpy.print_compiled(parpy_fn, [dst, x, y], opts)
        assert len(code) != 0

# All allowed pairs of types in arithmetic operations, where the LHS should be
# coerced to the RHS type.
arith_ty_pairs = [
    (parpy.types.I8, parpy.types.I16),
    (parpy.types.I8, parpy.types.I16),
    (parpy.types.I8, parpy.types.I32),
    (parpy.types.I8, parpy.types.I64),
    (parpy.types.I16, parpy.types.I32),
    (parpy.types.I16, parpy.types.I64),
    (parpy.types.I32, parpy.types.I64),
    (parpy.types.U8, parpy.types.U16),
    (parpy.types.U8, parpy.types.U32),
    (parpy.types.U8, parpy.types.U64),
    (parpy.types.U16, parpy.types.U32),
    (parpy.types.U16, parpy.types.U64),
    (parpy.types.U32, parpy.types.U64),
    (parpy.types.F16, parpy.types.F32),
    (parpy.types.F16, parpy.types.F64),
    (parpy.types.F32, parpy.types.F64),
]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', arith_tests)
@pytest.mark.parametrize('dtypes', arith_ty_pairs)
def test_run_binop_distinct_operand_types(backend, test_data, dtypes):
    def helper():
        parpy_fn, seq_fn = test_data
        ldtype, rdtype = dtypes
        x = np.ndarray((1,), dtype=ldtype.to_numpy())
        y = np.ndarray((1,), dtype=rdtype.to_numpy())
        x[0] = 13
        y[0] = 4
        dst = np.zeros((1,), dtype=rdtype.to_numpy())
        opts = par_opts(backend, {})
        err = op_expected_error(backend, parpy_fn, ldtype, rdtype)
        if err is not None:
            with pytest.raises(err):
                parpy_fn(dst, x, y, opts=opts)
        else:
            parpy_fn(dst, x, y, opts=opts)
            expected = seq_fn(x, y)
            assert np.allclose(dst, expected, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('test_data', arith_tests)
@pytest.mark.parametrize('dtypes', arith_ty_pairs)
def test_compile_binop_distinct_operand_types(backend, test_data, dtypes):
    parpy_fn, _ = test_data
    ldtype, rdtype = dtypes
    x = np.ndarray((1,), dtype=ldtype.to_numpy())
    y = np.ndarray((1,), dtype=rdtype.to_numpy())
    x[0] = 13
    y[0] = 4
    dst = np.zeros((1,), dtype=rdtype.to_numpy())
    opts = par_opts(backend, {})
    err = op_expected_error(backend, parpy_fn, ldtype, rdtype)
    if err is not None:
        with pytest.raises(err):
            parpy.print_compiled(parpy_fn, [dst, x, y], opts)
    else:
        code = parpy.print_compiled(parpy_fn, [dst, x, y], opts)
        assert len(code) != 0
