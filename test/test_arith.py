import math
import parir
import pytest
import torch

from common import *

torch.manual_seed(1234)

@parir.jit
def parir_add(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] + b[0]

@parir.jit
def parir_sub(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] - b[0]

@parir.jit
def parir_mul(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] * b[0]

@parir.jit
def parir_div_int(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] // b[0]

@parir.jit
def parir_div(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] / b[0]

@parir.jit
def parir_rem(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] % b[0]

@parir.jit
def parir_pow(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] ** b[0]

@parir.jit
def parir_abs(dst, a, b):
    with parir.gpu:
        dst[0] = abs(a[0]) + abs(b[0])

@parir.jit
def parir_bit_and(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] & b[0]

@parir.jit
def parir_bit_or(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] | b[0]

@parir.jit
def parir_bit_xor(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] ^ b[0]

@parir.jit
def parir_bit_shl(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] << b[0]

@parir.jit
def parir_bit_shr(dst, a, b):
    with parir.gpu:
        dst[0] = a[0] >> b[0]

@parir.jit
def parir_aug_ops(dst, a, b):
    with parir.gpu:
        dst[0] += a[0]
        dst[0] -= b[0]

@parir.jit
def parir_max(dst, a, b):
    with parir.gpu:
        dst[0] = parir.max(a[0], b[0])

@parir.jit
def parir_min(dst, a, b):
    with parir.gpu:
        dst[0] = parir.min(a[0], b[0])

def arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend):
    a = torch.randint(1, 10, (1,), dtype=ldtype)
    b = torch.randint(1, 10, (1,), dtype=rdtype)
    dst = torch.zeros((1,), dtype=rdtype)
    if compile_only:
        s = parir.print_compiled(fn, [dst, a, b], par_opts(backend, {}))
        assert len(s) != 0
    else:
        dst_device = torch.zeros_like(dst)
        fn(dst_device, a, b, opts=par_opts(backend, {}))
        fn(dst, a, b, opts=seq_opts(backend))
        assert torch.allclose(dst, dst_device, atol=1e-5)

bitwise_funs = [
    parir_bit_and, parir_bit_or, parir_bit_xor, parir_bit_shl, parir_bit_shr
]
arith_funs = [
    parir_add, parir_sub, parir_mul, parir_div_int, parir_div, parir_rem,
    parir_pow, parir_abs, parir_aug_ops, parir_max, parir_min
] + bitwise_funs
arith_tys = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]

def is_float_dtype(dtype):
    return dtype == torch.float16 or dtype == torch.float32 or dtype == torch.float64

def is_invalid_div_or_rem_call(fn, ldtype, rdtype):
    return ((fn.__name__ == "parir_div_int" or fn.__name__ == "parir_rem") and
        (is_float_dtype(ldtype) or is_float_dtype(rdtype)))

# There is no 'pow' implementation for 16-bit floats in CUDA
def is_invalid_pow_call(fn, ldtype, rdtype):
    return (fn.__name__ == "parir_pow" and
        ((not is_float_dtype(ldtype) and not is_float_dtype(rdtype)) or
        (ldtype == torch.float16 and rdtype == torch.float16)))

def set_expected_behavior_binop(fn, ldtype, rdtype, backend):
    if is_invalid_div_or_rem_call(fn, ldtype, rdtype):
        return True, r"Invalid type .* of integer arithmetic operation"
    elif is_invalid_pow_call(fn, ldtype, rdtype):
        return True, r"Invalid type .* of floating-point arithmetic operation"
    elif fn in bitwise_funs and (is_float_dtype(ldtype) or is_float_dtype(rdtype)):
        return True, r"Invalid type .* of bitwise operation"
    elif backend == parir.CompileBackend.Metal and \
         (ldtype == torch.float64 or rdtype == torch.float64):
        return True, r"Metal does not support double-precision floating-point numbers."
    else:
        return False, None

def bin_arith_helper(fn, ldtype, rdtype, compile_only, backend):
    should_fail, msg_regex = set_expected_behavior_binop(fn, ldtype, rdtype, backend)
    if should_fail:
        with pytest.raises(TypeError) as e_info:
            arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend)
        assert e_info.match(msg_regex)
    else:
        arith_binop_dtype(fn, ldtype, rdtype, compile_only, backend)

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith(fn, dtype, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: bin_arith_helper(fn, dtype, dtype, False, backend)
    )

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith_compile(fn, dtype, backend):
    bin_arith_helper(fn, dtype, dtype, True, backend)

# All allowed pairs of types in arithmetic operations, where the LHS should be
# coerced to the RHS type.
arith_ty_pairs = [
    (torch.int8, torch.int16),
    (torch.int8, torch.int32),
    (torch.int8, torch.int64),
    (torch.int16, torch.int32),
    (torch.int16, torch.int64),
    (torch.int32, torch.int64),
    (torch.float16, torch.float32),
    (torch.float16, torch.float64),
    (torch.float32, torch.float64),
]

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtypes', arith_ty_pairs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_bin_arith_mixed_types(fn, dtypes, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: bin_arith_helper(fn, dtypes[0], dtypes[1], False, backend)
    )

@parir.jit
def parir_cos(dst, src):
    with parir.gpu:
        dst[0] = parir.cos(src[0])

@parir.jit
def parir_sin(dst, src):
    with parir.gpu:
        dst[0] = parir.sin(src[0])

@parir.jit
def parir_tanh(dst, src):
    with parir.gpu:
        dst[0] = parir.tanh(src[0])

@parir.jit
def parir_atan2(dst, src):
    with parir.gpu:
        dst[0] = parir.atan2(src[0], src[0])

@parir.jit
def parir_sqrt(dst, src):
    with parir.gpu:
        dst[0] = parir.sqrt(src[0])

def arith_unop_dtype(fn, dtype, compile_only, backend):
    src = torch.tensor([0.5], dtype=dtype)
    dst = torch.zeros_like(src)
    if compile_only:
        s = parir.print_compiled(fn, [src, dst], par_opts(backend, {}))
        assert len(s) != 0
    else:
        dst_device = torch.zeros_like(dst)
        fn(dst_device, src, opts=par_opts(backend, {}))
        fn(dst, src, opts=seq_opts(backend))
        assert torch.allclose(dst, dst_device, atol=1e-5)

float_funs = [parir_cos, parir_sin, parir_tanh, parir_atan2, parir_sqrt]
float_tys = [torch.float16, torch.float32, torch.float64]

def set_expected_behavior_unop(fn, dtype, backend):
    if fn.__name__ == "parir_tanh" and dtype == torch.float16:
        return True, "Operation tanh not supported for 16-bit floats.*"
    elif fn.__name__ == "parir_atan2" and dtype != torch.float64:
        return True, "Operation atan2 is only supported for 64-bit floats.*"
    elif backend == parir.CompileBackend.Metal and dtype == torch.float64:
        return True, r"Metal does not support double-precision floating-point numbers."
    else:
        return False, None

def float_unop_helper(fn, dtype, compile_only, backend):
    should_fail, msg_regex = set_expected_behavior_unop(fn, dtype, backend)
    if should_fail:
        with pytest.raises(TypeError) as e_info:
            arith_unop_dtype(fn, dtype, compile_only, backend)
        assert e_info.match(msg_regex)
    else:
        arith_unop_dtype(fn, dtype, compile_only, backend)

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_float_unop_arith(fn, dtype, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: float_unop_helper(fn, dtype, False, backend)
    )

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
@pytest.mark.parametrize('backend', compiler_backends)
def test_float_unop_arith_compile(fn, dtype, backend):
    float_unop_helper(fn, dtype, True, backend)
