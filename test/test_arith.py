import math
import parir
import pytest
import torch

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
        dst[0] = max(a[0], b[0])

@parir.jit
def parir_min(dst, a, b):
    with parir.gpu:
        dst[0] = min(a[0], b[0])

def arith_binop_dtype(fn, ldtype, rdtype, compile_only):
    a = torch.randint(1, 10, (1,), dtype=ldtype)
    b = torch.randint(1, 10, (1,), dtype=rdtype)
    dst = torch.zeros((1,), dtype=rdtype)
    if compile_only:
        s = parir.print_compiled(fn, [dst, a, b])
        assert len(s) != 0
    else:
        dst_cu = torch.zeros_like(dst).cuda()
        fn(dst_cu, a.cuda(), b.cuda(), cache=False)
        fn(dst, a, b, seq=True)
        assert torch.allclose(dst, dst_cu.cpu(), atol=1e-5)

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

def bin_arith_helper(fn, ldtype, rdtype, compile_only):
    if (is_invalid_div_or_rem_call(fn, ldtype, rdtype) or
        is_invalid_pow_call(fn, ldtype, rdtype) or
        (fn in bitwise_funs and (is_float_dtype(ldtype) or is_float_dtype(rdtype)))):
        with pytest.raises(TypeError):
            arith_binop_dtype(fn, ldtype, rdtype, compile_only)
    else:
        arith_binop_dtype(fn, ldtype, rdtype, compile_only)

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_bin_arith(fn, dtype):
    bin_arith_helper(fn, dtype, dtype, False)

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
def test_bin_arith_compile(fn, dtype):
    bin_arith_helper(fn, dtype, dtype, True)

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_bin_arith_mixed_types(fn, dtypes):
    bin_arith_helper(fn, dtypes[0], dtypes[1], False)

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
        dst[0] = parir.atan2(src[0], 1.0)

@parir.jit
def parir_sqrt(dst, src):
    with parir.gpu:
        dst[0] = parir.sqrt(src[0])

def arith_unop_dtype(fn, dtype, compile_only):
    src = torch.tensor([0.5], dtype=dtype)
    dst = torch.empty_like(src)
    if compile_only:
        s = parir.print_compiled(fn, [src, dst])
        assert len(s) != 0
    else:
        dst_cu = torch.empty_like(dst).cuda()
        fn(dst_cu, src.cuda(), cache=False)
        fn(dst, src, seq=True)
        assert torch.allclose(dst, dst_cu.cpu(), atol=1e-5)

float_funs = [parir_cos, parir_sin, parir_tanh, parir_atan2, parir_sqrt]
float_tys = [torch.float16, torch.float32, torch.float64]

def float_unop_helper(fn, dtype, compile_only):
    if fn.__name__ == "parir_tanh" and dtype == torch.float16:
        with pytest.raises(TypeError):
            arith_unop_dtype(fn, dtype, compile_only)
    else:
        arith_unop_dtype(fn, dtype, compile_only)

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_float_unop_arith(fn, dtype):
    float_unop_helper(fn, dtype, False)

@pytest.mark.parametrize('fn', float_funs)
@pytest.mark.parametrize('dtype', float_tys)
def test_float_unop_arith_compile(fn, dtype):
    float_unop_helper(fn, dtype, True)
