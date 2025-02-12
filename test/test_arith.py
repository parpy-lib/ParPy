import math
import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def parir_add(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] + b[i]

@parir.jit
def parir_sub(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] - b[i]

@parir.jit
def parir_mul(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] * b[i]

@parir.jit
def parir_div_int(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] // b[i]

@parir.jit
def parir_div(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] / b[i]

@parir.jit
def parir_rem(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] % b[i]

@parir.jit
def parir_pow(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] ** b[i]

@parir.jit
def parir_abs(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = abs(a[i]) + abs(b[i])

@parir.jit
def parir_bit_and(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] & b[i]

@parir.jit
def parir_bit_or(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] | b[i]

@parir.jit
def parir_bit_xor(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] ^ b[i]

@parir.jit
def parir_bit_shl(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] << b[i]

@parir.jit
def parir_bit_shr(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] = a[i] >> b[i]

@parir.jit
def parir_aug_ops(dst, a, b):
    parir.label('i')
    for i in range(1):
        dst[i] += a[i]
        dst[i] -= b[i]

def arith_binop_dtype(fn, dtype, compile_only):
    a = torch.randint(1, 10, (1,), dtype=dtype)
    b = torch.randint(1, 10, (1,), dtype=dtype)
    dst = torch.zeros((1,), dtype=dtype)
    p = {'i': [parir.threads(32)]}
    if compile_only:
        s = parir.print_compiled(fn, [dst, a, b], p)
        assert len(s) != 0
    else:
        dst_cu = torch.zeros_like(dst).cuda()
        fn(dst_cu, a.cuda(), b.cuda(), parallelize=p, cache=False)
        fn(dst, a, b)
        assert torch.allclose(dst, dst_cu.cpu(), atol=1e-5)

bitwise_funs = [
    parir_bit_and, parir_bit_or, parir_bit_xor, parir_bit_shl, parir_bit_shr
]
arith_funs = [
    parir_add, parir_sub, parir_mul, parir_div_int, parir_div, parir_rem,
    parir_pow, parir_abs, parir_aug_ops
] + bitwise_funs
arith_tys = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]

def is_float_dtype(dtype):
    return dtype == torch.float16 or dtype == torch.float32 or dtype == torch.float64

def bin_arith_helper(fn, dtype, compile_only):
    if (((fn.__name__ == "parir_div_int" or fn.__name__ == "parir_rem")
             and is_float_dtype(dtype)) or
         (fn.__name__ == "parir_pow" and
             not (dtype == torch.float32 or dtype == torch.float64)) or
         (fn in bitwise_funs and is_float_dtype(dtype))):
        with pytest.raises(TypeError):
            arith_binop_dtype(fn, dtype, compile_only)
    else:
        arith_binop_dtype(fn, dtype, compile_only)

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_bin_arith(fn, dtype):
    bin_arith_helper(fn, dtype, False)

@pytest.mark.parametrize('fn', arith_funs)
@pytest.mark.parametrize('dtype', arith_tys)
def test_bin_arith_compile(fn, dtype):
    bin_arith_helper(fn, dtype, True)

@parir.jit
def parir_cos(dst, src):
    parir.label('i')
    for i in range(1):
        dst[i] = parir.cos(src[i])

@parir.jit
def parir_sin(dst, src):
    parir.label('i')
    for i in range(1):
        dst[i] = parir.sin(src[i])

@parir.jit
def parir_tanh(dst, src):
    parir.label('i')
    for i in range(1):
        dst[i] = parir.tanh(src[i])

@parir.jit
def parir_atan2(dst, src):
    parir.label('i')
    for i in range(1):
        dst[i] = parir.atan2(src[i], 1.0)

@parir.jit
def parir_sqrt(dst, src):
    parir.label('i')
    for i in range(1):
        dst[i] = parir.sqrt(src[i])

def arith_unop_dtype(fn, dtype, compile_only):
    src = torch.tensor([0.5], dtype=dtype)
    dst = torch.empty_like(src)
    p = {'i': [parir.threads(32)]}
    if compile_only:
        s = parir.print_compiled(fn, [src, dst], p)
        assert len(s) != 0
    else:
        # Compare result running sequentially in Python to parallelized version
        # running on the GPU.
        fn(dst, src)
        dst_cu = torch.empty_like(src).cuda()
        fn(dst_cu, src.cuda(), parallelize=p, cache=False)
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
