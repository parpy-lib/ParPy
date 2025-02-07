import math
import parir
from parir import ParKind
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def parir_cos(dst, src):
    for i in range(1):
        dst[i] = parir.cos(src[i])

@parir.jit
def parir_sin(dst, src):
    for i in range(1):
        dst[i] = parir.sin(src[i])

@parir.jit
def parir_tanh(dst, src):
    for i in range(1):
        dst[i] = parir.tanh(src[i])

@parir.jit
def parir_atan2(dst, src):
    for i in range(1):
        dst[i] = parir.atan2(src[i], 1.0)

@parir.jit
def parir_sqrt(dst, src):
    for i in range(1):
        dst[i] = parir.sqrt(src[i])

@parir.jit
def parir_pow(dst, src):
    for i in range(1):
        dst[i] = src[i] ** 2.0

def arith_dtype(fn, dtype, compile_only):
    src = torch.tensor([0.5], dtype=dtype)
    dst = torch.empty_like(src)
    p = {'i': [ParKind.GpuThreads(32)]}
    if compile_only:
        s = parir.print_compiled(fn, [src, dst], p)
        assert len(s) != 0
    else:
        # Compare result running sequentially in Python to parallelized version
        # running on the GPU.
        fn(dst, src)
        dst_cu = torch.empty_like(src).cuda()
        fn(dst_cu, src.cuda(), parallelize=p)
        assert torch.allclose(dst, dst_cu.cpu(), atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_cos_f16():
    arith_dtype(parir_cos, torch.float16, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_cos_f32():
    arith_dtype(parir_cos, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_cos_f64():
    arith_dtype(parir_cos, torch.float64, False)

def test_cos_compiles_f16():
    arith_dtype(parir_cos, torch.float16, True)

def test_cos_compiles_f32():
    arith_dtype(parir_cos, torch.float32, True)

def test_cos_compiles_f64():
    arith_dtype(parir_cos, torch.float64, True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sin_f16():
    arith_dtype(parir_sin, torch.float16, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sin_f32():
    arith_dtype(parir_sin, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sin_f64():
    arith_dtype(parir_sin, torch.float64, False)

def test_sin_compiles_f16():
    arith_dtype(parir_sin, torch.float16, True)

def test_sin_compiles_f32():
    arith_dtype(parir_sin, torch.float32, True)

def test_sin_compiles_f64():
    arith_dtype(parir_sin, torch.float64, True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_tanh_f32():
    arith_dtype(parir_tanh, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_tanh_f64():
    arith_dtype(parir_tanh, torch.float64, False)

def test_tanh_compiles_f16():
    with pytest.raises(TypeError):
        arith_dtype(parir_tanh, torch.float16, True)

def test_tanh_compiles_f32():
    arith_dtype(parir_tanh, torch.float32, True)

def test_tanh_compiles_f64():
    arith_dtype(parir_tanh, torch.float64, True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_atan2_f16():
    arith_dtype(parir_atan2, torch.float16, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_atan2_f32():
    arith_dtype(parir_atan2, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_atan2_f64():
    arith_dtype(parir_atan2, torch.float64, False)

def test_atan2_compiles_f16():
    arith_dtype(parir_atan2, torch.float16, True)

def test_atan2_compiles_f32():
    arith_dtype(parir_atan2, torch.float32, True)

def test_atan2_compiles_f64():
    arith_dtype(parir_atan2, torch.float64, True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sqrt_f16():
    arith_dtype(parir_sqrt, torch.float16, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sqrt_f32():
    arith_dtype(parir_sqrt, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_sqrt_f64():
    arith_dtype(parir_sqrt, torch.float64, False)

def test_sqrt_compiles_f16():
    arith_dtype(parir_sqrt, torch.float16, True)

def test_sqrt_compiles_f32():
    arith_dtype(parir_sqrt, torch.float32, True)

def test_sqrt_compiles_f64():
    arith_dtype(parir_sqrt, torch.float64, True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_pow_f16():
    arith_dtype(parir_pow, torch.float16, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_pow_f32():
    arith_dtype(parir_pow, torch.float32, False)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_pow_f64():
    arith_dtype(parir_pow, torch.float64, False)

def test_pow_compiles_f16():
    arith_dtype(parir_pow, torch.float16, True)

def test_pow_compiles_f32():
    arith_dtype(parir_pow, torch.float32, True)

def test_pow_compiles_f64():
    arith_dtype(parir_pow, torch.float64, True)

