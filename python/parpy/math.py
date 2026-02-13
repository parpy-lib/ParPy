from .main import external, jit
from .builtin import static_backend_eq, static_types_eq, static_fail
from .parpy import CompileBackend, Target
from .types import *

import math

# Unary function definitions

T = type_var()

@external("abs", CompileBackend.Cuda, Target.Device)
def abs_cuda_i8(x: I8) -> I8:
    return abs(x)

@external("abs", CompileBackend.Cuda, Target.Device)
def abs_cuda_i16(x: I16) -> I16:
    return abs(x)

@external("abs", CompileBackend.Cuda, Target.Device)
def abs_cuda_i32(x: I32) -> I32:
    return abs(x)

@external("abs", CompileBackend.Cuda, Target.Device)
def abs_cuda_i64(x: I64) -> I64:
    return abs(x)

@external("__habs", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def abs_cuda_f16(x: F16) -> F16:
    return abs(x)

@external("fabsf", CompileBackend.Cuda, Target.Device)
def abs_cuda_f32(x: F32) -> F32:
    return abs(x)

@external("fabs", CompileBackend.Cuda, Target.Device)
def abs_cuda_f64(x: F64) -> F64:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_i8(x: I8) -> I8:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_i16(x: I16) -> I16:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_i32(x: I32) -> I32:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_i64(x: I64) -> I64:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_f16(x: F16) -> F16:
    return abs(x)

@external("metal::abs", CompileBackend.Metal, Target.Device)
def abs_metal_f32(x: F32) -> F32:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_i8(x: I8) -> I8:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_i16(x: I16) -> I16:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_i32(x: I32) -> I32:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_i64(x: I64) -> I64:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_f16(x: F16) -> F16:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_f32(x: F32) -> F32:
    return abs(x)

@external("triton.language.abs", CompileBackend.Triton, Target.Device)
def abs_triton_f64(x: F64) -> F64:
    return abs(x)

@jit
def abs(x: T) -> T:
    """
    Implementation of the absolute function in CUDA and Metal using externals.

    When given an unsigned argument type, we immediately return the value as
    is. Otherwise, we select which external function to call based on the
    backend and the type of the argument.
    """
    if static_types_eq(T, U8) or static_types_eq(T, U16) or \
            static_types_eq(T, U32) or static_types_eq(T, U64):
        return x
    elif static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, I8):
            return abs_cuda_i8(x)
        elif static_types_eq(T, I16):
            return abs_cuda_i16(x)
        elif static_types_eq(T, I32):
            return abs_cuda_i32(x)
        elif static_types_eq(T, I64):
            return abs_cuda_i64(x)
        elif static_types_eq(T, F16):
            return abs_cuda_f16(x)
        elif static_types_eq(T, F32):
            return abs_cuda_f32(x)
        elif static_types_eq(T, F64):
            return abs_cuda_f64(x)
        else:
            static_fail("The abs function only supports int and float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, I8):
            return abs_metal_i8(x)
        elif static_types_eq(T, I16):
            return abs_metal_i16(x)
        elif static_types_eq(T, I32):
            return abs_metal_i32(x)
        elif static_types_eq(T, I64):
            return abs_metal_i64(x)
        elif static_types_eq(T, F16):
            return abs_metal_f16(x)
        elif static_types_eq(T, F32):
            return abs_metal_f32(x)
        else:
            static_fail("The abs function only supports int and float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, I8):
            return abs_triton_i8(x)
        elif static_types_eq(T, I16):
            return abs_triton_i16(x)
        elif static_types_eq(T, I32):
            return abs_triton_i32(x)
        elif static_types_eq(T, I64):
            return abs_triton_i64(x)
        elif static_types_eq(T, F16):
            return abs_triton_f16(x)
        elif static_types_eq(T, F32):
            return abs_triton_f32(x)
        elif static_types_eq(T, F64):
            return abs_triton_f64(x)
        else:
            static_fail("The abs function only supports int and float scalars")
    else:
        static_fail("The abs function only supports the CUDA, Metal and Triton backends")

@external("hcos", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def cos_cuda_f16(x: F16) -> F16:
    return math.cos(x)

@external("__cosf", CompileBackend.Cuda, Target.Device)
def cos_cuda_f32(x: F32) -> F32:
    return math.cos(x)

@external("cos", CompileBackend.Cuda, Target.Device)
def cos_cuda_f64(x: F64) -> F64:
    return math.cos(x)

@external("metal::cos", CompileBackend.Metal, Target.Device)
def cos_metal_f16(x: F16) -> F16:
    return math.cos(x)

@external("metal::cos", CompileBackend.Metal, Target.Device)
def cos_metal_f32(x: F32) -> F32:
    return math.cos(x)

@external("triton.language.cos", CompileBackend.Triton, Target.Device)
def cos_triton_f32(x: F32) -> F32:
    return math.cos(x)

@external("triton.language.cos", CompileBackend.Triton, Target.Device)
def cos_triton_f64(x: F64) -> F64:
    return math.cos(x)

@jit
def cos(x: T) -> T:
    """
    Implementation of the cosine function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return cos_cuda_f16(x)
        elif static_types_eq(T, F32):
            return cos_cuda_f32(x)
        elif static_types_eq(T, F64):
            return cos_cuda_f64(x)
        else:
            static_fail("The cos function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return cos_metal_f16(x)
        elif static_types_eq(T, F32):
            return cos_metal_f32(x)
        else:
            static_fail("The cos function only supports float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, F16):
            static_fail("The cos function does not support 16-bit floats in Triton")
        elif static_types_eq(T, F32):
            return cos_triton_f32(x)
        elif static_types_eq(T, F64):
            return cos_triton_f64(x)
        else:
            static_fail("The cos function only supports float scalars")
    else:
        static_fail("The cos function only supports the CUDA, Metal and Triton backends")

@external("hexp", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def exp_cuda_f16(x: F16) -> F16:
    return math.exp(x)

@external("__expf", CompileBackend.Cuda, Target.Device)
def exp_cuda_f32(x: F32) -> F32:
    return math.exp(x)

@external("exp", CompileBackend.Cuda, Target.Device)
def exp_cuda_f64(x: F64) -> F64:
    return math.exp(x)

@external("metal::exp", CompileBackend.Metal, Target.Device)
def exp_metal_f16(x: F16) -> F16:
    return math.exp(x)

@external("metal::exp", CompileBackend.Metal, Target.Device)
def exp_metal_f32(x: F32) -> F32:
    return math.exp(x)

@external("triton.language.exp", CompileBackend.Triton, Target.Device)
def exp_triton_f32(x: F32) -> F32:
    return math.exp(x)

@external("triton.language.exp", CompileBackend.Triton, Target.Device)
def exp_triton_f64(x: F64) -> F64:
    return math.exp(x)

@jit
def exp(x: T) -> T:
    """
    Implementation of the exponentiation function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return exp_cuda_f16(x)
        elif static_types_eq(T, F32):
            return exp_cuda_f32(x)
        elif static_types_eq(T, F64):
            return exp_cuda_f64(x)
        else:
            static_fail("The exp function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return exp_metal_f16(x)
        elif static_types_eq(T, F32):
            return exp_metal_f32(x)
        else:
            static_fail("The exp function only supports float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, F16):
            static_fail("The exp function does not support 16-bit floats in Triton")
        elif static_types_eq(T, F32):
            return exp_triton_f32(x)
        elif static_types_eq(T, F64):
            return exp_triton_f64(x)
        else:
            static_fail("The exp function only supports float scalars")
    else:
        static_fail("The exp function only supports the CUDA, Metal and Triton backends")

@external("hlog", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def log_cuda_f16(x: F16) -> F16:
    return math.log(x)

@external("__logf", CompileBackend.Cuda, Target.Device)
def log_cuda_f32(x: F32) -> F32:
    return math.log(x)

@external("log", CompileBackend.Cuda, Target.Device)
def log_cuda_f64(x: F64) -> F64:
    return math.log(x)

@external("metal::log", CompileBackend.Metal, Target.Device)
def log_metal_f16(x: F16) -> F16:
    return math.log(x)

@external("metal::log", CompileBackend.Metal, Target.Device)
def log_metal_f32(x: F32) -> F32:
    return math.log(x)

@external("triton.language.log", CompileBackend.Triton, Target.Device)
def log_triton_f32(x: F32) -> F32:
    return math.log(x)

@external("triton.language.log", CompileBackend.Triton, Target.Device)
def log_triton_f64(x: F64) -> F64:
    return math.log(x)

@jit
def log(x: T) -> T:
    """
    Implementation of the natural logarithm function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return log_cuda_f16(x)
        elif static_types_eq(T, F32):
            return log_cuda_f32(x)
        elif static_types_eq(T, F64):
            return log_cuda_f64(x)
        else:
            static_fail("The log function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return log_metal_f16(x)
        elif static_types_eq(T, F32):
            return log_metal_f32(x)
        else:
            static_fail("The log function only supports float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, F16):
            static_fail("The log function does not support 16-bit floats in Triton")
        elif static_types_eq(T, F32):
            return log_triton_f32(x)
        elif static_types_eq(T, F64):
            return log_triton_f64(x)
        else:
            static_fail("The log function only supports float scalars")
    else:
        static_fail("The log function only supports the CUDA, Metal and Triton backends")

@external("hsin", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def sin_cuda_f16(x: F16) -> F16:
    return math.sin(x)

@external("__sinf", CompileBackend.Cuda, Target.Device)
def sin_cuda_f32(x: F32) -> F32:
    return math.sin(x)

@external("sin", CompileBackend.Cuda, Target.Device)
def sin_cuda_f64(x: F64) -> F64:
    return math.sin(x)

@external("metal::sin", CompileBackend.Metal, Target.Device)
def sin_metal_f16(x: F16) -> F16:
    return math.sin(x)

@external("metal::sin", CompileBackend.Metal, Target.Device)
def sin_metal_f32(x: F32) -> F32:
    return math.sin(x)

@external("triton.language.sin", CompileBackend.Triton, Target.Device)
def sin_triton_f32(x: F32) -> F32:
    return math.sin(x)

@external("triton.language.sin", CompileBackend.Triton, Target.Device)
def sin_triton_f64(x: F64) -> F64:
    return math.sin(x)

@jit
def sin(x: T) -> T:
    """
    Implementation of the sine function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return sin_cuda_f16(x)
        elif static_types_eq(T, F32):
            return sin_cuda_f32(x)
        elif static_types_eq(T, F64):
            return sin_cuda_f64(x)
        else:
            static_fail("The sin function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return sin_metal_f16(x)
        elif static_types_eq(T, F32):
            return sin_metal_f32(x)
        else:
            static_fail("The sin function only supports float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, F16):
            static_fail("The sin function does not support 16-bit floats in Triton")
        elif static_types_eq(T, F32):
            return sin_triton_f32(x)
        elif static_types_eq(T, F64):
            return sin_triton_f64(x)
        else:
            static_fail("The sin function only supports float scalars")
    else:
        static_fail("The sin function only supports the CUDA, Metal and Triton backends")

@external("hsqrt", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def sqrt_cuda_f16(x: F16) -> F16:
    return math.sqrt(x)

@external("sqrtf", CompileBackend.Cuda, Target.Device)
def sqrt_cuda_f32(x: F32) -> F32:
    return math.sqrt(x)

@external("sqrt", CompileBackend.Cuda, Target.Device)
def sqrt_cuda_f64(x: F64) -> F64:
    return math.sqrt(x)

@external("metal::sqrt", CompileBackend.Metal, Target.Device)
def sqrt_metal_f16(x: F16) -> F16:
    return math.sqrt(x)

@external("metal::sqrt", CompileBackend.Metal, Target.Device)
def sqrt_metal_f32(x: F32) -> F32:
    return math.sqrt(x)

@external("triton.language.sqrt", CompileBackend.Triton, Target.Device)
def sqrt_triton_f32(x: F32) -> F32:
    return math.sqrt(x)

@external("triton.language.sqrt", CompileBackend.Triton, Target.Device)
def sqrt_triton_f64(x: F64) -> F64:
    return math.sqrt(x)

@jit
def sqrt(x: T) -> T:
    """
    Implementation of the square root function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return sqrt_cuda_f16(x)
        elif static_types_eq(T, F32):
            return sqrt_cuda_f32(x)
        elif static_types_eq(T, F64):
            return sqrt_cuda_f64(x)
        else:
            static_fail("The sqrt function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return sqrt_metal_f16(x)
        elif static_types_eq(T, F32):
            return sqrt_metal_f32(x)
        else:
            static_fail("The sqrt function only supports float scalars")
    elif static_backend_eq(CompileBackend.Triton):
        if static_types_eq(T, F16):
            static_fail("The sqrt function does not support 16-bit floats in Triton")
        elif static_types_eq(T, F32):
            return sqrt_triton_f32(x)
        elif static_types_eq(T, F64):
            return sqrt_triton_f64(x)
        else:
            static_fail("The sqrt function only supports float scalars")
    else:
        static_fail("The sqrt function only supports the CUDA, Metal and Triton backends")

@external("htanh", CompileBackend.Cuda, Target.Device, header="<cuda_fp16.h>")
def tanh_cuda_f16(x: F16) -> F16:
    return math.tanh(x)

@external("tanhf", CompileBackend.Cuda, Target.Device)
def tanh_cuda_f32(x: F32) -> F32:
    return math.tanh(x)

@external("tanh", CompileBackend.Cuda, Target.Device)
def tanh_cuda_f64(x: F64) -> F64:
    return math.tanh(x)

@external("metal::tanh", CompileBackend.Metal, Target.Device)
def tanh_metal_f16(x: F16) -> F16:
    return math.tanh(x)

@external("metal::tanh", CompileBackend.Metal, Target.Device)
def tanh_metal_f32(x: F32) -> F32:
    return math.tanh(x)

@jit
def tanh(x: T) -> T:
    """
    Implementation of the tanh function in CUDA and Metal.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            return tanh_cuda_f16(x)
        elif static_types_eq(T, F32):
            return tanh_cuda_f32(x)
        elif static_types_eq(T, F64):
            return tanh_cuda_f64(x)
        else:
            static_fail("The tanh function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return tanh_metal_f16(x)
        elif static_types_eq(T, F32):
            return tanh_metal_f32(x)
        else:
            static_fail("The tanh function only supports float scalars")
    else:
        static_fail("The tanh function only supports the CUDA and Metal backends")

# Binary functions

@external("atan2f", CompileBackend.Cuda, Target.Device)
def atan2_cuda_f32(y: F32, x: F32) -> F32:
    return math.atan2(y, x)

@external("atan2", CompileBackend.Cuda, Target.Device)
def atan2_cuda_f64(y: F64, x: F64) -> F64:
    return math.atan2(y, x)

@external("metal::atan2", CompileBackend.Metal, Target.Device)
def atan2_metal_f16(y: F16, x: F16) -> F16:
    return math.atan2(y, x)

@external("metal::atan2", CompileBackend.Metal, Target.Device)
def atan2_metal_f32(y: F32, x: F32) -> F32:
    return math.atan2(y, x)

@jit
def atan2(y: T, x: T) -> T:
    """
    Implementation of atan2 in CUDA and Metal. This function computes the arc
    tangent of (y / x), using the sign of the result to determine the correct
    quadrant.
    """
    if static_backend_eq(CompileBackend.Cuda):
        if static_types_eq(T, F16):
            static_fail("The atan2 function is not supported for 16-bit floats in CUDA")
        elif static_types_eq(T, F32):
            return atan2_cuda_f32(y, x)
        elif static_types_eq(T, F64):
            return atan2_cuda_f64(y, x)
        else:
            static_fail("The atan2 function only supports float scalars")
    elif static_backend_eq(CompileBackend.Metal):
        if static_types_eq(T, F16):
            return atan2_metal_f16(y, x)
        elif static_types_eq(T, F32):
            return atan2_metal_f32(y, x)
        else:
            static_fail("The atan2 function only supports float scalars")
    else:
        static_fail("The atan2 function only supports the CUDA and Metal backends")
