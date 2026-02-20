import parpy
import triton
import triton.language as tl

def _parpy_builtin_to_torch(t):
    return t.torch()

def _parpy_builtin_alloc(sz, dtype):
    return parpy.buffer.empty((sz,), dtype, parpy.CompileBackend.Triton)

@triton.jit
def _parpy_builtin_not(x):
    return not x

@triton.jit
def _parpy_builtin_any(x):
    return tl.max(x) == True

@triton.jit
def _parpy_builtin_mul(x, y):
    return x * y

@triton.jit
def _parpy_builtin_prod(x):
    return tl.reduce(x, None, _parpy_builtin_mul)
