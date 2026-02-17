import triton
import triton.language as tl

def _parpy_builtin_to_torch(t):
    return t.torch()

@triton.jit
def _parpy_builtin_any(x):
    return tl.max(x) == True

@triton.jit
def _parpy_builtin_mul(x, y):
    return x * y

@triton.jit
def _parpy_builtin_prod(x):
    return tl.reduce(x, None, _parpy_builtin_mul)
