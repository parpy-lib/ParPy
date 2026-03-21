import parpy
import triton
import triton.language as tl
from triton.language.extra import libdevice as libdevice

def _merge_attrs(lattrs, rattrs):
    common_keys = lattrs.keys() & rattrs.keys()
    return {k: lattrs[k] for k in common_keys}

def _parpy_builtin_launch_kernel(attrs, kernel, bx, by, bz, *args):
    compiled_fn = kernel[lambda _: (bx, by, bz)](*args)

    # Track the attributes associated with each function, to be used when we
    # compile it.
    name = compiled_fn.name
    if not name in attrs:
        attrs[name] = compiled_fn.src.attrs
    else:
        attrs[name] = _merge_attrs(attrs[name], compiled_fn.src.attrs)
    return attrs

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
