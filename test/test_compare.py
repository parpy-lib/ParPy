import math
import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

@parpy.jit
def parpy_eq(dst, a, b):
    with parpy.gpu:
        if a[0] == b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

@parpy.jit
def parpy_neq(dst, a, b):
    with parpy.gpu:
        if a[0] != b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

@parpy.jit
def parpy_leq(dst, a, b):
    with parpy.gpu:
        if a[0] <= b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

@parpy.jit
def parpy_geq(dst, a, b):
    with parpy.gpu:
        if a[0] >= b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

@parpy.jit
def parpy_lt(dst, a, b):
    with parpy.gpu:
        if a[0] < b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

@parpy.jit
def parpy_gt(dst, a, b):
    with parpy.gpu:
        if a[0] > b[0]:
            dst[0] = 1
        else:
            dst[0] = 0

def set_expected_behavior(dtype, backend):
    if dtype == torch.float64 and backend == parpy.CompileBackend.Metal:
        return True, r"Metal does not support double-precision floating-point numbers."
    else:
        return False, None

def compare_dtype_helper(fn, dtype, backend, compile_only):
    a = torch.randint(1, 10, (1,), dtype=dtype)
    b = torch.randint(1, 10, (1,), dtype=dtype)
    dst = torch.zeros((1,), dtype=torch.int32)
    if compile_only:
        s = parpy.print_compiled(fn, [dst, a, b], par_opts(backend, {}))
        assert len(s) != 0
    else:
        dst_device = torch.zeros_like(dst)
        fn(dst_device, a, b, opts=par_opts(backend, {}))
        fn(dst, a, b, opts=seq_opts(backend))
        assert dst == dst_device

def compare_dtype(fn, arg_dtype, backend, compile_only):
    should_fail, msg_regex = set_expected_behavior(arg_dtype, backend)
    if should_fail:
        with pytest.raises(TypeError) as e_info:
            compare_dtype_helper(fn, arg_dtype, backend, compile_only)
        assert e_info.match(msg_regex)
    else:
        compare_dtype_helper(fn, arg_dtype, backend, compile_only)

functions = [
    parpy_eq, parpy_neq, parpy_leq, parpy_geq, parpy_lt, parpy_gt
]
cmp_dtypes = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]

@pytest.mark.parametrize('fn', functions)
@pytest.mark.parametrize('dtype', cmp_dtypes)
@pytest.mark.parametrize('backend', compiler_backends)
def test_compare(fn, dtype, backend):
    run_if_backend_is_enabled(
        backend,
        lambda: compare_dtype(fn, dtype, backend, False)
    )

@pytest.mark.parametrize('fn', functions)
@pytest.mark.parametrize('dtype', cmp_dtypes)
@pytest.mark.parametrize('backend', compiler_backends)
def test_compare_compiles(fn, dtype, backend):
    compare_dtype(fn, dtype, backend, True)
