import math
import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def parir_eq(dst, a, b):
    for i in range(1):
        if a[i] == b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

@parir.jit
def parir_neq(dst, a, b):
    for i in range(1):
        if a[i] != b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

@parir.jit
def parir_leq(dst, a, b):
    for i in range(1):
        if a[i] <= b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

@parir.jit
def parir_geq(dst, a, b):
    for i in range(1):
        if a[i] >= b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

@parir.jit
def parir_lt(dst, a, b):
    for i in range(1):
        if a[i] < b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

@parir.jit
def parir_gt(dst, a, b):
    for i in range(1):
        if a[i] > b[i]:
            dst[i] = 1
        else:
            dst[i] = 0

def compare_dtype(fn, arg_dtype, compile_only):
    a = torch.randint(1, 10, (1,), dtype=arg_dtype)
    b = torch.randint(1, 10, (1,), dtype=arg_dtype)
    dst = torch.empty((1,), dtype=torch.int32)
    p = {'i': [parir.threads(32)]}
    if compile_only:
        s = parir.print_compiled(fn, [dst, a, b], p)
        assert len(s) != 0
    else:
        fn(dst, a, b)
        dst_cu = torch.empty_like(dst).cuda()
        fn(dst_cu, a.cuda(), b.cuda(), parallelize=p, cache=False)
        assert dst == dst_cu.cpu()

functions = [
    parir_eq, parir_neq, parir_leq, parir_geq, parir_lt, parir_gt
]
cmp_dtypes = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]

@pytest.mark.parametrize('fn', functions)
@pytest.mark.parametrize('dtype', cmp_dtypes)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_compare(fn, dtype):
    compare_dtype(fn, dtype, False)

@pytest.mark.parametrize('fn', functions)
@pytest.mark.parametrize('dtype', cmp_dtypes)
def test_compare_compiles(fn, dtype):
    compare_dtype(fn, dtype, True)
