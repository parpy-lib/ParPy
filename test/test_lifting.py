import numpy as np
import parpy
from parpy.types import *
import pytest
import re

from common import *

N = shape_var()
M = shape_var()
K = shape_var()

@parpy.jit
def seq_updates(
    x: buffer(F32, [N, K]),
    mul: buffer(F32, [M]),
    add: buffer(F32, [M])
):
    for i in range(N):
        for j in range(M):
            for k in range(K):
                x[i,k] = (x[i,k] + add[j]) * mul[j]
        for j in range(M):
            a = 1.0
            t = parpy.reduce.sum(x[i,:K] + a)
            for k in range(K):
                x[i,k] = x[i,k] + add[j] / t

@pytest.mark.parametrize('backend', compiler_backends)
def test_seq_updates(backend):
    def helper():
        np.random.seed(1234)
        N, M, K = 10, 10, 1024
        x = np.random.randn(N, K).astype(np.float32)
        x2 = x.copy()
        mul = np.random.randn(M).astype(np.float32)
        add = np.random.randn(M).astype(np.float32)

        p1 = {'N': parpy.threads(N)}
        seq_updates(x, mul, add, opts=par_opts(backend, p1))

        p2 = {
            'N': parpy.threads(N),
            'K': parpy.threads(K).tpb(32)
        }
        seq_updates(x2, mul, add, opts=par_opts(backend, p2))

        assert np.allclose(x, x2, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

def count_kernels(backend, code):
    if backend == parpy.CompileBackend.Cuda:
        return len(re.findall("<<<.*>>>", code))
    elif backend == parpy.CompileBackend.Metal:
        return len(re.findall("parpy_metal::launch_kernel", code))
    elif backend == parpy.CompileBackend.Triton:
        return len(re.findall("_parpy_builtin_launch_kernel", code))
    else:
        raise RuntimeError(f"Unsupported backend {backend}")

@pytest.mark.parametrize('backend', compiler_backends)
def test_kernel_count_seq_updates(backend):
    N, M, K = 10, 10, 1024
    x = np.random.randn(N, K).astype(np.float32)
    mul = np.random.randn(M).astype(np.float32)
    add = np.random.randn(M).astype(np.float32)
    p = {'N': parpy.threads(N), 'K': parpy.threads(K).tpb(32)}
    code = parpy.print_compiled(seq_updates, [x, mul, add], par_opts(backend, p))
    # We expect a total of four kernels:
    # - One kernel for the first for-loop nest
    # - Two kernels for the summation because it is an inter-block reduction
    # - One kernel for the final loop over k
    assert count_kernels(backend, code) == 4

A = shape_var()
B = shape_var()
C = shape_var()
D = shape_var()
E = shape_var()

@parpy.jit
def multiple_seq_loops(
    x: buffer(F32, [A, B, E]),
    add: buffer(F32, [C]),
    mul: buffer(F32, [D]),
):
    for a in range(A):
        for b in range(B):
            for c in range(C):
                for d in range(D):
                    for e in range(E):
                        x[a,b,e] = (x[a,b,e] + add[c]) * mul[d]

@pytest.mark.parametrize('backend', compiler_backends)
def test_multiple_seq_loops_exec(backend):
    def helper():
        np.random.seed(1234)
        A, B, C, D, E = 10, 10, 10, 10, 64
        x = np.random.randn(A, B, E).astype(np.float32)
        x2 = x.copy()
        x3 = x.copy()
        add = np.random.randn(C).astype(np.float32)
        mul = np.random.randn(D).astype(np.float32)
        p1 = {'A': parpy.threads(A)}
        multiple_seq_loops(x, add, mul, opts=par_opts(backend, p1))
        p2 = {'A': parpy.threads(A), 'B': parpy.threads(B)}
        multiple_seq_loops(x2, add, mul, opts=par_opts(backend, p2))
        p3 = {
            'A': parpy.threads(A),
            'B': parpy.threads(B),
            'E': parpy.threads(E).tpb(32)
        }
        multiple_seq_loops(x3, add, mul, opts=par_opts(backend, p3))
        assert np.allclose(x, x2, atol=1e-5)
        assert np.allclose(x, x3, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

def seq_loops_list(code):
    p1 = r"extern \"C\".*"
    entry_point_code = re.findall(p1, code, re.DOTALL)[0]
    p2 = r"for \(int32_t (\w) = .*;"
    return re.findall(p2, entry_point_code)

def test_multiple_seq_loops_ordering_cuda():
    A, B, C, D, E = 10, 10, 10, 10, 64
    x = np.random.randn(A, B, E).astype(np.float32)
    add = np.random.randn(C).astype(np.float32)
    mul = np.random.randn(D).astype(np.float32)

    # If we make 'a' sequential, it should be first in the list, followed by
    # the 'c' and 'd' loops, which should be lifted outside of the parallel
    # loop over 'b'.
    p = {
        'B': parpy.threads(B),
        'E': parpy.threads(E).tpb(32)
    }
    opts = par_opts(parpy.CompileBackend.Cuda, p)
    code = parpy.print_compiled(multiple_seq_loops, [x, add, mul], opts)
    assert seq_loops_list(code) == ["a", "c", "d"]

    # If we parallelize 'a', the 'c' and 'd' loops should be lifted outside of
    # it as well.
    p['A'] = parpy.threads(A)
    opts = par_opts(parpy.CompileBackend.Cuda, p)
    code = parpy.print_compiled(multiple_seq_loops, [x, add, mul], opts)
    assert seq_loops_list(code) == ["c", "d"]

    # If the 'E' loop is not parallelized over multiple blocks, the sequential
    # loop lifting is not needed.
    p['E'] = parpy.threads(E)
    opts = par_opts(parpy.CompileBackend.Cuda, p)
    code = parpy.print_compiled(multiple_seq_loops, [x, add, mul], opts)
    assert seq_loops_list(code) == []

@parpy.jit
def repeated_seq_loops(
    x: buffer(F32, [A, D]),
    add: buffer(F32, [B]),
    mul: buffer(F32, [C])
):
    for a in range(A):
        for b in range(B):
            for c in range(C):
                x[a,:D] = (x[a,:D] + add[b]) * mul[c]
            for c in range(C):
                t = parpy.reduce.sum(x[a,:D])
                x[a,:D] /= t * mul[c]
        for b in range(B):
            x[a,:D] -= add[b]

@pytest.mark.parametrize('backend', compiler_backends)
def test_repeated_seq_loops_lifting(backend):
    def helper():
        A, B, C, D = 10, 10, 10, 1024
        x = np.random.randn(A, D).astype(np.float32)
        x2 = x.copy()
        add = np.random.randn(B).astype(np.float32)
        mul = np.random.randn(C).astype(np.float32)
        p = {'A': parpy.threads(A)}
        repeated_seq_loops(x, add, mul, opts=par_opts(backend, p))

        p['D'] = parpy.threads(D).tpb(32)
        repeated_seq_loops(x2, add, mul, opts=par_opts(backend, p))
        assert np.allclose(x, x2, atol=1e-5)
    run_if_backend_is_enabled(backend, helper)

@parpy.jit
def seq_dependent_loop(
    x: buffer(F32, [A, C]),
    mul: buffer(F32, [B])
):
    for a in range(A):
        for b in range(a+1, B):
            x[a,:C] = x[a,:C] * mul[b]

@parpy.jit
def if_preventing_lifting(
    x: buffer(F32, [A, C]),
    mul: buffer(F32, [B])
):
    for a in range(A):
        for b in range(B):
            if a > b:
                x[a,:C] = x[a,:C] * mul[b]
            else:
                x[a,:C] = x[a,:C] + 1.0

@parpy.jit
def while_preventing_lifting(
    x: buffer(F32, [A, C]),
    mul: buffer(F32, [B])
):
    for a in range(A):
        for b in range(B):
            while a < b:
                x[a,:C] = x[a,:C] * mul[b]
                a += 1

failing_funcs = [seq_dependent_loop, if_preventing_lifting, while_preventing_lifting]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('fn', failing_funcs)
def test_lifting_fails(backend, fn):
    A, B, C = 10, 10, 1024
    x = np.random.randn(A, C).astype(np.float32)
    mul = np.random.randn(B).astype(np.float32)
    p = {'A': parpy.threads(A), 'C': parpy.threads(C).tpb(32)}
    with pytest.raises(RuntimeError) as e_info:
        parpy.print_compiled(fn, [x, mul], par_opts(backend, p))
    assert e_info.match(r"inter-block transformation failed")
