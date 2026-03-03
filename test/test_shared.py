import numpy as np
import parpy
import pytest
import re
import subprocess
import tempfile

from common import *

np.random.seed(1234)

T = parpy.types.type_var()
M = parpy.types.shape_var()
N = parpy.types.shape_var()

# As Triton manages shared memory automatically, we cannot use any shared
# memory related functionality on the Triton backend.
smem_backends = \
    [backend for backend in compiler_backends if backend != parpy.CompileBackend.Triton]

@parpy.jit
def transpose_blocked(
        x: parpy.types.buffer(T, [M, N]),
        y: parpy.types.buffer(T, [N, M]),
        BM, BN):
    parpy.label('grid')
    for i, j in parpy.builtin.ranges((0, M, BM), (0, N, BN)):
        smem = parpy.builtin.alloc_shared((BM, BN+1), T)
        parpy.label('block')
        for m, n in parpy.builtin.ranges(BM, BN):
            smem[m, n] = x[i+m, j+n] if i+m < M and j+n < N else 0.0
        parpy.label('block')
        for n, m in parpy.builtin.ranges(BN, BM):
            if i+m < M and j+n < N:
                y[j+n, i+m] = smem[m, n]

@parpy.jit
def transpose_identity(
        x: parpy.types.buffer(T, [M, N]),
        y: parpy.types.buffer(T, [N, M]),
        BM, BN):
    parpy.label('grid')
    for i, j in parpy.builtin.ranges((0, M, BM), (0, N, BN)):
        # Transpose step 1
        smem = parpy.builtin.alloc_shared((BM, BN+1), T)
        parpy.label('block')
        for m, n in parpy.builtin.ranges(BM, BN):
            smem[m, n] = x[i+m, j+n] if i+m < M and j+n < N else 0.0
        parpy.label('block')
        for n, m in parpy.builtin.ranges(BN, BM):
            if i+m < M and j+n < N:
                y[j+n, i+m] = smem[m, n]
        # Transpose step 2
        smem2 = parpy.builtin.alloc_shared((BN, BM+1), T)
        parpy.label('block')
        for n, m in parpy.builtin.ranges(BN, BM):
            smem2[n, m] = y[j+n, i+m] if i+m < M and j+n < N else 0.0
        parpy.label('block')
        for m, n in parpy.builtin.ranges(BM, BN):
            if i+m < M and j+n < N:
                x[i+m, j+n] = smem2[n, m]

@parpy.jit
def arbitrary_smem_alloc(N):
    with parpy.gpu:
        smem = parpy.builtin.alloc_shared((N,), parpy.types.I8)
        parpy.label('block')
        for i in range(N):
            smem[i] = i

@pytest.mark.parametrize('backend', smem_backends)
def test_transpose_smem(backend):
    def helper():
        N = 100
        M = 53
        x = torch.randn(N, M, dtype=torch.float32)
        y = torch.zeros(M, N, dtype=torch.float32)
        BM, BN = 32, 32
        p = {
            'grid': parpy.threads((M + BM - 1) // BM * (N + BN - 1) // BN),
            'block': parpy.threads(64)
        }
        transpose_blocked(x, y, BM, BN, opts=par_opts(backend, p))
        assert torch.allclose(y, x.transpose(0, 1))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', smem_backends)
def test_smem_reuse(backend):
    N = 100
    M = 53
    x = torch.randn(N, M, dtype=torch.float32)
    y = torch.randn(M, N, dtype=torch.float32)
    BM, BN = 32, 32
    p = {
        'grid': parpy.threads((M + BM - 1) // BM * (N + BN - 1) // BN),
        'block': parpy.threads(64)
    }
    # If we allocate shared memory twice, but these allocations are not live at
    # the same time, the compiler should reuse memory to avoid allocating twice
    # as much memory as is necessary.
    expected_smem_usage = BM * (BN + 1) * 4
    s = parpy.print_compiled(transpose_identity, [x, y, BM, BN], par_opts(backend, p))
    if backend == parpy.CompileBackend.Cuda:
        pat = r".*cudaFuncSetAttribute(.*MaxDynamicSharedMemorySize, {smem}).*".format(smem=expected_smem_usage)
    elif backend == parpy.CompileBackend.Metal:
        pat = r".*parpy_metal::launch_kernel(.*, {smem}).*".format(smem=expected_smem_usage)
    else:
        raise RuntimeError(f"Unsupported backend {backend}")
    assert re.search(pat, s, re.DOTALL) is not None

@pytest.mark.parametrize('backend', smem_backends)
def test_smem_oom(backend):
    def helper():
        # We allocate an absurd amount of shared memory and test that this
        # leads to an error message that clearly communicates the problem.
        N = 2**64-1
        with pytest.raises(RuntimeError) as e_info:
            arbitrary_smem_alloc(N, opts=par_opts(backend, {'block': parpy.threads(32)}))
        assert e_info.match(r"Insufficient shared memory\..*current device only supports up to .* bytes")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', smem_backends)
def test_smem_invalid_shape(backend):
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def invalid_shape(N):
            with parpy.gpu:
                x = parpy.builtin.alloc_shared(N, parpy.types.I32)
    assert e_info.match(r"First argument of .* must be a tuple of dimensions")

@pytest.mark.parametrize('backend', smem_backends)
def test_smem_invalid_type(backend):
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def invalid_type(N):
            with parpy.gpu:
                x = parpy.builtin.alloc_shared(N, torch.float32)
    assert e_info.match(r"Second argument of .* must be a scalar ParPy type")

@parpy.jit
def transpose_called(
        x: parpy.types.buffer(T, [M, N]),
        y: parpy.types.buffer(T, [N, M]),
        i, j, BM, BN):
    smem = parpy.builtin.alloc_shared((BM, BN+1), T)
    parpy.label('block')
    for m, n in parpy.builtin.ranges(BM, BN):
        smem[m, n] = x[i+m, j+n] if i+m < M and j+n < N else 0.0
    parpy.label('block')
    for n, m in parpy.builtin.ranges(BN, BM):
        if i+m < M and j+n < N:
            y[j+n, i+m] = smem[m, n]

@parpy.jit
def transpose_blocked_call(
        x: parpy.types.buffer(T, [M, N]),
        y: parpy.types.buffer(T, [N, M]),
        BM, BN):
    parpy.label('grid')
    for i, j in parpy.builtin.ranges((0, M, BM), (0, N, BN)):
        transpose_called(x, y, i, j, BM, BN)

@pytest.mark.parametrize('backend', smem_backends)
def test_smem_alloc_in_called_function(backend):
    def helper():
        N = 123
        M = 44
        x = torch.randn(N, M, dtype=torch.float32)
        y = torch.zeros(M, N, dtype=torch.float32)
        BM, BN = 32, 32
        p = {
            'grid': parpy.threads((N + BN - 1) // BN * (M + BM - 1) // BM),
            'block': parpy.threads(128),
        }
        with pytest.raises(RuntimeError) as e_info:
            transpose_blocked_call(x, y, BM, BN, opts=par_opts(backend, p))
        assert e_info.match(r"Shared memory allocations.*not supported outside the entry point")
    run_if_backend_is_enabled(backend, helper)
