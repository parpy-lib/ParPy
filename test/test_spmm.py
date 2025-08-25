import numpy as np
from math import inf
import parpy
import pytest
import torch
import warnings

from common import *

torch.manual_seed(1234)
np.random.seed(1234)

# Multiplies a sparse CSR matrix A with a dense matrix B, and stores the result
# in C. This implementation computes the result of each cell in parallel in
# separate blocks.
@parpy.jit
def spmm_cell(A, B, C):
    parpy.label('i')
    for i in range(A["nrows"]):
        parpy.label('j')
        for j in range(B["ncols"]):
            parpy.label('aidx')
            for aidx in range(A["rows"][i], A["rows"][i+1]):
                C[i,j] += A["values"][aidx] * B["values"][A["cols"][aidx], j]

def spmm_wrap(A, B, target, opts):
    # Remove the original reference to the CSR matrix, since this cannot be
    # directly passed to the GPU code.
    del A["original"]
    N = A["nrows"]
    K = B["ncols"]
    C = torch.zeros((N, K), dtype=torch.float32)
    target(A, B, C, opts=opts)
    return C

def uniform_random_csr_f32_i64(N, M, d):
    nnz = int(N * M * d)
    s = set()
    while len(s) < nnz:
        s.update(np.random.randint(0, N*M, nnz - len(s)))
    flat_idxs = np.array(list(s), dtype=np.int64)
    rows, cols = np.divmod(flat_idxs, M)
    values = torch.randn(nnz, dtype=torch.float32)
    idxs = np.array((rows, cols))
    A = torch.sparse_coo_tensor(idxs, values)
    # Ignore warning about sparse CSR support being in beta state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.to_sparse_csr()

def spmm_test_data():
    N, M, K = 1024, 256, 512
    sparsity = 0.01
    A = uniform_random_csr_f32_i64(N, M, sparsity)
    B = torch.randn(M, K, dtype=torch.float32)
    N, M = A.shape
    M, K = B.shape
    # Include the 'original' pointer to the A CSR matrix, so that we can
    # compute the expected output using Torch for validation purposes.
    A = {
        'original': A,
        'values': A.values(),
        'rows': A.crow_indices(),
        'cols': A.col_indices(),
        'nnz': len(A.values()),
        'nrows': N,
        'ncols': M
    }
    B = {
        'values': B,
        'nrows': M,
        'ncols': K
    }
    return A, B

@pytest.mark.parametrize('backend', compiler_backends)
def test_spmm(backend):
    def helper():
        A, B = spmm_test_data()
        expected = A["original"].matmul(B["values"])
        p = {
            'i': parpy.threads(A["nrows"]),
            'j': parpy.threads(B["ncols"]),
            'aidx': parpy.threads(32).reduce()
        }
        C = spmm_wrap(A, B, spmm_cell, par_opts(backend, p))
        assert torch.allclose(C, expected, atol=1e-5), f"{C}\n{expected}"
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_spmm_compiles(backend):
    A, B = spmm_test_data()
    del A["original"]
    C = torch.zeros((A["nrows"], B["ncols"]), dtype=torch.float32)
    p = { 'i': parpy.threads(A["nrows"]) }
    s = parpy.print_compiled(spmm_cell, [A, B, C], par_opts(backend, p))
    assert len(s) != 0

    p = {
        'i': parpy.threads(A["nrows"]),
        'j': parpy.threads(B["ncols"])
    }
    s = parpy.print_compiled(spmm_cell, [A, B, C], par_opts(backend, p))
    assert len(s) != 0

    p = {
        'i': parpy.threads(A["nrows"]),
        'j': parpy.threads(B["ncols"]),
        'aidx': parpy.threads(32).reduce()
    }
    s = parpy.print_compiled(spmm_cell, [A, B, C], par_opts(backend, p))
    assert len(s) != 0
