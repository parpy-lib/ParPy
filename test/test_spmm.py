import numpy as np
from math import inf
import parir
from parir import ParKind
import pytest
import torch
import warnings

torch.manual_seed(1234)
np.random.seed(1234)

# Multiplies a sparse CSR matrix A with a dense matrix B, and stores the result
# in C. This implementation computes the result of each cell in parallel in
# separate blocks.
@parir.jit
def spmm_cell(A, B, C):
    for i in range(A["nrows"]):
        for j in range(B["ncols"]):
            s = parir.float32(0.0)
            for aidx in range(A["rows"][i], A["rows"][i+1]):
                s = s + A["values"][aidx] * B["values"][A["cols"][aidx], j]
            C[i, j] = s

def spmm_wrap(A, B, target, p=None):
    # Remove the original reference to the CSR matrix, since this cannot be
    # directly passed to the CUDA code.
    del A["original"]
    N = A["nrows"]
    K = B["ncols"]
    C = torch.zeros((N, K), dtype=torch.float32, device='cuda')
    target(A, B, C, parallelize=p)
    return C

def uniform_random_csr_f32_i64(N, M, d, device):
    nnz = int(N * M * d)
    s = set()
    while len(s) < nnz:
        s.update(np.random.randint(0, N*M, nnz - len(s)))
    flat_idxs = np.array(list(s), dtype=np.int64)
    rows, cols = np.divmod(flat_idxs, M)
    values = torch.randn(nnz, dtype=torch.float32, device=device)
    idxs = np.array((rows, cols))
    A = torch.sparse_coo_tensor(idxs, values, device=device)
    # Ignore warning about sparse CSR support being in beta state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.to_sparse_csr()

def spmm_test_data(device):
    N, M, K = 1024, 256, 512
    sparsity = 0.01
    A = uniform_random_csr_f32_i64(N, M, sparsity, device)
    B = torch.randn(M, K, dtype=torch.float32, device=device)
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_spmm():
    A, B = spmm_test_data('cuda')
    expected = A["original"].matmul(B["values"])
    p = {
        'i': [ParKind.GpuThreads(A["nrows"])],
        'j': [ParKind.GpuThreads(B["ncols"])],
        'aidx': [ParKind.GpuThreads(32), ParKind.GpuReduction()]
    }
    C = spmm_wrap(A, B, spmm_cell, p=p)
    assert torch.allclose(C, expected, atol=1e-5), f"{C}\n{expected}"

def test_spmm_compiles():
    A, B = spmm_test_data('cpu')
    del A["original"]
    C = torch.zeros((A["nrows"], B["ncols"]), dtype=torch.float32)
    p = {
        'i': [ParKind.GpuThreads(A["nrows"])]
    }
    s = parir.print_compiled(spmm_cell, [A, B, C], p)
    assert len(s) != 0

    p = {
        'i': [ParKind.GpuThreads(A["nrows"])],
        'j': [ParKind.GpuThreads(B["ncols"])]
    }
    s = parir.print_compiled(spmm_cell, [A, B, C], p)
    assert len(s) != 0

    p = {
        'i': [ParKind.GpuThreads(A["nrows"])],
        'j': [ParKind.GpuThreads(B["ncols"])],
        'aidx': [ParKind.GpuThreads(32), ParKind.GpuReduction()]
    }
    s = parir.print_compiled(spmm_cell, [A, B, C], p)
    assert len(s) != 0
