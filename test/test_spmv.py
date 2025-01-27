import numpy as np
import parir
from parir import ParKind
import pytest
import torch

import warnings

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def spmv_row(A_values, A_rows, A_cols, A_nrows, x, y):
    for row in range(A_nrows):
        s = 0.0
        for i in range(A_rows[row], A_rows[row+1]):
            s = s + A_values[i] * x[A_cols[i]]
        y[row] = s

def spmv_wrap(A_values, A_rows, A_cols, A_nrows, x, p=None):
    y = torch.empty((A_nrows,), dtype=x.dtype, device=x.device)
    spmv_row(A_values, A_rows, A_cols, A_nrows, x, y, parallelize=p, cache=False)
    return y

def uniform_random_csr_f32_i64(N, M, d):
    nnz = int(N * M * d)
    s = set()
    while len(s) < nnz:
        s.update(np.random.randint(0, N*M, nnz - len(s)))
    flat_idxs = np.array(list(s), dtype=np.int64)
    rows, cols = np.divmod(flat_idxs, M)
    values = torch.randn(nnz, dtype=torch.float32, device='cuda')
    idxs = np.array((rows, cols))
    A = torch.sparse_coo_tensor(idxs, values, device='cuda')
    # Ignore warning about sparse CSR support being in beta state
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.to_sparse_csr()

def compare_spmv(N, M, p=None):
    sparsity = 0.01
    A = uniform_random_csr_f32_i64(N, M, sparsity)
    x = torch.randn(M, dtype=torch.float32, device='cuda')
    # Compare result using PyTorch against parallelized code
    y1 = A.matmul(x)
    y2 = spmv_wrap(A.values(), A.crow_indices(), A.col_indices(), N, x, p)
    torch.cuda.synchronize()
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_spmv_seq_reduce():
    N = 256
    M = 4096
    p = { "row": [ParKind.GpuThreads(N)] }
    compare_spmv(N, M, p)

@pytest.mark.skip(reason="Parallel reductions are not supported")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_spmv_gpu():
    N = 256
    M = 4096
    p = {
        "row": [ParKind.GpuThreads(N)],
        "i": [ParKind.GpuThreads(128)]
    }
    compare_spmv(N, M, p)
