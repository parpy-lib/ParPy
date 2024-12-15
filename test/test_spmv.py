import numpy as np
import parir
from parir.parir import ParKind
import pytest
import torch

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def spmv_row(A_values, A_rows, A_cols, A_nrows, x, y):
    for row in range(A_nrows):
        nnz_row = A_rows[row+1] - A_rows[row]
        sum = 0.0
        for i in range(A_rows[row], A_rows[row+1]):
            sum = sum + A_values[i] * x[A_cols[i]]
        y[row] = sum

def spmv_wrap(A_values, A_rows, A_cols, A_nrows, x, parallelize=None):
    y = torch.empty((A_nrows,), dtype=x.dtype, device=x.device)
    spmv_row(A_values, A_rows, A_cols, A_nrows, x, y, parallelize=parallelize)
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
    return A.to_sparse_csr()

# This test only runs when Torch has been compiled with CUDA support
def test_spmv_gpu():
    N = 256
    M = 4096
    sparsity = 0.01
    if torch.cuda.is_available():
        A = uniform_random_csr_f32_i64(N, M, sparsity)
        x = torch.randn(M, dtype=torch.float32, device='cuda')
        y1 = A.matmul(x)
        p = { "row": [ParKind.GpuBlocks(N)], "i": [ParKind.GpuThreads(128)] }
        y2 = spmv_wrap(A.values(), A.crow_indices(), A.col_indices(), N, x, p)
        torch.cuda.synchronize()
        assert torch.allclose(y1, y2, atol=1e-5)
