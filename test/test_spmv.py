import numpy as np
import parir
import pytest
import torch

import warnings

torch.manual_seed(1234)
np.random.seed(1234)

@parir.jit
def spmv_row(A, x, y):
    parir.label('row')
    for row in range(A["nrows"]):
        parir.label('i')
        for i in range(A["rows"][row], A["rows"][row+1]):
            y[row] += A["values"][i] * x[A["cols"][i]]

def spmv_wrap(A, x, N, p):
    A = {
        'values': A.values(),
        'rows': A.crow_indices(),
        'cols': A.col_indices(),
        'nrows': N
    }
    y = torch.zeros((A["nrows"],), dtype=x.dtype, device=x.device)
    spmv_row(A, x, y, parallelize=p, cache=False)
    return y

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

def compare_spmv(N, M, p=None):
    sparsity = 0.01
    A = uniform_random_csr_f32_i64(N, M, sparsity, 'cuda')
    x = torch.randn(M, dtype=torch.float32, device='cuda')
    # Compare result using PyTorch against parallelized code
    y1 = A.matmul(x)
    y2 = spmv_wrap(A, x, N, p)
    torch.cuda.synchronize()
    assert torch.allclose(y1, y2, atol=1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_spmv_seq_reduce():
    N, M = 256, 4096
    p = { "row": parir.threads(N) }
    compare_spmv(N, M, p)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_spmv_gpu():
    N, M = 256, 4096
    p = {
        "row": parir.threads(N),
        "i": parir.threads(128).reduce()
    }
    compare_spmv(N, M, p)

def test_spmv_compiles():
    N, M = 256, 4096
    sparsity = 0.01
    A = uniform_random_csr_f32_i64(N, M, sparsity, 'cpu')
    x = torch.randn(M, dtype=torch.float32)
    A = {
        'values': A.values(),
        'rows': A.crow_indices(),
        'cols': A.col_indices(),
        'nrows': N
    }
    y = torch.empty((A["nrows"],), dtype=x.dtype)
    p = {
        "row": parir.threads(N),
        "i": parir.threads(128).reduce()
    }
    s = parir.print_compiled(spmv_row, [A, x, y], p)
    assert len(s) != 0
