import ctypes
import numpy as np
import pandas as pd
import parir
import ssgetpy
import subprocess
import torch
import statistics
import sys

import common

torch.manual_seed(42)
alpha = 1.5
beta = 2.0

def torch_sddmm(A, B, C):
    return torch.sparse.sampled_addmm(C, A, B, alpha=alpha, beta=beta)

@parir.jit
def parir_sddmm_decompress_csr(C_crows, C_rows, N):
    parir.label('N')
    for row in range(N):
        parir.label('M')
        for i in range(C_crows[row], C_crows[row+1]):
            C_rows[i] = row

@parir.jit
def parir_sddmm_csr_kernel(A, B, C, D, N, alpha, beta):
    # Convert CSR to a COO representation by decompressing the rows. This
    # allows us to naively parallelize across all non-zero values without the
    # need to do load-balancing, as we would need to using CSR (or binary
    # search, which is costly).
    parir_sddmm_decompress_csr(C["crows"], C["rows"], N)
    parir.label('nnz')
    for i in range(C["nnz"]):
        row = C["rows"][i]
        col = C["cols"][i]
        t = parir.sum(A[row, :] * B[:, col])
        D[i] = alpha * t + beta * C["values"][i]

def parir_sddmm_csr(A, B, C):
    D = torch.empty_like(C)
    N, K = A.shape
    nnz = C._nnz()
    C_dict = {
        "crows": C.crow_indices(),
        "rows": torch.empty_like(C.col_indices()),
        "cols": C.col_indices(),
        "values": C.values(),
        "nnz": nnz
    }
    p = {
        "N": parir.threads(N),
        "M": parir.threads(32),
        "nnz": parir.threads(nnz),
    }
    opts = parir.par(p)
    parir_sddmm_csr_kernel(A, B, C_dict, D.values(), N, alpha, beta, opts=opts)
    return D

@parir.jit
def parir_sddmm_coo_kernel(A, B, C, D, alpha, beta):
    parir.label('nnz')
    for i in range(C["nnz"]):
        row = C["rows"][i]
        col = C["cols"][i]
        t = parir.sum(A[row, :] * B[:, col])
        D[i] = alpha * t + beta * C["values"][i]

def parir_sddmm_coo(A, B, C, C_rows):
    D = C.detach().clone()
    _, K = A.shape
    nnz = C._nnz()
    C_dict = {
        "rows": C_rows,
        "cols": C.col_indices(),
        "values": C.values(),
        "nnz": nnz
    }
    opts = parir.par({"nnz": parir.threads(nnz)})
    parir_sddmm_coo_kernel(A, B, C_dict, D.values(), alpha, beta, opts=opts)
    return D

def validate_sparse_result(A, actual_A):
    if actual_A is None:
        return True
    diff = torch.abs(A.values() - actual_A.values())
    atol = 1e-5
    rtol = 1e-5
    rhs = atol + rtol * torch.abs(actual_A.values())
    nfailed = len(actual_A.values()[diff > rhs])
    # If at least five elements failed to validate, i.e., were too far away
    # from the expected value, and this is more than 5 % of the total number of
    # elements in the matrix, we consider this fatal.
    if nfailed >= 5 and nfailed >= 0.05 * A._nnz():
        sys.stderr.write(f"Fatal: validation failed for {nfailed} out of {A._nnz()} elements\n")
        sys.stdout.write(f"{A}\n{actual_A}\n")
        return False
    if nfailed > 0:
        sys.stderr.write(f"Validation failed for {nfailed} out of {A._nnz()} elements\n")
    return True

# Compute the decompressed rows of a CSR matrix and pass this along with the
# CSR matrix. We do it this way because converting to a COO matrix uses more
# memory for some reason.
def csr_rows(csr_matrix):
    crows = csr_matrix.crow_indices()
    rows = torch.empty_like(csr_matrix.col_indices())
    N = len(crows)-1
    p = {"N": parir.threads(N), "M": parir.threads(32)}
    parir_sddmm_decompress_csr(crows, rows, N, opts=parir.par(p))
    return rows

def run_sddmm(framework, matrix_id, k):
    # Allocate tensors involved in the SDDMM operation. If this fails, we report an
    # OOM error.
    try:
        sparse_c = common.ssgetpy_matrix_to_csr(matrix_id)
        nnz = sparse_c._nnz()
        dense_a = torch.randn((sparse_c.shape[0], k), dtype=torch.float32, device='cuda')
        dense_b = torch.randn((k, sparse_c.shape[1]), dtype=torch.float32, device='cuda')
    except torch.cuda.OutOfMemoryError:
        return 34

    # Validate output
    torch_d = None
    sparse_d = None
    try:
        torch_d = torch_sddmm(dense_a, dense_b, sparse_c)
        # Use the cuSPARSE result as a baseline, and compare the output if using a
        # Parir framework.
        if framework == "PyTorch":
            sparse_d = None
        elif framework == "Parir-CSR":
            sparse_d = parir_sddmm_csr(dense_a, dense_b, sparse_c)
        elif framework == "Parir-COO":
            # Convert from CSR to COO on the CPU to reduce peak memory usage on
            # the GPU, which is typically the limiting factor.
            sparse_c_rows = csr_rows(sparse_c)
            sparse_d = parir_sddmm_coo(dense_a, dense_b, sparse_c, sparse_c_rows)
            del sparse_c_rows
        if not validate_sparse_result(torch_d, sparse_d):
            return 1
    except torch.cuda.OutOfMemoryError:
        sys.stderr.write("Skipping validation due to insufficient GPU memory\n")
    del sparse_d
    del torch_d
    torch.cuda.empty_cache()

    def mk_framework_entry(framework, t):
        return {
            "framework": framework,
            "benchmark": matrix_id,
            "time": t,
            "nnz": nnz
        }

    # Run the selected benchmark and report the results immediately
    try:
        if framework == "PyTorch":
            fn = lambda: torch_sddmm(dense_a, dense_b, sparse_c)
        elif framework == "Parir-CSR":
            fn = lambda: parir_sddmm_csr(dense_a, dense_b, sparse_c)
        elif framework == "Parir-COO":
            sparse_c_rows = csr_rows(sparse_c)
            fn = lambda: parir_sddmm_coo(dense_a, dense_b, sparse_c, sparse_c_rows)
        times = common.bench(matrix_id, fn, nwarmup=1)
        result = mk_framework_entry(framework, np.mean(times))
        common.append_csv(f"{common.SDDMM_NAME}-{k}.csv", [result])
        if framework == "Parir-COO":
            del sparse_c_rows
    except torch.cuda.OutOfMemoryError:
        return 34
    return 0

if __name__ == "__main__":
    framework = sys.argv[1]
    matrix_id = sys.argv[2]
    k = int(sys.argv[3])
    run_sddmm(framework, matrix_id, k)
