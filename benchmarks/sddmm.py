import ctypes
import pandas as pd
import parir
import ssgetpy
import subprocess
import torch
import statistics
import sys

import common

torch.manual_seed(42)

# We use exit code 34 to indicate we ran out of memory.
def fail_oom():
    exit(34)

# Build the cuSPARSE library and initialize it when it is the selected
# framework.
r = subprocess.run(["make"], capture_output=True)
if r.returncode != 0:
    print("Could not build cuSPARSE wrapper library")
    print(r.stdout.decode('ascii'))
    print(r.stderr.decode('ascii'))
    exit(r.returncode)
lib = ctypes.cdll.LoadLibrary("./sddmm_cusparse.so")
lib.cusparse_init_handle()
lib.sddmm_init.restype = ctypes.c_int
lib.sddmm_init.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
lib.sddmm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]

def cusparse_sddmm_init(B, C, D):
    A = torch.empty_like(B)
    N, K = C.shape
    K, M = D.shape
    nnz = B._nnz()
    res = lib.sddmm_init(
        A.crow_indices().data_ptr(), A.col_indices().data_ptr(),
        A.values().data_ptr(), C.data_ptr(), D.data_ptr(), N, M, K, nnz
    )
    if res != 0:
        fail_oom()
    return A

def cusparse_sddmm_deinit():
    lib.sddmm_deinit()

def cusparse_sddmm(A, B):
    nnz = B._nnz()
    lib.sddmm(A.values().data_ptr(), B.values().data_ptr(), nnz)
    return A

def torch_sddmm(B, C, D):
    A = torch.sparse.sampled_addmm(B, C, D, beta=0.0)
    A *= B
    return A

@parir.jit
def parir_sddmm_csr_kernel(A, B, C, D, K, N):
    # Convert CSR to a COO representation by decompressing the rows. This
    # allows us to naively parallelize across all non-zero values without the
    # need to do load-balancing, as we would need to using CSR (or binary
    # search, which is costly).
    parir.label('N')
    for row in range(N):
        parir.label('M')
        for i in range(B["crows"][row], B["crows"][row+1]):
            B["rows"][i] = row
    parir.label('nnz')
    for i in range(B["nnz"]):
        row = B["rows"][i]
        col = B["cols"][i]
        t = parir.sum(C[row, :] * D[:, col])
        A[i] = B["values"][i] * t

def parir_sddmm_csr(B, C, D):
    A = torch.empty_like(B)
    N, _ = A.shape
    _, K = C.shape
    nnz = B._nnz()
    B_dict = {
        "crows": B.crow_indices(),
        "rows": torch.empty_like(B.col_indices()),
        "cols": B.col_indices(),
        "values": B.values(),
        "nnz": nnz
    }
    p = {
        "N": [parir.threads(N)],
        "M": [parir.threads(32)],
        "nnz": [parir.threads(nnz)],
    }
    parir_sddmm_csr_kernel(A.values(), B_dict, C, D, K, N, parallelize=p)
    return A

@parir.jit
def parir_sddmm_coo_kernel(A, B, C, D):
    parir.label('nnz')
    for i in range(B["nnz"]):
        row = B["rows"][i]
        col = B["cols"][i]
        t = parir.sum(C[row, :] * D[:, col])
        A[i] = B["values"][i] * t

def parir_sddmm_coo(B, C, D):
    A = B.detach().clone()
    nnz = B._nnz()
    B_dict = {
        "rows": B.indices()[0],
        "cols": B.indices()[1],
        "values": B.values(),
        "nnz": nnz
    }
    p = {"nnz": [parir.threads(nnz)]}
    parir_sddmm_coo_kernel(A.values(), B_dict, C, D, parallelize=p)
    return A

def validate_sparse_result(A, actual_A):
    if actual_A is None:
        return
    diff = torch.abs(A.values() - actual_A.values())
    rows = A.crow_indices()
    cols = A.col_indices()
    failed = False
    atol = 1e-5
    rtol = 1e-5
    rhs = atol + rtol * torch.abs(actual_A.values())
    nfailed = len(actual_A.values()[diff > rhs])
    if nfailed >= 0.05 * A._nnz():
        sys.stderr.write(f"Fatal: validation failure for {nfailed} out of {A._nnz()}\n")
        print(A)
        print(actual_A)
        exit(1)
    if nfailed > 0:
        sys.stderr.write(f"Validation failed for {nfailed} out of {A._nnz()} elements\n")

framework = sys.argv[1]
matrix_id = sys.argv[2]
k = 64

# Allocate tensors involved in the SDDMM operation. If this fails, we report an
# OOM error.
try:
    sparse_b = common.ssgetpy_matrix_to_csr(matrix_id)
    nnz = sparse_b._nnz()
    dense_c = torch.randn((sparse_b.shape[0], k), dtype=torch.float32, device='cuda')
    dense_d = torch.randn((k, sparse_b.shape[1]), dtype=torch.float32, device='cuda')
except torch.cuda.OutOfMemoryError:
    fail_oom()

# Validate output 
cusparse_a = None
sparse_a = None
try:
    cusparse_a = cusparse_sddmm_init(sparse_b, dense_c, dense_d)
    cusparse_a = cusparse_sddmm(cusparse_a, sparse_b)
# Use the cuSPARSE result as a baseline, and compare the output if using a
# Parir framework.
    if framework == "cuSPARSE" or framework == "PyTorch":
        sparse_a = None
    elif framework == "Parir-CSR":
        sparse_a = parir_sddmm_csr(sparse_b, dense_c, dense_d)
    elif framework == "Parir-COO":
        sparse_b_coo = sparse_b.to_sparse_coo()
        sparse_a = parir_sddmm_coo(sparse_b_coo, dense_c, dense_d)
    validate_sparse_result(cusparse_a, sparse_a)
    cusparse_sddmm_deinit()
except torch.cuda.OutOfMemoryError:
    sys.stderr.write("Skipping validation due to insufficient GPU memory\n")
del sparse_a
del cusparse_a
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
    if framework == "cuSPARSE":
        sparse_a = cusparse_sddmm_init(sparse_b, dense_c, dense_d)
        fn = lambda: cusparse_sddmm(sparse_a, sparse_b)
    elif framework == "PyTorch":
        fn = lambda: torch_sddmm(sparse_b, dense_c, dense_d)
    elif framework == "Parir-CSR":
        fn = lambda: parir_sddmm_csr(sparse_b, dense_c, dense_d)
    elif framework == "Parir-COO":
        sparse_b_coo = sparse_b.to_sparse_coo()
        del sparse_b
        fn = lambda: parir_sddmm_coo(sparse_b_coo, dense_c, dense_d)
    times = common.bench(matrix_id, fn, nwarmup=1)
    results = [mk_framework_entry(framework, t) for t in times]
    common.append_csv(common.SDDMM_CSV, results)
    if framework == "cuSPARSE":
        cusparse_sddmm_deinit()
except torch.cuda.OutOfMemoryError:
    fail_oom()
