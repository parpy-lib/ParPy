# Sparse Matrix-Vector Multiplication (SpMV)
import prickle
import torch

@prickle.jit
def spmv_helper(A_row, A_col, A_val, N, x, y):
    prickle.label('i')
    for i in range(N - 1):
        prickle.label('j')
        for j in range(A_row[i], A_row[i+1]):
            y[i] += A_val[j] * x[A_col[j]]

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x, opts):
    N, = A_row.shape
    A_row = A_row.to(dtype=torch.int32)
    A_col = A_col.to(dtype=torch.int32)
    y = torch.zeros(N - 1, dtype=A_val.dtype)
    spmv_helper(A_row, A_col, A_val, N, x, y, opts=opts)
    return y
