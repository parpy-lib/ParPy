# Initial version using lists
def mv(A, b):
    return [sum(A_val * b_val for (A_val, b_val) in zip(A_row, b)) for A_row in A]

A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = mv(A, b)
print("Lists", out)

# Using in-place mutation instead of returning a value
def mv_inplace(A, b, out, N):
    for row in range(N):
        out[row] = sum(A_val * b_val for (A_val, b_val) in zip(A[row], b))

N = 2
A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = [0.0 for row in range(N)]
mv_inplace(A, b, out, N)
print("Lists (in-place)", out)

# Using NumPy arrays instead of Python lists
import numpy as np

def mv_numpy(A, b, out, N):
    for row in range(N):
        out[row] = sum(A[row] * b)

N = 2
M = 2
A = np.array([[2.5, 3.5], [1.5, 0.5]])
b = np.array([2.0, 1.0])
out = np.zeros((N,))
mv_numpy(A, b, out, N)
print("NumPy v2", out)

# Using parallelization with ParPy
import parpy

@parpy.jit
def mv_parpy(A, b, out, N):
    parpy.label('N')
    for row in range(N):
        parpy.label('M')
        out[row] = parpy.operators.sum(A[row,:] * b[:])

out = np.zeros((N,))
opts = parpy.par({'N': parpy.threads(N)})
mv_parpy(A, b, out, N, opts=opts)
print("ParPy", out)

# ParPy with pre-allocated data
backend = parpy.CompileBackend.Cuda
A = parpy.buffer.from_array(A, backend)
b = parpy.buffer.from_array(b, backend)
out = parpy.buffer.from_array(out, backend)

import time
t1 = time.time_ns()
mv_parpy(A, b, out, N, opts=opts)
out.sync()
t2 = time.time_ns()
print("Time:", (t2-t1)/1e9)
print("ParPy v2", out.numpy())
