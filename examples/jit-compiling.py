def mv(A, b):
    return [sum(A_val * b_val for (A_val, b_val) in zip(A_row, b)) for A_row in A]

A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
print(mv(A, b))

def mv(A, b, out, N):
    for row in range(N):
        out[row] = sum(A_val * b_val for (A_val, b_val) in zip(A[row], b))

N = 2
A = [[2.5, 3.5], [1.5, 0.5]]
b = [2.0, 1.0]
out = [0.0 for row in range(N)]
mv(A, b, out, N)
print(out)

import numpy as np

def mv(A, b, out, N):
    for row in range(N):
        out[row] = sum(A[row] * b)

N = 2
A = np.array([[2.5, 3.5], [1.5, 0.5]])
b = np.array([2.0, 1.0])
out = np.zeros((N,))
mv(A, b, out, N)
print(out)


import parpy

@parpy.jit
def mv(A, b, out, N):
    parpy.label('N')
    for row in range(N):
        parpy.label('M')
        out[row] = parpy.operators.sum(A[row,:] * b[:])

out = np.zeros((N,))
opts = parpy.par({'N': parpy.threads(N)})
mv(A, b, out, N, opts=opts)
print(out)
