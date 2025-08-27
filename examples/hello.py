# Implementation of row-wise summation in Python, including a ParPy label
import parpy

@parpy.jit
def sum_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        out[i] = parpy.sum(x[i,:])

# Generate input data using NumPy
import numpy as np

N = 100
M = 1024
x = np.random.randn(N, M).astype(np.float32)
y = np.empty((N,), dtype=np.float32)

# Use a Python dictionary to specify how to parallelize the code, and construct
# a default compile options object based on the parallel specification 'p'.
p = {'outer': parpy.threads(N)}
opts = parpy.par(p)

# Call the function with the defined arguments and the compile options, and
# verify that the result is correct with respect to NumPy after the call.
sum_rows(x, y, N, opts=opts)
assert np.allclose(y, np.sum(x, axis=1), atol=1e-3)
print("Test OK")
