# Example use of popcount implementation in Python
print(123 .bit_count())

# Declaring the external function
import parpy

@parpy.external("__popc", parpy.CompileBackend.Cuda, parpy.Target.Device)
def popcount(n: parpy.types.I32) -> parpy.types.I32:
    return n.bit_count()

# Calling the external function from sequential Python code
print(popcount(123))

# Implementation in ParPy using the declared external function
@parpy.jit
def popcount_many(x, count, N):
    parpy.label('N')
    for i in range(N):
        count[0] += popcount(x[i])

# Declaring data for the problem
import numpy as np
N = 100
x = np.random.randint(1, 1000, (N,)).astype(np.int32)

# Run sequentially
count_seq = sum([popcount(y) for y in x])

# Comparing parallelized results to sequential version
p = {'N': parpy.threads(N).reduce()}
opts = parpy.par(p)
count_par = np.array([0], dtype=np.int32)
popcount_many(x, count_par, N, opts=opts)

assert count_seq == count_par

print("Test OK")
