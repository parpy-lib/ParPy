import parpy

@parpy.jit
def elemwise_add(x, y, out, N):
    parpy.label('outer')
    for i in range(N):
        out[i] = x[i] + y[i]

# Generate input data using NumPy
import numpy as np

N = 1024
x = np.random.randn(N).astype(np.float32)
y = np.random.randn(N).astype(np.float32)
out = np.empty_like(x)

# Use a Python dictionary to specify how to parallelize the code, and construct
# a default compile options object based on the parallel specification 'p'.
p = {'outer': parpy.threads(N)}
opts = parpy.par(p)

# Call the function with the defined arguments and the compile options, and
# verify that the result is correct with respect to NumPy after the call.
elemwise_add(x, y, out, N, opts=opts)
assert np.allclose(out, x + y, atol=1e-5)
print("Test OK")
