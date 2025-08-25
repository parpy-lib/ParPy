# We import NumPy and ParPy. In this example, we store data in NumPy arrays.
import numpy as np
import parpy

# The axpy routine takes a scalar 'a' and one-dimensional arrays (vectors) 'x'
# and 'y', and computes a times x plus y (axpy). We annotate the function with
# @parpy.jit to indicate that we want it to be parallelized; when we call
# such a function, we specify how we want it to be parallelized.
#
# Note how we pass an extra argument 'N' representing the number of elements in
# 'x'. This is required because the dimensions of arrays are not accessible
# within a JIT-compiled function.
@parpy.jit
def axpy(a, x, y, N):
    # We add a label referring to the for-loop (the following statement). When
    # we call the function, we specify how we want to parallelize the for-loop
    # by referring to this label. A label can be reused multiple times.
    parpy.label('N')
    for i in range(N):
        y[i] = a * x[i] + y[i]

N = 1024
x = np.ones(N)
y = np.zeros_like(x)
a = 2.5
print(f"Computing AXPY: {a} * {x} + {y}")

# When we call the function, we provide the compilation options via the 'opts'
# keyword argument. We use the 'parpy.par' function to construct a default
# compilation option object, where the provided dictionary determines how to
# parallelize the function. In this case, we specify that the for-loop should
# be parallelized across 128 threads.
axpy(a, x, y, N, opts=parpy.par({'N': parpy.threads(128)}))
print(f"Result: {y}")

# Assert that the values in 'y' have been correctly mutated.
assert np.allclose(y, a)

# A ParPy function mutates its provided arguments, and it cannot allocate new
# arguments or return a value. If this behavior is desired, we can add a
# wrapping function that handles the allocation of data.
def axpy_wrap(a, x):
    N, = x.shape
    y = np.zeros_like(x)
    axpy(a, x, y, N, opts=parpy.par({'N': parpy.threads(128)}))
    return y

assert np.allclose(axpy_wrap(a, x), a)
