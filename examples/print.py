# Implementation of row-wise summation in Python, including ParPy labels
import parpy

@parpy.jit
def sum_rows(x, out, N):
    parpy.label('outer')
    for i in range(N):
        out[i] = parpy.reduce.sum(x[i,:])

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

# Print the generated code for the CUDA backend
opts.backend = parpy.CompileBackend.Cuda
#opts.print_debug = True  # uncomment to print the AST after each pass
code = parpy.print_compiled(sum_rows, [x, y, N], opts)
print("Generated code for CUDA C++:")
print(code)
print("=====")

# Print the generated code for the Metal backend
opts.backend = parpy.CompileBackend.Metal
code = parpy.print_compiled(sum_rows, [x, y, N], opts)
print("Generated code for Metal:")
print(code)
print("=====")

# Use the default backend to compile and run the code
opts.backend = parpy.default_backend()
code = parpy.print_compiled(sum_rows, [x, y, N], opts)
fn = parpy.compile_string("sum_rows", code, opts)
fn(x, y, N)
assert np.allclose(y, np.sum(x, axis=1), atol=1e-3)

print("Test OK")
