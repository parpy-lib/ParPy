# Implementation of row-wise summation in Python, including ParPy labels
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

# Print the generated code for the CUDA backend
opts.backend = parpy.CompileBackend.Cuda
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

# Use the automatically selected backend to avoid trouble
opts.backend = parpy.CompileBackend.Auto
code = parpy.print_compiled(sum_rows, [x, y, N], opts)

# Write the generated code to a file 'out.txt'
with open("out.txt", "w+") as f:
    f.write(code)

# Wait for the user to press enter before we read back the updated code
input("Press enter when finished updating 'out.txt' ")

# Read
with open("out.txt", "r") as f:
    code = f.read()
fn = parpy.compile_string("sum_rows", code, opts)
fn(x, y, N)
assert np.allclose(y, np.sum(x, axis=1), atol=1e-3)

# Remove the temporary file
import os
os.remove("out.txt")

print("Test OK")
