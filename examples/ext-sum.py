import os
import parpy

@parpy.external(
    "warp_sum", parpy.CompileBackend.Cuda, parpy.Target.Device,
    header="<cuda_helper.h>", parallelize=parpy.threads(32)
)
def warp_sum(x: parpy.types.pointer(parpy.types.I32)) -> parpy.types.I32:
    return sum(x)

# Declaring data for the problem
import numpy as np
N = 100
M = 32
x = np.random.randint(1, 1000, (N,M)).astype(np.int32)
y = np.empty((N,), dtype=np.int32)

@parpy.jit
def sum_rows(x, y, N):
    parpy.label('N')
    for i in range(N):
        y[i] = warp_sum(x[i])

p = {'N': parpy.threads(N)}
opts = parpy.par(p)
include_path = f"{os.path.dirname(os.path.realpath(__file__))}/code"
print(f"Using extra include path: {include_path}")
opts.includes += [include_path]
sum_rows(x, y, N, opts=opts)
assert np.allclose(y, np.sum(x, axis=1))

print("Test OK")
