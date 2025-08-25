import parpy
import torch

@parpy.jit
def kernel_wrap(A, B, TSTEPS):
    for t in range(1, TSTEPS):
        parpy.label('i')
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])

        parpy.label('i')
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

def jacobi_1d(TSTEPS, A, B, opts, compile_only=False):
    if compile_only:
        return parpy.print_compiled(kernel_wrap, [A, B, TSTEPS], opts)
    kernel_wrap(A, B, TSTEPS, opts=opts)
