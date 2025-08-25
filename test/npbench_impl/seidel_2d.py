import parpy
import torch


@parpy.jit
def parpy_kernel(TSTEPS, N, A):
    with parpy.gpu:
        for t in range(0, TSTEPS - 1):
            for i in range(1, N - 1):
                A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                               A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                               A[i + 1, 2:])
                for j in range(1, N - 1):
                    A[i, j] += A[i, j - 1]
                    A[i, j] /= 9.0

def seidel_2d(TSTEPS, N, A, opts, compile_only=False):
    if compile_only:
        return parpy.print_compiled(parpy_kernel, [TSTEPS, N, A], opts)
    parpy_kernel(TSTEPS, N, A, opts=opts)
