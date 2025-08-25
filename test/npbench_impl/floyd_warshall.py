import parpy
import torch

@parpy.jit
def kernel_helper(path, N):
    for k in range(N):
        parpy.label('i')
        for i in range(N):
            parpy.label('j')
            path[i,:] = parpy.min(path[i,:], path[i,k] + path[k,:])

def floyd_warshall(path, N, opts, compile_only=False):
    if compile_only:
        return parpy.print_compiled(kernel_helper, [path, N], opts)
    kernel_helper(path, N, opts=opts)
