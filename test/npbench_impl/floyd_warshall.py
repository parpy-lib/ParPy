import prickle
import torch

@prickle.jit
def kernel_helper(path, N):
    for k in range(N):
        prickle.label('i')
        for i in range(N):
            prickle.label('j')
            path[i,:] = prickle.min(path[i,:], path[i,k] + path[k,:])

def floyd_warshall(path, N, opts, compile_only=False):
    if compile_only:
        return prickle.print_compiled(kernel_helper, [path, N], opts)
    kernel_helper(path, N, opts=opts)
