import parpy
import torch

@parpy.jit
def parpy_kernel(A, R, Q, M, N):
    for k in range(N):
        with parpy.gpu:
            parpy.label('i_reduce')
            nrm = parpy.sum(A[:,k] * A[:,k])
            R[k,k] = parpy.sqrt(nrm)
        parpy.label('i')
        Q[:,k] = A[:,k] / R[k,k]
        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i_reduce')
            R[k,j] = parpy.sum(Q[:,k] * A[:,j])

        parpy.label('j')
        for j in range(k+1, N):
            parpy.label('i')
            A[:,j] -= Q[:,k] * R[k,j]

def gramschmidt(A, opts, compile_only=False):
    Q = torch.zeros_like(A)
    R = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)
    M, N = A.shape
    if compile_only:
        args = [A, R, Q, M, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(A, R, Q, M, N, opts=opts)
    return Q, R
