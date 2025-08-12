import prickle
import torch

@prickle.jit
def prickle_kernel(A, R, Q, M, N):
    for k in range(N):
        with prickle.gpu:
            prickle.label('i_reduce')
            nrm = prickle.sum(A[:,k] * A[:,k])
            R[k,k] = prickle.sqrt(nrm)
        prickle.label('i')
        Q[:,k] = A[:,k] / R[k,k]
        prickle.label('j')
        for j in range(k+1, N):
            prickle.label('i_reduce')
            R[k,j] = prickle.sum(Q[:,k] * A[:,j])

        prickle.label('j')
        for j in range(k+1, N):
            prickle.label('i')
            A[:,j] -= Q[:,k] * R[k,j]

def gramschmidt(A, opts):
    Q = torch.zeros_like(A)
    R = torch.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)
    M, N = A.shape
    prickle_kernel(A, R, Q, M, N, opts=opts)
    return Q, R
