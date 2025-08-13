import prickle
import torch

@prickle.jit
def prickle_kernel(table, seq, N):
    with prickle.gpu:
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                if j-1 >= 0:
                    table[i,j] = prickle.max(table[i,j], table[i,j-1])
                if i+1 < N:
                    table[i,j] = prickle.max(table[i,j], table[i+1,j])
                if j-1 >= 0 and i+1 < N:
                    if i < j-1:
                        m = 1 if seq[i] + seq[j] == 3 else 0
                        table[i,j] = prickle.max(table[i,j], table[i+1,j-1] + m)
                    else:
                        table[i,j] = prickle.max(table[i,j], table[i+1,j-1])
                for k in range(i+1, j):
                    table[i,j] = prickle.max(table[i,j], table[i,k] + table[k+1,j])

def nussinov(N, seq, opts, compile_only=False):
    table = torch.zeros((N, N), dtype=torch.int32)
    if compile_only:
        return prickle.print_compiled(prickle_kernel, [table, seq, N], opts)
    prickle_kernel(table, seq, N, opts=opts)
    return table
