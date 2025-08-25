# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import parpy
import torch

@parpy.jit
def parpy_kernel(u, v, p, q, a, b, c, d, e, f, TSTEPS, N):
    for t in range(1, TSTEPS+1):
        parpy.label('i')
        for i in range(1, N-1):
            v[0,i] = 1.0
            p[i,0] = 0.0
            q[i,0] = v[0,i]
        for j in range(1, N-1):
            parpy.label('i')
            for i in range(1, N-1):
                p[i,j] = -c / (a * p[i,j-1] + b)
                q[i,j] = (-d * u[j,i-1] + (1.0 + 2.0 * d) * u[j,i] - f * u[j,i+1] -
                          a * q[i,j-1]) / (a * p[i,j-1] + b)
        parpy.label('i')
        for i in range(1, N-1):
            v[N-1,i] = 1.0
        for j in range(N-2, -1, -1):
            parpy.label('i')
            for i in range(1, N-1):
                v[j,i] = p[i,j] * v[j+1,i] + q[i,j]
        parpy.label('i')
        for i in range(1, N-1):
            u[i,0] = 1.0
            p[i,0] = 0.0
            q[i,0] = u[i,0]
        for j in range(1, N-1):
            parpy.label('i')
            for i in range(1, N-1):
                p[i,j] = -f / (d * p[i,j-1] + e)
                q[i,j] = (-a * v[i-1,j] + (1.0 + 2.0 * a) * v[i,j] - c * v[i+1,j] -
                          d * q[i,j-1]) / (d * p[i,j-1] + e)
        parpy.label('i')
        for i in range(1, N-1):
            u[i,N-1] = 1.0
        for j in range(N-2, -1, -1):
            parpy.label('i')
            for i in range(1, N-1):
                u[i,j] = p[i,j] * u[i,j+1] + q[i,j]

def adi(TSTEPS, N, u, opts, compile_only=False):
    v = torch.empty_like(u)
    p = torch.empty_like(u)
    q = torch.empty_like(u)
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = c = torch.tensor(-mul1 / 2.0, dtype=u.dtype)
    b = e = torch.tensor(1.0 + mul2, dtype=u.dtype)
    d = f = torch.tensor(-mul2 / 2.0, dtype=u.dtype)
    if compile_only:
        args = [u, v, p, q, a, b, c, d, e, f, TSTEPS, N]
        return parpy.print_compiled(parpy_kernel, args, opts)
    parpy_kernel(u, v, p, q, a, b, c, d, e, f, TSTEPS, N, opts=opts)
    return u
