# Large parts of the initialization and the implementations in this test file
# are based on code from the NPBench suite. Mainly, we reuse the initialization
# code using input parameters corresponding to the 'S' configuration (the
# smallest input instance). We present the license of the NPBench repository
# below:
#
# BSD 3-Clause License
# 
# Copyright (c) 2021, SPCL
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import parpy
import pytest
import torch
import numpy as np

from common import *
import npbench_impl

def float_ty(backend):
    if backend == parpy.CompileBackend.Metal:
        return torch.float32
    else:
        return torch.float64

def complex_ty(backend):
    if backend == parpy.CompileBackend.Metal:
        return torch.complex64
    else:
        return torch.complex128

def adi_init(backend):
    TSTEPS = 5
    N = 100
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=np.float64)
    return TSTEPS, N, torch.tensor(u, dtype=float_ty(backend))

def run_adi(backend, compile_only):
    from npbench_impl.adi import adi
    TSTEPS, N, u1 = adi_init(backend)
    p = { 'i': parpy.threads(N-2) }
    opts = par_opts(backend, p)
    if compile_only:
        code = adi(TSTEPS, N, u1, opts, True)
        assert len(code) > 0
    else:
        a = adi(TSTEPS, N, u1, opts)
        TSTEPS, N, u2 = adi_init(backend)
        b = adi(TSTEPS, N, u2, opts)
        assert torch.allclose(a, b, atol=1e-3)

def arc_distance_init(backend):
    torch.manual_seed(1234)
    N = 100000
    t0 = torch.rand((N,), dtype=float_ty(backend))
    p0 = torch.rand((N,), dtype=float_ty(backend))
    t1 = torch.rand((N,), dtype=float_ty(backend))
    p1 = torch.rand((N,), dtype=float_ty(backend))
    return N, t0, p0, t1, p1

def run_arc_distance(backend, compile_only):
    from npbench_impl.arc_distance import arc_distance
    N, t0, p0, t1, p1 = arc_distance_init(backend)
    p = {'i': parpy.threads(N)}
    opts = par_opts(backend, p)
    if compile_only:
        code = arc_distance(t0, p0, t1, p1, opts, True)
        assert len(code) > 0
    else:
        a = arc_distance(t0, p0, t1, p1, opts)
        opts.seq = True
        b = arc_distance(t0, p0, t1, p1, opts)
        assert torch.allclose(a, b, atol=1e-3)

def azimint_naive_init(backend):
    torch.manual_seed(1234)
    npt = 100
    N = 40
    data = torch.rand(N, dtype=float_ty(backend))
    radius = torch.rand(N, dtype=float_ty(backend))
    return npt, data, radius

def run_azimint_naive(backend, compile_only):
    from npbench_impl.azimint_naive import azimint_naive
    npt, data, radius = azimint_naive_init(backend)
    p = {
        'i': parpy.threads(npt),
        'ix': parpy.threads(1024).reduce(),
        'j': parpy.threads(1024).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = azimint_naive(data, radius, npt, opts, True)
        assert len(code) > 0
    else:
        a = azimint_naive(data, radius, npt, opts)
        opts.seq = True
        b = azimint_naive(data, radius, npt, opts)
        assert torch.allclose(a, b, atol=1e-3)

def cavity_flow_init(backend):
    ny = 61
    nx = 61
    nt = 25
    nit = 5
    rho = 1.0
    nu = 0.1
    u = torch.zeros((ny, nx), dtype=float_ty(backend))
    v = torch.zeros((ny, nx), dtype=float_ty(backend))
    p = torch.zeros((ny, nx), dtype=float_ty(backend))
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / ((nx - 1) * (ny - 1))
    return nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu

def run_cavity_flow(backend, compile_only):
    from npbench_impl.cavity_flow import cavity_flow
    nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu = cavity_flow_init(backend)
    p = {'ny': parpy.threads(ny), 'nx': parpy.threads(nx)}
    opts = par_opts(backend, p)
    if compile_only:
        code = cavity_flow(nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu, opts, True)
        assert len(code) > 0
    else:
        cavity_flow(nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu, opts, True)
        opts.seq = True
        nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu = cavity_flow_init(backend)
        cavity_flow(nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu, opts, True)
        assert torch.allclose(u1, u2, atol=1e-3)
        assert torch.allclose(v1, v2, atol=1e-3)
        assert torch.allclose(p1, p2, atol=1e-3)

def channel_flow_init(backend):
    ny = 61
    nx = 61
    nit = 5
    rho = 1.0
    nu = 0.1
    F = 1.0
    u = torch.zeros((ny, nx), dtype=float_ty(backend))
    v = torch.zeros((ny, nx), dtype=float_ty(backend))
    p = torch.zeros((ny, nx), dtype=float_ty(backend))
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / ((nx - 1) * (ny - 1))
    return nit, u, v, dt, dx, dy, p, rho, nu, F

def run_channel_flow(backend, compile_only):
    from npbench_impl.channel_flow import channel_flow
    nit, u1, v1, dt, dx, dy, p1, rho, nu, F = channel_flow_init(backend)
    ny, nx = u1.shape
    p = {'ny': parpy.threads(ny), 'nx': parpy.threads(nx)}
    opts = par_opts(backend, p)
    if compile_only:
        code = channel_flow(nit, u1, v1, dt, dx, dy, p1, rho, nu, F, opts, True)
        assert len(code) > 0
    else:
        channel_flow(nit, u2, v2, dt, dx, dy, p2, rho, nu, F, opts)
        opts.seq = True
        nit, u2, v2, dt, dx, dy, p2, rho, nu, F = channel_flow_init(backend)
        channel_flow(nit, u2, v2, dt, dx, dy, p2, rho, nu, F, opts)
        assert torch.allclose(u1, u2, atol=1e-3)
        assert torch.allclose(v1, v2, atol=1e-3)
        assert torch.allclose(p1, p2, atol=1e-3)

def cholesky_init(backend):
    N = 100
    A = torch.empty((N, N), dtype=float_ty(backend))
    for i in range(N):
        A[i, :i+1] = torch.tensor(
            np.fromfunction(lambda j: (-j % N) / N + 1, (i+1,)), dtype=float_ty(backend)
        )
        A[i, i+1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ A.transpose(0, 1)
    return A

def run_cholesky(backend, compile_only):
    from npbench_impl.cholesky import cholesky
    p = { 'k': parpy.threads(256).reduce() }
    opts = par_opts(backend, p)
    A1 = cholesky_init(backend)
    if compile_only:
        code = cholesky(A1, opts, True)
        assert len(code) > 0
    else:
        cholesky(A1, opts)
        A2 = cholesky_init(backend)
        opts.seq = True
        cholesky(A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)

def compute_init(backend):
    torch.manual_seed(1234)
    N = 200
    M = 200
    array_1 = torch.randint(0, 1000, (M, N), dtype=torch.int64)
    array_2 = torch.randint(0, 1000, (M, N), dtype=torch.int64)
    a = 4
    b = 3
    c = 9
    return array_1, array_2, a, b, c

def run_compute(backend, compile_only):
    from npbench_impl.compute import compute
    array_1, array_2, a, b, c = compute_init(backend)
    N, _ = array_1.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = compute(array_1, array_2, a, b, c, opts, True)
        assert len(code) > 0
    else:
        r1 = compute(array_1, array_2, a, b, c, opts)
        opts.seq = True
        r2 = compute(array_1, array_2, a, b, c, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def conv2d_init(backend):
    torch.manual_seed(1234)
    N = 8
    C_in = 3
    C_out = 16
    K = 2
    H = 32
    W = 32
    input = torch.rand((N, H, W, C_in), dtype=torch.float32)
    weights = torch.rand((K, K, C_in, C_out), dtype=torch.float32)
    bias = torch.rand((C_out,), dtype=torch.float32)
    return input, weights, bias

def run_conv2d_bias(backend, compile_only):
    from npbench_impl.conv2d_bias import conv2d_bias
    input, weights, bias = conv2d_init(backend)
    H_out = input.shape[1] - weights.shape[0] + 1
    W_out = input.shape[2] - weights.shape[0] + 1
    p = {'i': parpy.threads(H_out), 'j': parpy.threads(W_out)}
    opts = par_opts(backend, p)
    if compile_only:
        code = conv2d_bias(input, weights, bias, opts, True)
        assert len(code) > 0
    else:
        r1 = conv2d_bias(input, weights, bias, opts)
        opts.seq = True
        r2 = conv2d_bias(input, weights, bias, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def correlation_init(backend):
    M = 500
    N = 600
    float_n = torch.tensor(N, dtype=float_ty(backend))
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
    return float_n, torch.tensor(data, dtype=float_ty(backend))

def run_correlation(backend, compile_only):
    from npbench_impl.correlation import correlation
    float_n, data = correlation_init(backend)
    _, M = data.shape
    p = {
        'i': parpy.threads(M-1),
        'j': parpy.threads(256),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = correlation(M, float_n, data, opts, True)
        assert len(code) > 0
    else:
        r1 = correlation(M, float_n, data, opts)
        float_n, data = correlation_init(backend)
        opts.seq = True
        r2 = correlation(M, float_n, data, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def covariance_init(backend):
    M = 500
    N = 600
    float_n = torch.tensor(N, dtype=float_ty(backend))
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
    return float_n, torch.tensor(data, dtype=float_ty(backend))

def run_covariance(backend, compile_only):
    from npbench_impl.covariance import covariance
    float_n, data = covariance_init(backend)
    _, M = data.shape
    p = {
        'i': parpy.threads(M),
        'j': parpy.threads(256),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = covariance(M, float_n, data, opts, True)
        assert len(code) > 0
    else:
        r1 = covariance(M, float_n, data, opts)
        float_n, data = covariance_init(backend)
        opts.seq = True
        r2 = covariance(M, float_n, data, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def crc16_init(backend):
    torch.manual_seed(1234)
    N = 1600
    data = torch.randint(0, 256, (N,), dtype=torch.int32)
    return data

def run_crc16(backend, compile_only):
    from npbench_impl.crc16 import crc16
    data = crc16_init(backend)
    opts = par_opts(backend, {})
    if compile_only:
        code = crc16(data, opts, True)
        assert len(code) > 0
    else:
        r1 = crc16(data, opts)
        data = crc16_init(backend)
        opts.seq = True
        r2 = crc16(data, opts)
        assert r1 == r2

def deriche_init(backend):
    W = 400
    H = 200
    alpha = 0.25
    img_in = np.fromfunction(lambda i, j:
                             ((313 * i + 991 * j) % 65536) / 65535.0, (W, H),
                             dtype=np.float64)
    return alpha, torch.tensor(img_in, dtype=float_ty(backend))

def run_deriche(backend, compile_only):
    from npbench_impl.deriche import deriche
    alpha, img_in = deriche_init(backend)
    W, H = img_in.shape
    p = {
        'i': parpy.threads(W),
        'j': parpy.threads(H)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = deriche(alpha, img_in, opts, True)
        assert len(code) > 0
    else:
        r1 = deriche(alpha, img_in, opts)
        opts.seq = True
        r2 = deriche(alpha, img_in, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def durbin_init(backend):
    N = 1000
    r = np.fromfunction(lambda i: N + 1 - i, (N,))
    return torch.tensor(r, dtype=float_ty(backend))

def run_durbin(backend, compile_only):
    if backend == parpy.CompileBackend.Metal:
        pytest.skip("Skip as test only seems to work using 64-bit floats")
    from npbench_impl.durbin import durbin
    p = {
        'k_red': parpy.threads(512).reduce(),
        'k': parpy.threads(512)
    }
    opts = par_opts(backend, p)
    r = durbin_init(backend)
    if compile_only:
        code = durbin(r, opts, True)
        assert len(code) > 0
    else:
        r1 = durbin(r, opts)
        opts.seq = True
        r2 = durbin(r, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def fdtd2d_init(backend):
    TMAX = 20
    NX = 200
    NY = 220
    ex = np.fromfunction(lambda i, j: (i*(j+1)) / NX, (NX, NY),
                         dtype=np.float64)
    ey = np.fromfunction(lambda i, j: (i*(j+2)) / NX, (NX, NY),
                         dtype=np.float64)
    hz = np.fromfunction(lambda i, j: (i*(j+3)) / NX, (NX, NY),
                         dtype=np.float64)
    _fict_ = np.fromfunction(lambda i: i, (TMAX,), dtype=np.float64)
    return (
        TMAX, NX, NY,
        torch.tensor(ex, dtype=float_ty(backend)),
        torch.tensor(ey, dtype=float_ty(backend)),
        torch.tensor(hz, dtype=float_ty(backend)),
        torch.tensor(_fict_, dtype=float_ty(backend))
    )

def run_fdtd2d(backend, compile_only):
    from npbench_impl.fdtd2d import fdtd2d
    TMAX, NX, NY, ex1, ey1, hz1, _fict_ = fdtd2d_init(backend)
    p = {
        'i': parpy.threads(NX-1),
        'j': parpy.threads(1024),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = fdtd2d(TMAX, ex1, ey1, hz1, _fict_, opts, True)
        assert len(code) > 0
    else:
        fdtd2d(TMAX, ex1, ey1, hz1, _fict_, opts)
        opts.seq = True
        TMAX, NX, NY, ex2, ey2, hz2, _fict_ = fdtd2d_init(backend)
        fdtd2d(TMAX, ex2, ey2, hz2, _fict_, opts)
        assert torch.allclose(ex1, ex2, atol=1e-3)
        assert torch.allclose(ey1, ey2, atol=1e-3)
        assert torch.allclose(hz1, hz2, atol=1e-3)

def floyd_warshall_init(backend):
    N = 200
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if (i+j) % 13 == 0 or (i+j) % 7 == 0 or (i+j) % 11 == 0:
                path[i,j] = 999
    return torch.tensor(path), N

def run_floyd_warshall(backend, compile_only):
    from npbench_impl.floyd_warshall import floyd_warshall
    path1, N = floyd_warshall_init(backend)
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(N)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = floyd_warshall(path1, N, opts, True)
        assert len(code) > 0
    else:
        floyd_warshall(path1, N, opts)
        opts.seq = True
        path2, N = floyd_warshall_init(backend)
        floyd_warshall(path2, N, opts)
        assert torch.allclose(path1, path2, atol=1e-3)

def go_fast_init(backend):
    torch.manual_seed(1234)
    N = 2000
    x = torch.rand((N, N), dtype=float_ty(backend))
    return x, N

def run_go_fast(backend, compile_only):
    from npbench_impl.go_fast import go_fast
    x, N = go_fast_init(backend)
    p = {
        'i': parpy.threads(1024).reduce(),
        'ix': parpy.threads(N),
        'j': parpy.threads(N)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = go_fast(x, opts, True)
        assert len(code) > 0
    else:
        r1 = go_fast(x, opts)
        opts.seq = True
        r2 = go_fast(x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def gramschmidt_init(backend):
    torch.manual_seed(1234)
    M = 70
    N = 60
    A = torch.rand((M, N), dtype=float_ty(backend))
    while torch.linalg.matrix_rank(A) < N:
        A = torch.rand((M, N), dtype=float_ty(backend))
    return A

def run_gramschmidt(backend, compile_only):
    from npbench_impl.gramschmidt import gramschmidt
    A = gramschmidt_init(backend)
    M, N = A.shape
    p = {
        'i': parpy.threads(M),
        'i_reduce': parpy.threads(128).reduce(),
        'j': parpy.threads(N)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = gramschmidt(A, opts, True)
        assert len(code) > 0
    else:
        Q1, R1 = gramschmidt(A, opts)
        opts.seq = True
        A = gramschmidt_init(backend)
        Q2, R2 = gramschmidt(A, opts)
        assert torch.allclose(Q1, Q2, atol=1e-3)
        assert torch.allclose(R1, R2, atol=1e-3)

def hdiff_init(backend):
    torch.manual_seed(1234)
    I = 64
    J = 64
    K = 60
    in_field = torch.rand((I+4, J+4, K), dtype=float_ty(backend))
    out_field = torch.rand((I, J, K), dtype=float_ty(backend))
    coeff = torch.rand((I, J, K), dtype=float_ty(backend))
    return in_field, out_field, coeff

def run_hdiff(backend, compile_only):
    from npbench_impl.hdiff import hdiff
    in_field, out_field, coeff = hdiff_init(backend)
    I, J, K = out_field.shape
    p = {'I': parpy.threads(I), 'J': parpy.threads(J), 'K': parpy.threads(K)}
    opts = par_opts(backend, p)
    if compile_only:
        code = hdiff(in_field, out_field, coeff, opts, True)
        assert len(code) > 0
    else:
        hdiff(in_field, out_field, coeff, opts, True)
        opts.seq = True
        in_field, out_field2, coeff = hdiff_init(backend)
        hdiff(in_field, out_field2, coeff, opts, True)
        assert torch.allclose(out_field, out_field2, atol=1e-3)

def heat_3d_init(backend):
    TSTEPS = 25
    N = 25
    A = np.fromfunction(lambda i,j,k: (i+j+(N-k)) * 10 / N, (N,N,N), dtype=np.float64)
    B = torch.tensor(np.copy(A), dtype=float_ty(backend))
    return TSTEPS, torch.tensor(A, dtype=float_ty(backend)), B

def run_heat_3d(backend, compile_only):
    from npbench_impl.heat_3d import heat_3d
    p = {
        'i': parpy.threads(64),
        'j': parpy.threads(64),
        'k': parpy.threads(64)
    }
    opts = par_opts(backend, p)
    TSTEPS, A1, B1 = heat_3d_init(backend)
    if compile_only:
        code = heat_3d(TSTEPS, A1, B1, opts, True)
        assert len(code) > 0
    else:
        heat_3d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = heat_3d_init(backend)
        heat_3d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)

def jacobi_1d_init(backend):
    TSTEPS = 800
    N = 3200
    A = np.fromfunction(lambda i: (i+2) / N, (N,), dtype=np.float64)
    B = np.fromfunction(lambda i: (i+3) / N, (N,), dtype=np.float64)
    fty = float_ty(backend)
    return TSTEPS, torch.tensor(A, dtype=fty), torch.tensor(B, dtype=fty)

def run_jacobi_1d(backend, compile_only):
    from npbench_impl.jacobi_1d import jacobi_1d
    TSTEPS, A1, B1 = jacobi_1d_init(backend)
    N, = A1.shape
    p = {'i': parpy.threads(N-2)}
    opts = par_opts(backend, p)
    if compile_only:
        code = jacobi_1d(TSTEPS, A1, B1, opts, True)
        assert len(code) > 0
    else:
        jacobi_1d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = jacobi_1d_init(backend)
        jacobi_1d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)

def jacobi_2d_init(backend):
    TSTEPS = 50
    N = 150
    A = np.fromfunction(lambda i,j: i*(j+2) / N, (N, N), dtype=np.float64)
    B = np.fromfunction(lambda i,j: i*(j+3) / N, (N, N), dtype=np.float64)
    fty = float_ty(backend)
    return TSTEPS, torch.tensor(A, dtype=fty), torch.tensor(B, dtype=fty)

def run_jacobi_2d(backend, compile_only):
    from npbench_impl.jacobi_2d import jacobi_2d
    TSTEPS, A1, B1 = jacobi_2d_init(backend)
    N, N = A1.shape
    p = {
        'i': parpy.threads(N-1),
        'j': parpy.threads(N-1),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = jacobi_2d(TSTEPS, A1, B1, opts, True)
        assert len(code) > 0
    else:
        jacobi_2d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = jacobi_2d_init(backend)
        jacobi_2d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)

def lenet_init(backend):
    torch.manual_seed(1234)
    N = 4
    H = 28
    W = 28
    H_conv1 = H - 4 
    W_conv1 = W - 4 
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4 
    W_conv2 = W_pool1 - 4 
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2

    # NHWC data layout
    input = torch.rand((N, H, W, 1), dtype=torch.float32)
    # Weights
    conv1 = torch.rand((5, 5, 1, 6), dtype=torch.float32)
    conv1bias = torch.rand((6, ), dtype=torch.float32)
    conv2 = torch.rand((5, 5, 6, 16), dtype=torch.float32)
    conv2bias = torch.rand((16, ), dtype=torch.float32)
    fc1w = torch.rand((C_before_fc1, 120), dtype=torch.float32)
    fc1b = torch.rand((120, ), dtype=torch.float32)
    fc2w = torch.rand((120, 84), dtype=torch.float32)
    fc2b = torch.rand((84, ), dtype=torch.float32)
    fc3w = torch.rand((84, 10), dtype=torch.float32)
    fc3b = torch.rand((10, ), dtype=torch.float32)
    return (input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
            fc3w, fc3b, C_before_fc1)

def run_lenet(backend, compile_only):
    from npbench_impl.lenet import lenet
    input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = lenet_init(backend)
    N, _, _, _ = input.shape
    opts = par_opts(backend, {})
    if compile_only:
        code = lenet(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1, opts, True)
        assert len(code) > 0
    else:
        r1 = lenet(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1, opts)
        opts.seq = True
        r2 = lenet(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def lu_init(backend):
    N = 60
    fty = float_ty(backend)
    A = torch.empty(N, N, dtype=fty)
    for i in range(N):
        A[i, :i+1] = torch.tensor(np.fromfunction(lambda j: (-j % N) / N + 1, (i+1,)), dtype=fty)
        A[i, i+1:] = 0.0
        A[i, i] = 1.0
    A[:] = A @ A.transpose(0, 1)
    return A

def run_lu(backend, compile_only):
    from npbench_impl.lu import lu
    p = {'k': parpy.threads(128).reduce()}
    opts = par_opts(backend, p)
    A1 = lu_init(backend)
    if compile_only:
        code = lu(A1, opts, True)
        assert len(code) > 0
    else:
        lu(A1, opts)
        opts.seq = True
        A2 = lu_init(backend)
        lu(A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)

def ludcmp_init(backend):
    N = 60
    A = lu_init(backend)
    fn = np.float64(N)
    b = np.fromfunction(lambda i: (i+1)/fn/2.0 + 4.0, (N,), dtype=np.float64)
    return A, torch.tensor(b, dtype=float_ty(backend))

def run_ludcmp(backend, compile_only):
    from npbench_impl.ludcmp import ludcmp
    p = {'k': parpy.threads(128).reduce()}
    opts = par_opts(backend, p)
    A1, b = ludcmp_init(backend)
    if compile_only:
        code = ludcmp(A1, b, opts, True)
        assert len(code) > 0
    else:
        x1, y1 = ludcmp(A1, b, opts)
        opts.seq = True
        A2, b = ludcmp_init(backend)
        x2, y2 = ludcmp(A2, b, opts)
        assert torch.allclose(x1, x2, atol=1e-3)
        assert torch.allclose(y1, y2, atol=1e-3)
        assert torch.allclose(A1, A2, atol=1e-3)

def mlp_init(backend):
    torch.manual_seed(1234)
    C_in = 3
    N = 8
    S0 = 30000
    S1 = 2000
    S2 = 2000
    mlp_sizes = [S0, S1, S2]
    input = torch.rand(N, C_in, dtype=torch.float32)
    w1 = torch.rand(C_in, mlp_sizes[0], dtype=torch.float32)
    b1 = torch.rand(mlp_sizes[0], dtype=torch.float32)
    w2 = torch.rand(mlp_sizes[0], mlp_sizes[1], dtype=torch.float32)
    b2 = torch.rand(mlp_sizes[1], dtype=torch.float32)
    w3 = torch.rand(mlp_sizes[1], mlp_sizes[2], dtype=torch.float32)
    b3 = torch.rand(mlp_sizes[2], dtype=torch.float32)
    return input, w1, b1, w2, b2, w3, b3

def run_mlp(backend, compile_only):
    from npbench_impl.mlp import mlp
    input, w1, b1, w2, b2, w3, b3 = mlp_init(backend)
    N, _ = input.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(1024),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = mlp(input, w1, b1, w2, b2, w3, b3, opts, True)
        assert len(code) > 0
    else:
        x1 = mlp(input, w1, b1, w2, b2, w3, b3, opts)
        opts.seq = True
        x2 = mlp(input, w1, b1, w2, b2, w3, b3, opts)
        assert torch.allclose(x1, x2, atol=1e-3)

def nbody_init(backend):
    torch.manual_seed(1234)
    N = 25
    t_end = 2.0
    dt = 0.05
    softening = 0.1
    G = 1.0
    mass = 20.0 * torch.ones(N, 1) / N
    pos = torch.rand(N, 3)
    vel = torch.rand(N, 3)
    Nt = int(np.ceil(t_end / dt))
    return mass, pos, vel, N, Nt, dt, G, softening

def run_nbody(backend, compile_only):
    from npbench_impl.nbody import nbody
    mass, pos, vel, N, Nt, dt, G, softening = nbody_init(backend)
    p = {
        'N2': parpy.threads(N*N),
        'N': parpy.threads(N),
        'reduce': parpy.threads(64).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = nbody(mass, pos, vel, N, Nt, dt, G, softening, opts, True)
        assert len(code) > 0
    else:
        KE1, PE1 = nbody(mass, pos, vel, N, Nt, dt, G, softening, opts)
        opts.seq = True
        KE2, PE2 = nbody(mass, pos, vel, N, Nt, dt, G, softening, opts)
        assert torch.allclose(KE1, KE2, atol=1e-3)
        assert torch.allclose(PE1, PE2, atol=1e-3)

def nussinov_init(backend):
    N = 40
    seq = np.fromfunction(lambda i: (i+1) % 4, (N,), dtype=np.int32)
    return torch.tensor(seq)

def run_nussinov(backend, compile_only):
    from npbench_impl.nussinov import nussinov
    opts = par_opts(backend, {})
    seq = nussinov_init(backend)
    N, = seq.shape
    if compile_only:
        code = nussinov(N, seq, opts, True)
        assert len(code) > 0
    else:
        r1 = nussinov(N, seq, opts)
        opts.seq = True
        r2 = nussinov(N, seq, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def resnet_init(backend):
    torch.manual_seed(1234)
    N = 8
    W = 14
    H = 14
    C1 = 32
    C2 = 8
    input = torch.rand(N, H, W, C1, dtype=torch.float32)
    conv1 = torch.rand(1, 1, C1, C2, dtype=torch.float32)
    conv2 = torch.rand(3, 3, C2, C2, dtype=torch.float32)
    conv3 = torch.rand(1, 1, C2, C1, dtype=torch.float32)
    return input, conv1, conv2, conv3

def run_resnet(backend, compile_only):
    from npbench_impl.resnet import resnet
    input, conv1, conv2, conv3 = resnet_init(backend)
    H_out = input.shape[1] - conv1.shape[0] + 1
    W_out = input.shape[2] - conv1.shape[0] + 1
    p = {'i': parpy.threads(H_out), 'j': parpy.threads(W_out)}
    opts = par_opts(backend, p)
    if compile_only:
        code = resnet(input, conv1, conv2, conv3, opts, True)
        assert len(code) > 0
    else:
        r1 = resnet(input, conv1, conv2, conv3, opts)
        opts.seq = True
        r2 = resnet(input, conv1, conv2, conv3, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def sselfeng_init(backend):
    torch.manual_seed(1234)
    Nkz = 2
    NE = 4
    Nqz = 2
    Nw = 2
    N3D = 2
    NA = 6
    NB = 2
    Norb = 3
    neigh_idx = torch.empty(NA, NB, dtype=torch.int32)
    for i in range(NA):
        neigh_idx[i] = torch.tensor(np.positive(np.arange(i - NB / 2, i + NB / 2) % NA))

    dH = torch.rand(NA, NB, N3D, Norb, Norb, dtype=complex_ty(backend))
    G = torch.rand(Nkz, NE, NA, Norb, Norb, dtype=complex_ty(backend))
    D = torch.rand(Nqz, Nw, NA, NB, N3D, N3D, dtype=complex_ty(backend))
    Sigma = torch.rand(Nkz, NE, NA, Norb, Norb, dtype=complex_ty(backend))
    return neigh_idx, dH, G, D, Sigma

def run_sselfeng(backend, compile_only):
    from npbench_impl.sselfeng import sselfeng
    neigh_idx, dH, G, D, Sigma1 = sselfeng_init(backend)
    Nkz, NE, NA, Norb, Norb = G.shape
    p = {
        'Nkz': parpy.threads(Nkz),
        'NE': parpy.threads(NE),
        'NA': parpy.threads(NA),
        'threads': parpy.threads(32)
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = sselfeng(neigh_idx, dH, G, D, Sigma1, opts, True)
        assert len(code) > 0
    else:
        sselfeng(neigh_idx, dH, G, D, Sigma1, opts)
        opts.seq = True
        neigh_idx, dH, G, D, Sigma2 = sselfeng_init(backend)
        sselfeng(neigh_idx, dH, G, D, Sigma2, opts)
        assert torch.allclose(Sigma1, Sigma2, atol=1e-3)

def seidel_2d_init(backend):
    TSTEPS = 8
    N = 50
    A = np.fromfunction(lambda i,j: (i * (j+2) + 2) / N, (N, N), dtype=np.float64)
    return TSTEPS, N, torch.tensor(A, dtype=float_ty(backend))

def run_seidel_2d(backend, compile_only):
    from npbench_impl.seidel_2d import seidel_2d
    TSTEPS, N, A1 = seidel_2d_init(backend)
    opts = par_opts(backend, {})
    if compile_only:
        code = seidel_2d(TSTEPS, N, A1, opts, True)
        assert len(code) > 0
    else:
        seidel_2d(TSTEPS, N, A1, opts)
        opts.seq = True
        TSTEPS, N, A2 = seidel_2d_init(backend)
        seidel_2d(TSTEPS, N, A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)

def softmax_init(backend):
    torch.manual_seed(1234)
    N = 16
    H = 16
    SM = 128
    x = torch.rand(N, H, SM, SM, dtype=torch.float32)
    return x

def run_softmax(backend, compile_only):
    from npbench_impl.softmax import softmax
    x = softmax_init(backend)
    N, H, SM, SM = x.shape
    p = {
        'i': parpy.threads(N),
        'j': parpy.threads(H),
        'k': parpy.threads(SM),
        'l': parpy.threads(256),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = softmax(x, opts, True)
        assert len(code) > 0
    else:
        r1 = softmax(x, opts)
        opts.seq = True
        r2 = softmax(x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def spmv_init(backend):
    torch.manual_seed(1234)
    M = 4096
    N = 4096
    nnz = 8192
    x = torch.rand(N, dtype=float_ty(backend))

    from scipy.sparse import random
    rng = np.random.default_rng(42)
    matrix = random(M, N, density=nnz / (M * N), format='csr', dtype=np.float64, random_state=rng)
    rows = torch.tensor(matrix.indptr, dtype=torch.int32)
    cols = torch.tensor(matrix.indices, dtype=torch.int32)
    vals = torch.tensor(matrix.data, dtype=float_ty(backend))

    return rows, cols, vals, x

def run_spmv(backend, compile_only):
    from npbench_impl.spmv import spmv
    rows, cols, vals, x = spmv_init(backend)
    N, = x.shape
    p = {
        'i': parpy.threads(N-1),
        'j': parpy.threads(64).reduce(),
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = spmv(rows, cols, vals, x, opts, True)
        assert len(code) > 0
    else:
        r1 = spmv(rows, cols, vals, x, opts)
        opts.seq = True
        r2 = spmv(rows, cols, vals, x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)

def symm_init(backend):
    M = 40
    N = 50
    alpha = 1.5
    beta = 1.2
    C = torch.tensor(
        np.fromfunction(lambda i,j: ((i+j) % 100) / M, (M, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    B = torch.tensor(
        np.fromfunction(lambda i,j: ((N+i-j) % 100) / M, (M, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    A = torch.empty(M, M, dtype=float_ty(backend))
    for i in range(M):
        v = np.fromfunction(lambda j: ((i+j) % 100) / M, (i+1,), dtype=np.float64)
        A[i, :i+1] = torch.tensor(v, dtype=float_ty(backend))
        A[i, i+1:] = -999
    return alpha, beta, C, A, B

def run_symm(backend, compile_only):
    from npbench_impl.symm import symm
    alpha, beta, C1, A, B = symm_init(backend)
    M, N = C1.shape
    p = {
        'M': parpy.threads(M),
        'N': parpy.threads(N),
        'i_red': parpy.threads(32).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = symm(alpha, beta, C1, A, B, opts, True)
        assert len(code) > 0
    else:
        symm(alpha, beta, C1, A, B, opts)
        opts.seq = True
        alpha, beta, C2, A, B = symm_init(backend)
        symm(alpha, beta, C2, A, B, opts)
        assert torch.allclose(C1, C2, atol=1e-3)

def syr2k_init(backend):
    M = 35
    N = 50
    alpha = 1.5
    beta = 1.2
    C = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j+3) % N) / N, (N, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    A = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j+1) % N) / N, (N, M), dtype=np.float64),
        dtype=float_ty(backend)
    )
    B = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j+2) % N) / N, (N, M), dtype=np.float64),
        dtype=float_ty(backend)
    )
    return alpha, beta, C, A, B

def run_syr2k(backend, compile_only):
    from npbench_impl.syr2k import syr2k
    alpha, beta, C1, A, B = syr2k_init(backend)
    N, M = A.shape
    p = {
        'i': parpy.threads(N),
        'k': parpy.threads(256).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = syr2k(alpha, beta, C1, A, B, opts, True)
        assert len(code) > 0
    else:
        syr2k(alpha, beta, C1, A, B, opts)
        opts.seq = True
        alpha, beta, C2, A, B = syr2k_init(backend)
        syr2k(alpha, beta, C2, A, B, opts)
        assert torch.allclose(C1, C2, atol=1e-3)

def syrk_init(backend):
    M = 35
    N = 50
    alpha = 1.5
    beta = 1.2
    C = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j+3) % N) / N, (N, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    A = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j+1) % N) / N, (N, M), dtype=np.float64),
        dtype=float_ty(backend)
    )
    return alpha, beta, C, A

def run_syrk(backend, compile_only):
    from npbench_impl.syrk import syrk
    alpha, beta, C1, A = syrk_init(backend)
    N, M = A.shape
    p = {
        'i': parpy.threads(N),
        'k': parpy.threads(256).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = syrk(alpha, beta, C1, A, opts, True)
        assert len(code) > 0
    else:
        syrk(alpha, beta, C1, A, opts)
        opts.seq = True
        alpha, beta, C2, A = syrk_init(backend)
        syrk(alpha, beta, C2, A, opts)
        assert torch.allclose(C1, C2, atol=1e-3)

def trisolv_init(backend):
    N = 2000
    L = torch.tensor(
        np.fromfunction(lambda i,j: (i+N-j+1)*2 / N, (N, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    x = torch.full((N,), -999, dtype=float_ty(backend))
    b = torch.tensor(
        np.fromfunction(lambda i: i, (N,), dtype=np.float64),
        dtype=float_ty(backend)
    )
    return L, x, b

def run_trisolv(backend, compile_only):
    from npbench_impl.trisolv import trisolv
    L, x, b1 = trisolv_init(backend)
    N, N = L.shape
    p = {
        'N': parpy.threads(N),
        'reduce': parpy.threads(256).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = trisolv(L, x, b1, opts, True)
        assert len(code) > 0
    else:
        trisolv(L, x, b1, opts)
        opts.seq = True
        L, x, b2 = trisolv_init(backend)
        trisolv(L, x, b2, opts)
        assert torch.allclose(b1, b2, atol=1e-3)

def trmm_init(backend):
    M = 65
    N = 80
    alpha = 1.5
    A = torch.tensor(
        np.fromfunction(lambda i,j: ((i*j) % M) / M, (M, M), dtype=np.float64),
        dtype=float_ty(backend)
    )
    for i in range(M):
        A[i, i] = 1.0
    B = torch.tensor(
        np.fromfunction(lambda i,j: ((N+i-j) % N) / N, (M, N), dtype=np.float64),
        dtype=float_ty(backend)
    )
    return alpha, A, B

def run_trmm(backend, compile_only):
    from npbench_impl.trmm import trmm
    alpha, A, B1 = trmm_init(backend)
    _, N = B1.shape
    p = {
        'j': parpy.threads(N),
        'k': parpy.threads(256).reduce()
    }
    opts = par_opts(backend, p)
    if compile_only:
        code = trmm(alpha, A, B1, opts, True)
        assert len(code) > 0
    else:
        trmm(alpha, A, B1, opts)
        opts.seq = True
        alpha, A, B2 = trmm_init(backend)
        trmm(alpha, A, B2, opts)
        assert torch.allclose(B1, B2, atol=1e-3)

def vadv_init(backend):
    torch.manual_seed(1234)
    I = 60
    J = 60
    K = 40
    dtr_stage = 3.0 / 20.0
    utens_stage = torch.rand(I, J, K)
    u_stage = torch.rand(I, J, K)
    wcon = torch.rand(I+1, J, K)
    u_pos = torch.rand(I, J, K)
    utens = torch.rand(I, J, K)
    return dtr_stage, utens_stage, u_stage, wcon, u_pos, utens

def run_vadv(backend, compile_only):
    from npbench_impl.vadv import vadv
    dtr_stage, utens_stage1, u_stage, wcon, u_pos, utens = vadv_init(backend)
    I, J, _ = utens_stage1.shape
    p = {'I': parpy.threads(I), 'J': parpy.threads(J)}
    opts = par_opts(backend, p)
    if compile_only:
        code = vadv(utens_stage1, u_stage, wcon, u_pos, utens, dtr_stage, opts, True)
        assert len(code) > 0
    else:
        vadv(utens_stage1, u_stage, wcon, u_pos, utens, dtr_stage, opts)
        opts.seq = True
        dtr_stage, utens_stage2, u_stage, wcon, u_pos, utens = vadv_init(backend)
        vadv(utens_stage2, u_stage, wcon, u_pos, utens, dtr_stage, opts)
        assert torch.allclose(utens_stage1, utens_stage2, atol=1e-3)

run_funs = [
    run_adi,
    run_arc_distance,
    run_azimint_naive,
    run_cholesky,
    run_compute,
    run_conv2d_bias,
    run_correlation,
    run_crc16,
    run_deriche,
    run_durbin,
    run_fdtd2d,
    run_floyd_warshall,
    run_go_fast,
    run_gramschmidt,
    run_heat_3d,
    run_jacobi_1d,
    run_jacobi_2d,
    run_lenet,
    run_lu,
    run_ludcmp,
    run_nussinov,
    run_resnet,
    run_sselfeng,
    run_seidel_2d,
    run_softmax,
    run_spmv,
    run_symm,
    run_syr2k,
    run_syrk,
    run_trisolv,
    run_trmm,
    run_vadv,
]
compile_only_funs = [
    run_cavity_flow,
    run_channel_flow,
    run_hdiff,
    run_mlp,
]
all_funs = run_funs + compile_only_funs

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('fn', run_funs)
def test_compile_and_run(backend, fn):
    def helper():
        fn(backend, False)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('fn', all_funs)
def test_compile_only(backend, fn):
    fn(backend, True)
