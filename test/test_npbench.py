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

import prickle
import pytest
import torch
import numpy as np

from common import *
import npbench_impl

def float_ty(backend):
    if backend == prickle.CompileBackend.Metal:
        return torch.float32
    else:
        return torch.float64

def complex_ty(backend):
    if backend == prickle.CompileBackend.Metal:
        return torch.complex64
    else:
        return torch.complex128

def adi_init(backend):
    TSTEPS = 5
    N = 100
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=np.float64)
    return TSTEPS, N, torch.tensor(u, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_adi(backend):
    def helper():
        if backend == prickle.CompileBackend.Metal:
            pytest.skip("Skipped due to lack of support for 64-bit floats in Metal")
        from npbench_impl.adi import adi
        TSTEPS, N, u1 = adi_init(backend)
        p = { 'i': prickle.threads(N-2) }
        opts = par_opts(backend, p)
        a = adi(TSTEPS, N, u1, opts)
        TSTEPS, N, u2 = adi_init(backend)
        b = adi(TSTEPS, N, u2, opts)
        assert torch.allclose(a, b, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def arc_distance_init(backend):
    torch.manual_seed(1234)
    N = 100000
    t0 = torch.rand((N,), dtype=float_ty(backend))
    p0 = torch.rand((N,), dtype=float_ty(backend))
    t1 = torch.rand((N,), dtype=float_ty(backend))
    p1 = torch.rand((N,), dtype=float_ty(backend))
    return N, t0, p0, t1, p1

@pytest.mark.parametrize('backend', compiler_backends)
def test_arc_distance(backend):
    def helper():
        from npbench_impl.arc_distance import arc_distance
        N, t0, p0, t1, p1 = arc_distance_init(backend)
        p = {'i': prickle.threads(N)}
        opts = par_opts(backend, p)
        a = arc_distance(t0, p0, t1, p1, opts)
        opts.seq = True
        b = arc_distance(t0, p0, t1, p1, opts)
        assert torch.allclose(a, b, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def azimint_naive_init(backend):
    npt = 100
    N = 40
    data = torch.rand(N, dtype=float_ty(backend))
    radius = torch.rand(N, dtype=float_ty(backend))
    return npt, data, radius

@pytest.mark.parametrize('backend', compiler_backends)
def test_azimint_naive(backend):
    def helper():
        from npbench_impl.azimint_naive import azimint_naive
        npt, data, radius = azimint_naive_init(backend)
        p = {
            'i': prickle.threads(npt),
            'ix': prickle.threads(1024).reduce(),
            'j': prickle.threads(1024).reduce()
        }
        opts = par_opts(backend, p)
        a = azimint_naive(data, radius, npt, opts)
        opts.seq = True
        b = azimint_naive(data, radius, npt, opts)
        assert torch.allclose(a, b, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.skip("Fails due to a bug in sequential execution")
@pytest.mark.parametrize('backend', compiler_backends)
def test_cavity_flow(backend):
    def helper():
        from npbench_impl.cavity_flow import cavity_flow
        nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu = cavity_flow_init(backend)
        p = {'ny': prickle.threads(ny), 'nx': prickle.threads(nx)}
        opts = par_opts(backend, p)
        cavity_flow(nx, ny, nt, nit, u1, v1, dt, dx, dy, p1, rho, nu, opts)
        opts.seq = True
        nx, ny, nt, nit, u2, v2, dt, dx, dy, p2, rho, nu = cavity_flow_init(backend)
        cavity_flow(nx, ny, nt, nit, u2, v2, dt, dx, dy, p2, rho, nu, opts)
        assert torch.allclose(u1, u2, atol=1e-3)
        assert torch.allclose(v1, v2, atol=1e-3)
        assert torch.allclose(p1, p2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.skip("Fails due to a bug in sequential execution")
@pytest.mark.parametrize('backend', compiler_backends)
def test_channel_flow(backend):
    def helper():
        from npbench_impl.channel_flow import channel_flow
        nit, u1, v1, dt, dx, dy, p1, rho, nu, F = channel_flow_init(backend)
        ny, nx = u1.shape
        p = {'ny': prickle.threads(ny), 'nx': prickle.threads(nx)}
        opts = par_opts(backend, p)
        channel_flow(nit, u1, v1, dt, dx, dy, p1, rho, nu, F, opts)
        opts.seq = True
        nit, u2, v2, dt, dx, dy, p2, rho, nu, F = channel_flow_init(backend)
        channel_flow(nit, u2, v2, dt, dx, dy, p2, rho, nu, F, opts)
        assert torch.allclose(u1, u2, atol=1e-3)
        assert torch.allclose(v1, v2, atol=1e-3)
        assert torch.allclose(p1, p2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_cholesky(backend):
    def helper():
        from npbench_impl.cholesky import cholesky
        p = { 'k': prickle.threads(256).reduce() }
        opts = par_opts(backend, p)
        A1 = cholesky_init(backend)
        cholesky(A1, opts)
        A2 = cholesky_init(backend)
        opts.seq = True
        cholesky(A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_compute(backend):
    def helper():
        from npbench_impl.compute import compute
        array_1, array_2, a, b, c = compute_init(backend)
        N, _ = array_1.shape
        p = {
            'i': prickle.threads(N),
            'j': prickle.threads(1024)
        }
        opts = par_opts(backend, p)
        r1 = compute(array_1, array_2, a, b, c, opts)
        opts.seq = True
        r2 = compute(array_1, array_2, a, b, c, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_conv2d_bias(backend):
    def helper():
        from npbench_impl.conv2d_bias import conv2d_bias
        input, weights, bias = conv2d_init(backend)
        H_out = input.shape[1] - weights.shape[0] + 1
        W_out = input.shape[2] - weights.shape[0] + 1
        p = {'i': prickle.threads(H_out), 'j': prickle.threads(W_out)}
        opts = par_opts(backend, p)
        r1 = conv2d_bias(input, weights, bias, opts)
        opts.seq = True
        r2 = conv2d_bias(input, weights, bias, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def correlation_init(backend):
    M = 500
    N = 600
    float_n = torch.tensor(N, dtype=float_ty(backend))
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
    return float_n, torch.tensor(data, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_correlation(backend):
    def helper():
        from npbench_impl.correlation import correlation
        float_n, data = correlation_init(backend)
        _, M = data.shape
        p = {
            'i': prickle.threads(M-1),
            'j': prickle.threads(256),
        }
        opts = par_opts(backend, p)
        r1 = correlation(M, float_n, data, opts)
        float_n, data = correlation_init(backend)
        opts.seq = True
        r2 = correlation(M, float_n, data, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def covariance_init(backend):
    M = 500
    N = 600
    float_n = torch.tensor(N, dtype=float_ty(backend))
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=np.float64)
    return float_n, torch.tensor(data, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_covariance(backend):
    def helper():
        from npbench_impl.covariance import covariance
        float_n, data = covariance_init(backend)
        _, M = data.shape
        p = {
            'i': prickle.threads(M),
            'j': prickle.threads(256),
        }
        opts = par_opts(backend, p)
        r1 = covariance(M, float_n, data, opts)
        float_n, data = covariance_init(backend)
        opts.seq = True
        r2 = covariance(M, float_n, data, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def crc16_init(backend):
    torch.manual_seed(1234)
    N = 1600
    data = torch.randint(0, 256, (N,), dtype=torch.int32)
    return data

@pytest.mark.parametrize('backend', compiler_backends)
def test_crc16(backend):
    def helper():
        from npbench_impl.crc16 import crc16
        data = crc16_init(backend)
        opts = par_opts(backend, {})
        r1 = crc16(data, opts)
        data = crc16_init(backend)
        opts.seq = True
        r2 = crc16(data, opts)
        assert r1 == r2
    run_if_backend_is_enabled(backend, helper)

def deriche_init(backend):
    W = 400
    H = 200
    alpha = 0.25
    img_in = np.fromfunction(lambda i, j:
                             ((313 * i + 991 * j) % 65536) / 65535.0, (W, H),
                             dtype=np.float64)
    return alpha, torch.tensor(img_in, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_deriche(backend):
    def helper():
        from npbench_impl.deriche import deriche
        alpha, img_in = deriche_init(backend)
        W, H = img_in.shape
        p = {
            'i': prickle.threads(W),
            'j': prickle.threads(H)
        }
        opts = par_opts(backend, p)
        r1 = deriche(alpha, img_in, opts)
        opts.seq = True
        r2 = deriche(alpha, img_in, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def durbin_init(backend):
    N = 1000
    r = np.fromfunction(lambda i: N + 1 - i, (N,))
    return torch.tensor(r, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_durbin(backend):
    def helper():
        if backend == prickle.CompileBackend.Metal:
            pytest.skip("Skipped due to lack of support for 64-bit floats in Metal")
        from npbench_impl.durbin import durbin
        p = {
            'k_red': prickle.threads(512).reduce(),
            'k': prickle.threads(512)
        }
        opts = par_opts(backend, p)
        r = durbin_init(backend)
        r1 = durbin(r, opts)
        opts.seq = True
        r2 = durbin(r, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_fdtd2d(backend):
    def helper():
        from npbench_impl.fdtd2d import fdtd2d
        TMAX, NX, NY, ex1, ey1, hz1, _fict_ = fdtd2d_init(backend)
        p = {
            'i': prickle.threads(NX-1),
            'j': prickle.threads(1024),
        }
        opts = par_opts(backend, p)
        fdtd2d(TMAX, ex1, ey1, hz1, _fict_, opts)
        opts.seq = True
        TMAX, NX, NY, ex2, ey2, hz2, _fict_ = fdtd2d_init(backend)
        fdtd2d(TMAX, ex2, ey2, hz2, _fict_, opts)
        assert torch.allclose(ex1, ex2, atol=1e-3)
        assert torch.allclose(ey1, ey2, atol=1e-3)
        assert torch.allclose(hz1, hz2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def floyd_warshall_init(backend):
    N = 200
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if (i+j) % 13 == 0 or (i+j) % 7 == 0 or (i+j) % 11 == 0:
                path[i,j] = 999
    return torch.tensor(path), N

@pytest.mark.parametrize('backend', compiler_backends)
def test_floyd_warshall(backend):
    def helper():
        from npbench_impl.floyd_warshall import floyd_warshall
        path1, N = floyd_warshall_init(backend)
        p = {
            'i': prickle.threads(N),
            'j': prickle.threads(N)
        }
        opts = par_opts(backend, p)
        floyd_warshall(path1, N, opts)
        opts.seq = True
        path2, N = floyd_warshall_init(backend)
        floyd_warshall(path2, N, opts)
        assert torch.allclose(path1, path2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def go_fast_init(backend):
    torch.manual_seed(1234)
    N = 2000
    x = torch.rand((N, N), dtype=float_ty(backend))
    return x, N

@pytest.mark.parametrize('backend', compiler_backends)
def test_go_fast(backend):
    def helper():
        from npbench_impl.go_fast import go_fast
        x, N = go_fast_init(backend)
        p = {
            'i': prickle.threads(1024).reduce(),
            'ix': prickle.threads(N),
            'j': prickle.threads(N)
        }
        opts = par_opts(backend, p)
        r1 = go_fast(x, opts)
        opts.seq = True
        r2 = go_fast(x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def gramschmidt_init(backend):
    torch.manual_seed(1234)
    M = 70
    N = 60
    A = torch.rand((M, N), dtype=float_ty(backend))
    while torch.linalg.matrix_rank(A) < N:
        A = torch.rand((M, N), dtype=float_ty(backend))
    return A

@pytest.mark.parametrize('backend', compiler_backends)
def test_gramschmidt(backend):
    def helper():
        from npbench_impl.gramschmidt import gramschmidt
        A = gramschmidt_init(backend)
        M, N = A.shape
        p = {
            'i': prickle.threads(M),
            'i_reduce': prickle.threads(128).reduce(),
            'j': prickle.threads(N)
        }
        opts = par_opts(backend, p)
        Q1, R1 = gramschmidt(A, opts)
        opts.seq = True
        A = gramschmidt_init(backend)
        Q2, R2 = gramschmidt(A, opts)
        assert torch.allclose(Q1, Q2, atol=1e-3)
        assert torch.allclose(R1, R2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def hdiff_init(backend):
    torch.manual_seed(1234)
    I = 64
    J = 64
    K = 60
    in_field = torch.rand((I+4, J+4, K), dtype=float_ty(backend))
    out_field = torch.rand((I, J, K), dtype=float_ty(backend))
    coeff = torch.rand((I, J, K), dtype=float_ty(backend))
    return in_field, out_field, coeff

@pytest.mark.parametrize('backend', compiler_backends)
def test_hdiff_compiles(backend):
    def helper():
        from npbench_impl.hdiff import hdiff
        in_field, out_field, coeff = hdiff_init(backend)
        I, J, K = out_field.shape
        p = {'I': prickle.threads(I), 'J': prickle.threads(J), 'K': prickle.threads(K)}
        opts = par_opts(backend, p)
        hdiff(in_field, out_field, coeff, opts)
    run_if_backend_is_enabled(backend, helper)

def heat_3d_init(backend):
    TSTEPS = 25
    N = 25
    A = np.fromfunction(lambda i,j,k: (i+j+(N-k)) * 10 / N, (N,N,N), dtype=np.float64)
    B = torch.tensor(np.copy(A), dtype=float_ty(backend))
    return TSTEPS, torch.tensor(A, dtype=float_ty(backend)), B

@pytest.mark.parametrize('backend', compiler_backends)
def test_heat_3d(backend):
    def helper():
        from npbench_impl.heat_3d import heat_3d
        p = {
            'i': prickle.threads(64),
            'j': prickle.threads(64),
            'k': prickle.threads(64)
        }
        opts = par_opts(backend, p)
        TSTEPS, A1, B1 = heat_3d_init(backend)
        heat_3d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = heat_3d_init(backend)
        heat_3d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def jacobi_1d_init(backend):
    TSTEPS = 800
    N = 3200
    A = np.fromfunction(lambda i: (i+2) / N, (N,), dtype=np.float64)
    B = np.fromfunction(lambda i: (i+3) / N, (N,), dtype=np.float64)
    fty = float_ty(backend)
    return TSTEPS, torch.tensor(A, dtype=fty), torch.tensor(B, dtype=fty)

@pytest.mark.parametrize('backend', compiler_backends)
def test_jacobi_1d(backend):
    def helper():
        from npbench_impl.jacobi_1d import jacobi_1d
        TSTEPS, A1, B1 = jacobi_1d_init(backend)
        N, = A1.shape
        p = {'i': prickle.threads(N-2)}
        opts = par_opts(backend, p)
        jacobi_1d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = jacobi_1d_init(backend)
        jacobi_1d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def jacobi_2d_init(backend):
    TSTEPS = 50
    N = 150
    A = np.fromfunction(lambda i,j: i*(j+2) / N, (N, N), dtype=np.float64)
    B = np.fromfunction(lambda i,j: i*(j+3) / N, (N, N), dtype=np.float64)
    fty = float_ty(backend)
    return TSTEPS, torch.tensor(A, dtype=fty), torch.tensor(B, dtype=fty)

@pytest.mark.parametrize('backend', compiler_backends)
def test_jacobi_2d(backend):
    def helper():
        from npbench_impl.jacobi_2d import jacobi_2d
        TSTEPS, A1, B1 = jacobi_2d_init(backend)
        N, N = A1.shape
        p = {
            'i': prickle.threads(N-1),
            'j': prickle.threads(N-1),
        }
        opts = par_opts(backend, p)
        jacobi_2d(TSTEPS, A1, B1, opts)
        opts.seq = True
        TSTEPS, A2, B2 = jacobi_2d_init(backend)
        jacobi_2d(TSTEPS, A2, B2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
        assert torch.allclose(B1, B2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_lenet(backend):
    def helper():
        from npbench_impl.lenet import lenet
        input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, C_before_fc1 = lenet_init(backend)
        N, _, _, _ = input.shape
        opts = par_opts(backend, {})
        r1 = lenet(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1, opts)
        opts.seq = True
        r2 = lenet(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, N, C_before_fc1, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_lu(backend):
    def helper():
        from npbench_impl.lu import lu
        p = {'k': prickle.threads(128).reduce()}
        opts = par_opts(backend, p)
        A1 = lu_init(backend)
        lu(A1, opts)
        opts.seq = True
        A2 = lu_init(backend)
        lu(A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def ludcmp_init(backend):
    N = 60
    A = lu_init(backend)
    fn = np.float64(N)
    b = np.fromfunction(lambda i: (i+1)/fn/2.0 + 4.0, (N,), dtype=np.float64)
    return A, torch.tensor(b, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_ludcmp(backend):
    def helper():
        from npbench_impl.ludcmp import ludcmp
        p = {'k': prickle.threads(128).reduce()}
        opts = par_opts(backend, p)
        A1, b = ludcmp_init(backend)
        x1, y1 = ludcmp(A1, b, opts)
        opts.seq = True
        A2, b = ludcmp_init(backend)
        x2, y2 = ludcmp(A2, b, opts)
        assert torch.allclose(x1, x2, atol=1e-3)
        assert torch.allclose(y1, y2, atol=1e-3)
        assert torch.allclose(A1, A2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_mlp(backend):
    def helper():
        from npbench_impl.mlp import mlp
        input, w1, b1, w2, b2, w3, b3 = mlp_init(backend)
        N, _ = input.shape
        p = {
            'i': prickle.threads(N),
            'j': prickle.threads(1024),
        }
        opts = par_opts(backend, p)
        x1 = mlp(input, w1, b1, w2, b2, w3, b3, opts)
        opts.seq = True
        x2 = mlp(input, w1, b1, w2, b2, w3, b3, opts)
        assert torch.allclose(x1, x2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.skip("Fails due to a bug in sequential execution")
@pytest.mark.parametrize('backend', compiler_backends)
def test_nbody(backend):
    def helper():
        from npbench_impl.nbody import nbody
        mass, pos, vel, N, Nt, dt, G, softening = nbody_init(backend)
        p = {
            'N2': prickle.threads(N*N),
            'N': prickle.threads(N),
            'reduce': prickle.threads(64).reduce()
        }
        opts = par_opts(backend, p)
        KE1, PE1 = nbody(mass, pos, vel, N, Nt, dt, G, softening, opts)
        opts.seq = True
        KE2, PE2 = nbody(mass, pos, vel, N, Nt, dt, G, softening, opts)
        assert torch.allclose(KE1, KE2, atol=1e-3)
        assert torch.allclose(PE1, PE2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def nussinov_init(backend):
    N = 40
    seq = np.fromfunction(lambda i: (i+1) % 4, (N,), dtype=np.int32)
    return torch.tensor(seq)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nussinov(backend):
    def helper():
        from npbench_impl.nussinov import nussinov
        opts = par_opts(backend, {})
        seq = nussinov_init(backend)
        N, = seq.shape
        r1 = nussinov(N, seq, opts)
        opts.seq = True
        r2 = nussinov(N, seq, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_resnet(backend):
    def helper():
        from npbench_impl.resnet import resnet
        input, conv1, conv2, conv3 = resnet_init(backend)
        H_out = input.shape[1] - conv1.shape[0] + 1
        W_out = input.shape[2] - conv1.shape[0] + 1
        p = {'i': prickle.threads(H_out), 'j': prickle.threads(W_out)}
        opts = par_opts(backend, p)
        r1 = resnet(input, conv1, conv2, conv3, opts)
        opts.seq = True
        r2 = resnet(input, conv1, conv2, conv3, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_sselfeng(backend):
    def helper():
        from npbench_impl.sselfeng import sselfeng
        neigh_idx, dH, G, D, Sigma1 = sselfeng_init(backend)
        Nkz, NE, NA, Norb, Norb = G.shape
        p = {
            'Nkz': prickle.threads(Nkz),
            'NE': prickle.threads(NE),
            'NA': prickle.threads(NA),
            'threads': prickle.threads(32)
        }
        opts = par_opts(backend, p)
        sselfeng(neigh_idx, dH, G, D, Sigma1, opts)
        opts.seq = True
        neigh_idx, dH, G, D, Sigma2 = sselfeng_init(backend)
        sselfeng(neigh_idx, dH, G, D, Sigma2, opts)
        assert torch.allclose(Sigma1, Sigma2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def seidel_2d_init(backend):
    TSTEPS = 8
    N = 50
    A = np.fromfunction(lambda i,j: (i * (j+2) + 2) / N, (N, N), dtype=np.float64)
    return TSTEPS, N, torch.tensor(A, dtype=float_ty(backend))

@pytest.mark.parametrize('backend', compiler_backends)
def test_seidel_2d(backend):
    def helper():
        from npbench_impl.seidel_2d import seidel_2d
        TSTEPS, N, A1 = seidel_2d_init(backend)
        opts = par_opts(backend, {})
        seidel_2d(TSTEPS, N, A1, opts)
        opts.seq = True
        TSTEPS, N, A2 = seidel_2d_init(backend)
        seidel_2d(TSTEPS, N, A2, opts)
        assert torch.allclose(A1, A2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

def softmax_init(backend):
    torch.manual_seed(1234)
    N = 16
    H = 16
    SM = 128
    x = torch.rand(N, H, SM, SM, dtype=torch.float32)
    return x

@pytest.mark.parametrize('backend', compiler_backends)
def test_softmax(backend):
    def helper():
        from npbench_impl.softmax import softmax
        x = softmax_init(backend)
        N, H, SM, SM = x.shape
        p = {
            'i': prickle.threads(N),
            'j': prickle.threads(H),
            'k': prickle.threads(SM),
            'l': prickle.threads(256),
        }
        opts = par_opts(backend, p)
        r1 = softmax(x, opts)
        opts.seq = True
        r2 = softmax(x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_spmv(backend):
    def helper():
        from npbench_impl.spmv import spmv
        rows, cols, vals, x = spmv_init(backend)
        N, = x.shape
        p = {
            'i': prickle.threads(N-1),
            'j': prickle.threads(64).reduce(),
        }
        opts = par_opts(backend, p)
        r1 = spmv(rows, cols, vals, x, opts)
        opts.seq = True
        r2 = spmv(rows, cols, vals, x, opts)
        assert torch.allclose(r1, r2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_symm(backend):
    def helper():
        from npbench_impl.symm import symm
        alpha, beta, C1, A, B = symm_init(backend)
        M, N = C1.shape
        p = {
            'M': prickle.threads(M),
            'N': prickle.threads(N),
            'i_red': prickle.threads(32).reduce()
        }
        opts = par_opts(backend, p)
        symm(alpha, beta, C1, A, B, opts)
        opts.seq = True
        alpha, beta, C2, A, B = symm_init(backend)
        symm(alpha, beta, C2, A, B, opts)
        assert torch.allclose(C1, C2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_syr2k(backend):
    def helper():
        from npbench_impl.syr2k import syr2k
        alpha, beta, C1, A, B = syr2k_init(backend)
        N, M = A.shape
        p = {
            'i': prickle.threads(N),
            'k': prickle.threads(256).reduce()
        }
        opts = par_opts(backend, p)
        syr2k(alpha, beta, C1, A, B, opts)
        opts.seq = True
        alpha, beta, C2, A, B = syr2k_init(backend)
        syr2k(alpha, beta, C2, A, B, opts)
        assert torch.allclose(C1, C2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_syrk(backend):
    def helper():
        from npbench_impl.syrk import syrk
        alpha, beta, C1, A = syrk_init(backend)
        N, M = A.shape
        p = {
            'i': prickle.threads(N),
            'k': prickle.threads(256).reduce()
        }
        opts = par_opts(backend, p)
        syrk(alpha, beta, C1, A, opts)
        opts.seq = True
        alpha, beta, C2, A = syrk_init(backend)
        syrk(alpha, beta, C2, A, opts)
        assert torch.allclose(C1, C2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_trisolv(backend):
    def helper():
        from npbench_impl.trisolv import trisolv
        L, x, b1 = trisolv_init(backend)
        N, N = L.shape
        p = {
            'N': prickle.threads(N),
            'reduce': prickle.threads(256).reduce()
        }
        opts = par_opts(backend, p)
        trisolv(L, x, b1, opts)
        opts.seq = True
        L, x, b2 = trisolv_init(backend)
        trisolv(L, x, b2, opts)
        assert torch.allclose(b1, b2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_trmm(backend):
    def helper():
        from npbench_impl.trmm import trmm
        alpha, A, B1 = trmm_init(backend)
        _, N = B1.shape
        p = {
            'j': prickle.threads(N),
            'k': prickle.threads(256).reduce()
        }
        opts = par_opts(backend, p)
        trmm(alpha, A, B1, opts)
        opts.seq = True
        alpha, A, B2 = trmm_init(backend)
        trmm(alpha, A, B2, opts)
        assert torch.allclose(B1, B2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)

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

@pytest.mark.parametrize('backend', compiler_backends)
def test_vadv(backend):
    def helper():
        from npbench_impl.vadv import vadv
        dtr_stage, utens_stage1, u_stage, wcon, u_pos, utens = vadv_init(backend)
        I, J, _ = utens_stage1.shape
        p = {'I': prickle.threads(I), 'J': prickle.threads(J)}
        opts = par_opts(backend, p)
        vadv(utens_stage1, u_stage, wcon, u_pos, utens, dtr_stage, opts)
        opts.seq = True
        dtr_stage, utens_stage2, u_stage, wcon, u_pos, utens = vadv_init(backend)
        vadv(utens_stage2, u_stage, wcon, u_pos, utens, dtr_stage, opts)
        assert torch.allclose(utens_stage1, utens_stage2, atol=1e-3)
    run_if_backend_is_enabled(backend, helper)
