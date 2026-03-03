import parpy
import pytest
import torch

from common import *

@parpy.jit
def conv2d_kernel(inputs, weights, output, H_out, W_out, N, C_in, C_out, K):
    parpy.label('H_out')
    for i in range(H_out):
        parpy.label('W_out')
        for j in range(W_out):
            output[:,i,j,:] = 0.0
            for a in range(N):
                for b in range(K):
                    for c in range(K):
                        for d in range(C_in):
                            parpy.label('C_out')
                            for e in range(C_out):
                                output[a,i,j,e] += inputs[a,i+b,j+c,d] * weights[b,c,d,e]

@pytest.mark.parametrize('backend', compiler_backends)
def test_conv2d_runs(backend):
    def helper():
        N = 8
        C_in = 3
        C_out = 16
        K = 2
        H = 32
        W = 32
        H_out = H - K + 1
        W_out = W - K + 1
        input = torch.randn((N, H, W, C_in), dtype=torch.float32)
        weights = torch.randn((K, K, C_in, C_out), dtype=torch.float32)
        output = torch.zeros((N, H_out, W_out, C_out), dtype=torch.float32)
        args = [input, weights, output, H_out, W_out, N, C_in, C_out, K]
        opts = par_opts(backend, {
            'H_out': parpy.threads(H_out),
            'W_out': parpy.threads(W_out),
            'C_out': parpy.threads(C_out),
        })
        conv2d_kernel(*args, opts=opts)
    run_if_backend_is_enabled(backend, helper)
