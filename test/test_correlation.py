import numpy as np
import parpy
import pytest
import warnings

from common import *

torch.manual_seed(1234)

@parpy.jit
def correlation(corr, data, M):
    parpy.label('i')
    for i in range(M-1):
        parpy.label('j')
        for j in range(i+1, M):
            parpy.label('k')
            corr[i, j] = parpy.reduce.sum(data[:, i] * data[:, j])
            corr[j, i] = corr[i, j]

def correlation_wrap(M, N, opts, run):
    data = np.fromfunction(lambda i, j: (i*j)/M+i, (N,M), dtype=np.float32)
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(N) * stddev
    corr = np.eye(M, dtype=np.float32)
    if run:
        corr2 = np.eye(M, dtype=np.float32)
        for i in range(M-1):
            corr[i+1:M,i] = corr[i,i+1:M] = data[:,i] @ data[:,i+1:M]
        correlation(corr2, data, M, opts=opts)
        assert np.allclose(corr, corr2, atol=1e-4)
    else:
        code = parpy.print_compiled(correlation, [corr, data, M], opts)
        assert len(code) > 0

parallelizations = [
    {'i': parpy.threads(499)},
    {'i': parpy.threads(499), 'j': parpy.threads(64)},
    {'i': parpy.threads(499), 'k': parpy.threads(128)},
    {'i': parpy.threads(499), 'j': parpy.threads(64), 'k': parpy.threads(128)},
]

@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('p', parallelizations)
def test_correlation_run(backend, p):
    def helper():
        M = 500
        N = 600
        opts = par_opts(backend, p)
        correlation_wrap(M, N, opts, True)
    run_if_backend_is_enabled(backend, helper)


@pytest.mark.parametrize('backend', compiler_backends)
@pytest.mark.parametrize('p', parallelizations)
def test_correlation_compiles(backend, p):
    M = 500
    N = 600
    opts = par_opts(backend, p)
    correlation_wrap(M, N, opts, False)
