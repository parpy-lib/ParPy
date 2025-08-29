import parpy
import pytest
import torch

from common import *

torch.manual_seed(1234)

def collatz(out, N):
    parpy.label('i')
    for i in range(1, N+1):
        v = i
        count = 0
        while v > 1:
            if v % 2 == 0:
                v = v / 2
            else:
                v = 3 * v + 1
            count = count + 1
        out[i] = count

def collatz_wrap(N, opts=None):
    out = torch.zeros(N+1, dtype=torch.int32)
    if opts is None:
        collatz(out, N)
    else:
        parpy.jit(collatz)(out, N, opts=opts)
    return out

@pytest.mark.parametrize('backend', compiler_backends)
def test_collatz_gpu(backend):
    def helper():
        N = 1000
        expected = collatz_wrap(N)
        p = {'i': parpy.threads(256)}
        actual = collatz_wrap(N, par_opts(backend, p))
        assert torch.allclose(expected, actual)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_collatz_compiles_with_parallelism(backend):
    N = 1000
    out = torch.zeros(N+1, dtype=torch.int32)
    p = {'i': parpy.threads(128)}
    s = parpy.print_compiled(collatz, [out, N], par_opts(backend, p))
    assert len(s) != 0
