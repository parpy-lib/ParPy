import parir
import pytest
import torch

from common import *

@parir.jit
def normalize_rows(t, nrows, ncols):
    parir.label('i')
    for i in range(nrows):
        parir.label('j1')
        s = parir.sum(t[i, :])

        parir.label('j2')
        t[i, :] /= s

def normalize_wrap(t, opts):
    nrows, ncols = t.shape
    out = t.clone()
    normalize_rows(out, nrows, ncols, opts=opts)
    return out

@pytest.mark.parametrize('backend', compiler_backends)
def test_normalize_single_row(backend):
    def helper():
        t = torch.ones((1, 1024), dtype=torch.float32)
        y1 = torch.nn.functional.normalize(t, p=1, dim=1)
        p = { "i": parir.threads(256) }
        y2 = normalize_wrap(t, par_opts(backend, p))
        assert torch.allclose(y1, y2, 1e-5)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_normalize_multirow(backend):
    def helper():
        t = torch.ones((256, 1024), dtype=torch.float32)
        y1 = torch.nn.functional.normalize(t, p=1, dim=1)
        p = {
            "i": parir.threads(256),
            "j1": parir.threads(128).reduce(),
            "j2": parir.threads(128)
        }
        y2 = normalize_wrap(t, par_opts(backend, p))
        assert torch.allclose(y1, y2, 1e-5)
    run_if_backend_is_enabled(backend, helper)

def normalize_rows_no_annot(t, nrows, ncols):
    parir.label('i')
    for i in range(nrows):
        s = parir.float32(0.0)
        parir.label('j1')
        for j in range(ncols):
            s = s + t[i, j]

        parir.label('j2')
        for j in range(ncols):
            t[i, j] = t[i, j] / s

@pytest.mark.parametrize('backend', compiler_backends)
def test_normalize_print_ast(backend):
    args = [
        torch.ones((256, 1024), dtype=torch.float32),
        256,
        1024
    ]
    p = {
        "i": parir.threads(256),
        "j1": parir.threads(128).reduce(),
        "j2": parir.threads(128)
    }
    s = parir.print_compiled(normalize_rows_no_annot, args, par_opts(backend, p))
    assert len(s) != 0
