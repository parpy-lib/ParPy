import parir
import pytest
import torch

@parir.jit
def normalize_rows(t, nrows, ncols):
    parir.label('i')
    for i in range(nrows):
        parir.label('j1')
        s = parir.sum(t[i, :])

        parir.label('j2')
        t[i, :] /= s

def normalize_wrap(t, p=None):
    nrows, ncols = t.shape
    out = t.clone()
    normalize_rows(out, nrows, ncols, parallelize=p, cache=False)
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_normalize_single_row():
    t = torch.ones((1, 1024), dtype=torch.float32, device='cuda')
    y1 = torch.nn.functional.normalize(t, p=1, dim=1)
    p = { "i": parir.threads(256) }
    y2 = normalize_wrap(t, p)
    assert torch.allclose(y1, y2, 1e-5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_normalize_multirow():
    t = torch.ones((256, 1024), dtype=torch.float32, device='cuda')
    y1 = torch.nn.functional.normalize(t, p=1, dim=1)
    p = {
        "i": parir.threads(256),
        "j1": parir.threads(128).reduce(),
        "j2": parir.threads(128)
    }
    y2 = normalize_wrap(t, p)
    assert torch.allclose(y1, y2, 1e-5)

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

def test_normalize_print_ast():
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
    s = parir.print_compiled(normalize_rows_no_annot, args, p)
    assert len(s) != 0
