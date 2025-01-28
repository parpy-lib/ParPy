import parir
from parir import ParKind
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_nested_annotation():
    @parir.jit
    def copy(dst, src, N):
        for i in range(N):
            dst[i] = src[i]

    t1 = torch.ones(1024, dtype=torch.float32, device='cuda')
    t2 = torch.zeros(1024, dtype=torch.float32, device='cuda')
    p = {'i': [ParKind.GpuThreads(256)]}
    copy(t2, t1, 1024, parallelize=p)
    assert torch.allclose(t1, t2)
