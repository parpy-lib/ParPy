import parir
from parir import ParKind
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_record_args():
    @parir.jit
    def dummy(x, y):
        for i in range(1):
            y[i] = x["a"][0] + x["b"][0]
    x = {
        'a': torch.tensor([4], dtype=torch.int64, device='cuda'),
        'b': torch.tensor([2], dtype=torch.int64, device='cuda')
    }
    y = torch.tensor([0], dtype=torch.int32, device='cuda')
    p = {'i': [ParKind.GpuThreads(2)]}
    dummy(x, y, parallelize=p, cache=False)
    assert y[0] == 6
