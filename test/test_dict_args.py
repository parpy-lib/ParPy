import parpy
import pytest
import torch

from common import *

@pytest.mark.parametrize('backend', compiler_backends)
def test_dict_args(backend):
    @parpy.jit
    def dummy(x, y):
        with parpy.gpu:
            y[0] = x["a"][0] + x["b"][0]
    def helper():
        x = {
            'a': torch.tensor([4], dtype=torch.int64),
            'b': torch.tensor([2], dtype=torch.int64)
        }
        y = torch.tensor([0], dtype=torch.int32)
        dummy(x, y, opts=par_opts(backend, {}))
        assert y[0] == 6
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_dict(backend):
    @parpy.jit
    def dummy(x, y):
        with parpy.gpu:
            y[0] = x['a']['b']
    def helper():
        x = {
            'a': {
                'b': torch.tensor([3], dtype=torch.int64)
            }
        }
        y = torch.tensor([0], dtype=torch.int32)
        with pytest.raises(RuntimeError) as e_info:
            dummy(x, y, opts=par_opts(backend, {}))
        assert e_info.match(r".*nested dictionary.*")
    run_if_backend_is_enabled(backend, helper)
