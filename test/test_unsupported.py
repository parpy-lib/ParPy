# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import numpy as np
import parir
import pytest
import torch

def test_while_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def while_fun(x, y, N):
            i = 0
            while i < N:
                y[i] = x[i]
                i += 1
            else:
                y[i] = 0.0
    assert e_info.match(r".*lines 14-18.*")

def test_for_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def for_else(x, y, N):
            for i in range(N):
                y[i] = x[i]
            else:
                y[0] += 1
    assert e_info.match(r".*lines 25-28.*")

def test_with_unsupported_context():
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def with_context():
            with 5:
                pass
    assert e_info.match(r".*lines 35-36.*")

def test_with_as():
    with pytest.raises(RuntimeError) as e_info:
        @parir.jit
        def with_as():
            with parir.gpu as x:
                a = x + 1
    assert e_info.match(r".*lines 43-44.*")

def test_dict_with_non_string_keys():
    @parir.jit
    def dict_arg(a):
        with parir.gpu:
            a["x"] = a["y"]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 'y': 4, 3: 5})
    assert e_info.match(r".*non-string key.*")

def test_dict_with_int_key():
    @parir.jit
    def dict_arg(a):
        with parir.gpu:
            a["x"] = a[2]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 2: 4})
    assert e_info.match(r".*non-string key.*")

@parir.jit
def add(a, b, c, N):
    parir.label("N")
    for i in range(N):
        c[i] = a[i] + b[i]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_add_cpu_args():
    a = torch.randn(10)
    b = torch.randn(10)
    c = torch.randn(10)
    with pytest.raises(RuntimeError) as e_info:
        add(a, b, c, 10, parallelize={'N': [parir.threads(10)]}, cache=False)
    assert e_info.match(r".*is on device.*expected to be on device.*")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_add_numpy_array_arg():
    a = torch.randn(10, device='cuda')
    b = torch.randn(10, device='cuda')
    c = np.ndarray(10)
    with pytest.raises(RuntimeError) as e_info:
        add(a, b, c, 10, parallelize={'N': [parir.threads(10)]}, cache=False)
    assert e_info.match(r".*unsupported type.*")
