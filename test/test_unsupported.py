# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import numpy as np
import parpy
import pytest
import torch

from common import *

def test_while_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def while_fun(x, y, N):
            i = 0
            while i < N:
                y[i] = x[i]
                i += 1
            else:
                y[i] = 0.0
    assert e_info.match(r".*lines 16-20.*")

def test_for_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def for_else(x, y, N):
            for i in range(N):
                y[i] = x[i]
            else:
                y[0] += 1
    assert e_info.match(r".*lines 27-30.*")

def test_with_unsupported_context():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def with_context():
            with 5:
                pass
    assert e_info.match(r".*lines 37-38.*")

def test_with_as():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def with_as():
            with parpy.gpu as x:
                a = x + 1
    assert e_info.match(r".*lines 45-46.*")

def test_dict_with_non_string_keys():
    @parpy.jit
    def dict_arg(a):
        with parpy.gpu:
            a["x"] = a["y"]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 'y': 4, 3: 5})
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")

def test_dict_with_int_key():
    @parpy.jit
    def dict_arg(a):
        with parpy.gpu:
            a["x"] = a[2]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 2: 4})
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")

def test_invalid_max_thread_blocks_per_cluster():
    opts = parpy.CompileOptions()
    with pytest.raises(RuntimeError) as e_info:
        opts.max_thread_blocks_per_cluster = 3
    assert e_info.match(r".*number of thread blocks per cluster must be a power of two.*")

def test_empty_return():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def empty_return():
            return
    assert e_info.match(r"Empty return statements are not supported")

def test_return_in_main_function():
    enabled_backends = [b for b in compiler_backends if parpy.backend.is_enabled(b)]
    if len(enabled_backends) == 0:
        pytest.skip("No available backends to use for compilation")
    else:
        @parpy.jit
        def f_return(x):
            with parpy.gpu:
                y = parpy.operators.sum(x[:])
                return y
        backend = enabled_backends[0]
        with pytest.raises(RuntimeError) as e_info:
            f_return(np.ndarray(10), opts=par_opts(backend, {}))
        assert e_info.match(r"The called function f_return cannot return a value")
