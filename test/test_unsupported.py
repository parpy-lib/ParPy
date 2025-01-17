# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import parir
import pytest
import torch

def assert_runtime_error_on_jit(fn):
    with pytest.raises(RuntimeError):
        parir.jit(fn)

def add_slice(x, y, out, N):
    out[:N] = x[:N] + y[:N]
def test_slicing_rejected():
    assert_runtime_error_on_jit(add_slice)

def while_fun(x, y, N):
    i = 0
    while i < N:
        y[i] = x[i]
        i += 1
def test_while_rejected():
    assert_runtime_error_on_jit(while_fun)

def for_steps(x, y, N):
    for i in range(0, N, 2):
        y[i] = x[i]
        y[i+1] = x[i+1]
def test_for_steps_rejected():
    assert_runtime_error_on_jit(for_steps)

def for_else(x, y, N):
    for i in range(N):
        y[i] = x[i]
    else:
        y[0] += 1
def test_for_else_rejected():
    assert_runtime_error_on_jit(for_else)
