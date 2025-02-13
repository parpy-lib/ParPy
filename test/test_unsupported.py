# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import parir
import pytest
import torch

def test_slicing_rejected():
    with pytest.raises(RuntimeError):
        @parir.jit
        def add_slice(x, y, out, N):
            out[:N] = x[:N] + y[:N]

def test_while_else_rejected():
    with pytest.raises(RuntimeError):
        @parir.jit
        def while_fun(x, y, N):
            i = 0
            while i < N:
                y[i] = x[i]
                i += 1
            else:
                y[i] = 0.0

def test_for_else_rejected():
    with pytest.raises(RuntimeError):
        @parir.jit
        def for_else(x, y, N):
            for i in range(N):
                y[i] = x[i]
            else:
                y[0] += 1

def test_with_unsupported_context():
    with pytest.raises(RuntimeError):
        @parir.jit
        def with_context():
            with 5:
                pass

def test_with_as():
    with pytest.raises(RuntimeError):
        @parir.jit
        def with_as():
            with parir.gpu as x:
                a = x + 1
