from builtins import abs
from torch import atan2, cos, exp, inf, log, sin, sqrt, tanh
import builtins
import torch
import contextlib

gpu = contextlib.nullcontext()

def min(x, y, axis=None):
    if y is None:
        return torch.min(x, axis=axis)
    else:
        assert axis is None
        return builtins.min(x, y)

def max(x, y, axis=None):
    if y is None:
        return torch.max(x, axis=axis)
    else:
        assert axis is None
        return builtins.max(x, y)

def sum(x, axis=None):
    return torch.sum(x, axis=axis)

def float16(x):
    return float(x)

def float32(x):
    return float(x)

def float64(x):
    return float(x)

def int8(x):
    return int(x)

def int16(x):
    return int(x)

def int32(x):
    return int(x)

def int64(x):
    return int(x)

def label(x):
    assert x is not None, "parir.label expects one argument"
