from builtins import abs
from torch import atan2, cos, exp, inf, log, sin, sqrt, tanh
import builtins
import torch
import contextlib

gpu = contextlib.nullcontext()

def min(x, y=None, axis=None):
    if y is None:
        if axis is None:
            return torch.min(x)
        else:
            return torch.min(x, dim=axis).values
    else:
        assert axis is None
        return builtins.min(x, y)

def max(x, y=None, axis=None):
    if y is None:
        if axis is None:
            return torch.max(x)
        else:
            return torch.max(x, dim=axis).values
    else:
        assert axis is None
        return builtins.max(x, y)

def sum(x, axis=None):
    if axis is None:
        return torch.sum(x)
    else:
        return torch.sum(x, dim=axis)

def prod(x, axis=None):
    if axis is None:
        return torch.prod(x)
    else:
        return torch.prod(x, dim=axis)

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

def uint8(x):
    return int(x)

def uint16(x):
    return int(x)

def uint32(x):
    return int(x)

def uint64(x):
    return int(x)

def label(x):
    assert x is not None, "parir.label expects one argument"
