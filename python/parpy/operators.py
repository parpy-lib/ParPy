from builtins import abs
from numpy import cos, exp, inf, log, sin, sqrt, tanh
from numpy import arctan2 as atan2
import builtins
import numpy as np
import contextlib

gpu = contextlib.nullcontext()

def min(x, y=None, axis=None):
    if y is None:
        if axis is None:
            return np.min(x)
        else:
            return np.min(x, axis=axis)
    else:
        assert axis is None
        return np.minimum(x, y)

def max(x, y=None, axis=None):
    if y is None:
        if axis is None:
            return np.max(x)
        else:
            return np.max(x, axis=axis)
    else:
        assert axis is None
        return np.maximum(x, y)

def sum(x, axis=None):
    if axis is None:
        return np.sum(x)
    else:
        return np.sum(x, axis=axis)

def prod(x, axis=None):
    if axis is None:
        return np.prod(x)
    else:
        return np.prod(x, axis=axis)

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
    assert x is not None, "parpy.label expects one argument"
