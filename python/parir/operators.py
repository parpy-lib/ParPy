from builtins import min, max, abs
from math import atan2, cos, exp, inf, log, sin, sqrt, tanh
import contextlib

gpu = contextlib.nullcontext()

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
