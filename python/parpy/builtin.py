from numpy import inf
import numpy as np
import contextlib

# Binary operators

def maximum(x, y):
    return np.maximum(x, y)

def minimum(x, y):
    return np.minimum(x, y)

# Built-in utility functions for controlling the generated code

gpu = contextlib.nullcontext()

def convert(e, ty):
    return e

def label(x):
    assert x is not None, "parpy.label expects one argument"

def inline(e):
    pass

def ranges(*args):
    def interpret_range(arg):
        if isinstance(arg, tuple):
            if len(arg) > 0 and len(arg) <= 3:
                return range(*arg)
            else:
                raise RuntimeError(f"Invalid number of arguments to range: {len(arg)}")
        else:
            return range(arg)
    ranges = [interpret_range(a) for a in args]
    if len(ranges) == 1:
        return ranges[0]
    else:
        import itertools
        return itertools.product(*ranges)

def static_backend_eq(x):
    return False

def static_types_eq(l, r):
    return l == r

def static_fail(s):
    raise RuntimeError(s)

def alloc_shared(shape, dtype):
    pass

def sync():
    pass
