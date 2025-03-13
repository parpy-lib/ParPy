# Code for producing a key for a particular function call. This key is used to
# determine whether a particular version of a function has been compiled before
# or not.

from . import parir

import inspect
import hashlib
import os
import torch

def arg_to_string(arg):
    if isinstance(arg, torch.Tensor):
        if arg.ndim == 0:
            return f"({arg.dtype}){arg.item()}"
        else:
            dims = ",".join([str(n) for n in arg.shape])
            return f"<{arg.dtype},{dims}>"
    elif isinstance(arg, int) or isinstance(arg, float):
        return f"({type(arg)}){arg}"
    elif isinstance(arg, dict):
        return ",".join([f"{k}:{arg_to_string(v)}" for k, v in arg.items()])
    else:
        return str(type(arg))

def print_type_signature(args):
    return ",".join([arg_to_string(arg) for arg in args])

def print_par_kwargs(kwargs):
    if "parallelize" in kwargs and kwargs["parallelize"] is not None:
        return ",".join([f"{k}:{v}" for k, v in kwargs["parallelize"].items()])
    else:
        return ""

def generate_quick_function_key(fn, args, kwargs):
    return f"{fn.__name__}+{print_type_signature(args)}+{print_par_kwargs(kwargs)}"

def generate_function_key(fn, args, kwargs):
    h = hashlib.new("sha256")
    h.update(generate_quick_function_key(fn, args, kwargs).encode("ascii"))
    h.update(inspect.getsource(fn).encode("ascii"))
    return h.hexdigest()

def generate_code_key(code):
    h = hashlib.new("sha256")
    h.update(code.encode('ascii'))
    return h.hexdigest()
