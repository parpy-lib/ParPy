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
        dims = ",".join([str(n) for n in arg.shape])
        return f"<{arg.dtype},{dims}>"
    elif isinstance(arg, int) or isinstance(arg, float):
        return f"<{type(arg)}>"
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

def generate_function_key(fn, args, kwargs):
    code = inspect.getsource(fn)
    key_str = f"{code}+{print_type_signature(args)}+{print_par_kwargs(kwargs)}"
    h = hashlib.new("sha256")
    h.update(key_str.encode('ascii'))
    return h.hexdigest()
