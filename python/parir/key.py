# Code for producing a key for a particular function call. This key is used to
# determine whether a particular version of a function has been compiled before
# or not.

from . import parir

import base64
import torch

def print_type_signature(args):
    def arg_type_to_string(arg):
        if isinstance(arg, torch.Tensor):
            return f"<{arg.dtype}>"
        elif isinstance(arg, int) or isinstance(arg, float):
            # If we get an integer or float literal value, we include this
            # directly in the key to be able to consider it in the compilation
            # process.
            return f"{arg}"
        else:
            return str(type(arg))
    return ",".join([arg_type_to_string(arg) for arg in args])

def print_par_arg(key, values):
    return f"{key}:{sorted(values)}"

def print_kwargs(kwargs):
    return ",".join(kwargs)

def generate_function_key(fn, args, kwargs):
    key_str = f"{fn.__name__}+{print_type_signature(args)}+{print_kwargs(kwargs)}"
    return base64.urlsafe_b64encode(key_str.encode('ascii')).decode('ascii')
