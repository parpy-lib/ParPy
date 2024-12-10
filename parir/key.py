# Code for producing a key for a particular function call. This key is used to
# determine whether a particular version of a function has been compiled before
# or not.

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
    return "|".join([arg_type_to_string(arg) for arg in args])

def print_kwargs(kwargs):
    if len(kwargs) == 0:
        return ""
    elif len(kwargs) == 1 and "parallelize" in kwargs:
        return f"{kwargs}"
    else:
        print(f"Unknown key-value arguments: {kwargs}")
        exit(1)

def generate_function_key(fn, args, kwargs):
    return f"{fn.__name__}+{print_type_signature(args)}+{print_kwargs(kwargs)}"
