# Code for producing a key for a particular function call. This key is used to
# determine whether a particular version of a function has been compiled before
# or not.

from . import parir

import inspect
import hashlib
import os
import torch

def arg_to_string(arg):
    from .buffer import Buffer
    if isinstance(arg, Buffer):
        if len(arg.shape) == 0:
            return f"({arg.dtype}){arg.numpy()}"
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

def print_par_kwargs(par):
    return ",".join([f"{k}:{v}" for k, v in par.items()])

def generate_quick_function_key(ir_ast, args, opts):
    code = parir.print_ir_ast(ir_ast)
    return f"{code}+{print_type_signature(args)}+{print_par_kwargs(opts.parallelize)}+{opts.backend}"

def generate_function_key(quick_key):
    h = hashlib.new("sha256")
    h.update(quick_key.encode("ascii"))
    return h.hexdigest()

def generate_code_key(code):
    h = hashlib.new("sha256")
    h.update(code.encode('ascii'))
    return h.hexdigest()
