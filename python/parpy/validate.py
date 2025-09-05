from .parpy import CompileBackend
from .buffer import Buffer
from .buffer import from_array
import numpy as np

def check_dict(arg, i, in_dict, opts, execute):
    if in_dict:
        raise RuntimeError(f"Dictionary argument {i} contains nested dictionary")
    arg_res = {}
    for k, v in arg.items():
        if not isinstance(k, str):
            raise RuntimeError(f"Dictionary argument {i} contains non-string key {k}")
        arg_res[k] = check_arg(v, f"{i}[\"{k}\"]", True, opts, execute)
    return arg_res

def check_arg(arg, i, in_dict, opts, execute):
    if isinstance(arg, int) or isinstance(arg, float):
        return arg
    elif isinstance(arg, dict):
        return check_dict(arg, i, in_dict, opts, execute)
    elif isinstance(arg, Buffer):
        return arg
    elif hasattr(arg, "__cuda_array_interface__"):
        if opts.backend == CompileBackend.Cuda:
            return from_array(arg, CompileBackend.Cuda)
        else:
            raise RuntimeError(f"Argument {i} is a CUDA array, which is not "
                                "supported in {opts.backend}.")
    elif hasattr(arg, "__array_interface__") or hasattr(arg, "__array__"):
        # Copy data to memory accessible from the GPU. If the resulting code
        # will not be executed, we do not copy data so we can generate code for
        # a backend even if it is not available.
        return from_array(arg, opts.backend if execute else None)
    else:
        raise RuntimeError(f"Argument {i} has unsupported type {type(arg)}")

# Validate all arguments, ensuring that they have a supported type and that
# all tensor data is contiguous and allocated on the GPU. At the same time, we
# convert data types. When converting to temporary buffers that require
# copying, we also include callback functions which are invoked after the
# kernel to ensure data is copied back.
def check_arguments(args, opts, execute):
    return [check_arg(arg, f"#{i+1}", False, opts, execute) for i, arg in enumerate(args)]
