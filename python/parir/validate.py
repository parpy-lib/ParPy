from .parir import CompileBackend
from .buffer import Buffer

def check_dummy_arg(arg, i, in_dict):
    array_attrs = ["__array__", "__array_interface__", "__cuda_array_interface__"]
    if isinstance(arg, int) or isinstance(arg, float):
        return None, arg
    elif isinstance(arg, dict):
        if in_dict:
            raise RuntimeError(f"Dictionary argument {i} contains a nested dictionary")
        for k, v in arg.items():
            if isinstance(k, str):
                v = check_dummy_arg(v, f"{i}[\"{k}\"]", in_dict)
            else:
                raise RuntimeError(f"Dictionary argument {i} contains non-string key")
        return None, arg
    elif isinstance(arg, Buffer):
        return None, arg.numpy()
    elif any([hasattr(arg, x) for x in array_attrs]):
        return None, Buffer.from_array(arg, None)
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_cuda_arg(arg, i, in_dict, seq):
    if isinstance(arg, int) or isinstance(arg, float):
        return None, arg
    elif isinstance(arg, dict):
        if in_dict:
            raise RuntimeError(f"Dictionary argument {i} contains a nested dictionary")
        for k, v in arg.items():
            if isinstance(k, str):
                v = check_cuda_arg(v, f"{i}[\"{k}\"]", True, seq)
            else:
                raise RuntimeError(f"Dictionary argument {i} contains non-string key")
        return None, arg
    elif isinstance(arg, Buffer):
        if seq:
            return None, arg.numpy()
        return None, arg
    elif hasattr(arg, "__cuda_array_interface__"):
        if seq:
            return None, arg
        return None, Buffer.from_array(arg, CompileBackend.Cuda)
    elif hasattr(arg, "__array_interface__") or hasattr(arg, "__array__"):
        if seq:
            return None, arg
        buf = Buffer.from_array(arg, CompileBackend.Cuda)
        callback = lambda: buf.cleanup()
        return callback, buf
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_metal_arg(arg, i, seq):
    if isinstance(arg, int) or isinstance(arg, float):
        return None, arg
    elif isinstance(arg, Buffer):
        if seq:
            return None, arg.numpy()
        return None, arg
    elif hasattr(arg, "__array_interface__") or hasattr(arg, "__array__"):
        if seq:
            return None, arg
        buf = Buffer.from_array(arg, CompileBackend.Metal)
        callback = lambda: buf.cleanup()
        return callback, buf
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_arg(arg, i, opts):
    # When generating code for printing without execution, the Dummy backend is
    # used to avoid assuming any particular platforms are available.
    if opts.backend == CompileBackend.Dummy:
        return check_dummy_arg(arg, i, False)
    if opts.backend == CompileBackend.Cuda:
        return check_cuda_arg(arg, i, False, opts.seq)
    elif opts.backend == CompileBackend.Metal:
        return check_metal_arg(arg, i, opts.seq)
    else:
        raise RuntimeError(f"Unsupported compilation backend {opts.backend}")

# Validate all arguments, ensuring that they have a supported type and that
# all tensor data is contiguous and allocated on the GPU. At the same time, we
# convert data types. When converting to temporary buffers that require
# copying, we also include callback functions which are invoked after the
# kernel to ensure data is copied back.
def check_arguments(args, opts, execute):
    # If we are only generating code without executing anything, we avoid using
    # the dependencies of that backend by using a generic buffer type.
    if not execute:
        old_backend = opts.backend
        opts.backend = CompileBackend.Dummy

    callbacks, args = zip(*[check_arg(arg, f"#{i+1}", opts) for (i, arg) in enumerate(args)])
    callbacks = [cb for cb in callbacks if cb is not None]

    # Restore the original backend for the code generation.
    if not execute:
        opts.backend = old_backend

    return callbacks, args

