from .parir import CompileBackend
from .buffer import Buffer

def check_dict(arg, i, in_dict, check_rec):
    if in_dict:
        raise RuntimeError(f"Dictionary argument {i} contains a nested dictionary")
    if not all([isinstance(k, str) for k in arg.keys()]):
        raise RuntimeError(f"Dictionary argument {i} contains non-string key")
    results = [check_rec(v, f"{i}[\"{k}\"]", True) for k, v in arg.items()]
    callbacks = [x for xs in results for x in xs[0]]
    arg = {k: r[1] for k, r in zip(arg.keys(), results)}
    return callbacks, arg

def check_dummy_arg(arg, i, in_dict):
    array_attrs = ["__array__", "__array_interface__", "__cuda_array_interface__"]
    if isinstance(arg, int) or isinstance(arg, float):
        return [], arg
    elif isinstance(arg, dict):
        return check_dict(arg, i, in_dict, check_dummy_arg)
    elif isinstance(arg, Buffer):
        return [], arg.numpy()
    elif any([hasattr(arg, x) for x in array_attrs]):
        return [], Buffer.from_array(arg, None)
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_cuda_arg(arg, i, in_dict, seq):
    if isinstance(arg, int) or isinstance(arg, float):
        return [], arg
    elif isinstance(arg, dict):
        def helper(arg, i, in_dict):
            return check_cuda_arg(arg, i, in_dict, seq)
        return check_dict(arg, i, in_dict, helper)
    elif isinstance(arg, Buffer):
        if seq:
            return [], arg.numpy()
        return [], arg
    elif hasattr(arg, "__cuda_array_interface__"):
        if seq:
            return [], arg
        return [], Buffer.from_array(arg, CompileBackend.Cuda)
    elif hasattr(arg, "__array_interface__") or hasattr(arg, "__array__"):
        if seq:
            return [], arg
        buf = Buffer.from_array(arg, CompileBackend.Cuda)
        callback = lambda: buf.cleanup()
        return [callback], buf
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_metal_arg(arg, i, in_dict, seq):
    if isinstance(arg, int) or isinstance(arg, float):
        return [], arg
    elif isinstance(arg, dict):
        def helper(arg, i, in_dict):
            return check_metal_arg(arg, i, in_dict, seq)
        return check_dict(arg, i, in_dict, helper)
    elif isinstance(arg, Buffer):
        if seq:
            return [], arg.numpy()
        return [], arg
    elif hasattr(arg, "__array_interface__") or hasattr(arg, "__array__"):
        if seq:
            return [], arg
        buf = Buffer.from_array(arg, CompileBackend.Metal)
        callback = lambda: buf.cleanup()
        return [callback], buf
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
        return check_metal_arg(arg, i, False, opts.seq)
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
    callbacks = [x for xs in callbacks for x in xs]

    # Restore the original backend for the code generation.
    if not execute:
        opts.backend = old_backend

    return callbacks, args

