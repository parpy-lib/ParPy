from .parir import CompileBackend
import torch

def check_cuda_arg(arg, i, in_dict, seq):
    if isinstance(arg, int) or isinstance(arg, float):
        return arg
    elif isinstance(arg, dict):
        if in_dict:
            raise RuntimeError(f"Dictionary argument {i} contains a nested dictionary")
        for k, v in arg.items():
            if isinstance(k, str):
                v = check_cuda_arg(v, f"{i}[\"{k}\"]", True, seq)
            else:
                raise RuntimeError(f"Dictionary argument {i} contains non-string key")
        return arg
    elif isinstance(arg, torch.Tensor):
        # If we are to run the code sequentially (in the Python
        # interpreter), it doesn't matter where the tensors are
        # allocated or whether they are contiguous.
        if seq:
            return arg
        if arg.ndim > 0 and arg.get_device() != torch.cuda.current_device():
            msg = [
                f"Tensor argument {i} is on device {arg.get_device()}, while ",
                f"it was expected to be on device {torch.cuda.current_device()}."
            ]
            raise RuntimeError("".join(msg))
        elif not arg.is_contiguous():
            raise RuntimeError(f"Tensor argument {i} contains non-contiguous data")
        return arg
    elif hasattr(arg, "__cuda_array_interface__"):
        # If the argument implements the CUDA array interface, such as a
        # CuPy array, we convert it to Torch and validate this, for
        # simplicity.
        return check_cuda_arg(torch.as_tensor(arg), i, in_dict, seq)
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_metal_arg(arg, i):
    if isinstance(arg, int) or isinstance(arg, float):
        return arg
    elif isinstance(arg, torch.Tensor):
        # TODO: Convert the argument to a Metal buffer
        raise RuntimeError(f"Torch tensors are not yet supported")
    elif hasattr(arg, "__array_interface__"):
        # TODO: Convert the argument to a Metal buffer
        raise RuntimeError(f"Arguments implementing the '__array_interface__' are not yet supported")
    else:
        raise RuntimeError(f"Argument {i} is of unsupported type {type(arg)}")

def check_arg(arg, i, opts):
    if opts.backend == CompileBackend.Cuda:
        return check_cuda_arg(arg, i, False, opts.seq)
    elif opts.backend == CompileBackend.Metal:
        return check_metal_arg(arg, i)
    else:
        raise RuntimeError(f"Unsupported compilation backend {opts.backend}")

# Validate all arguments, ensuring that they have a supported type and that
# all tensor data is contiguous and allocated on the GPU.
def check_arguments(args, opts):
    return [check_arg(arg, f"#{i+1}", opts) for (i, arg) in enumerate(args)]

