import ctypes
import itertools
import os
import tempfile
import torch
import shutil
import subprocess
from pathlib import Path
cache_path = Path(f"{os.path.expanduser('~')}/.cache/parir")
cache_path.mkdir(parents=True, exist_ok=True)

def clear_cache():
    shutil.rmtree(f"{cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

def get_library_path(key):
    return cache_path / f"{key}-lib.so"

def is_cached(key):
    libpath = get_library_path(key)
    return os.path.isfile(libpath)

def build_cuda_shared_library(key, source, includes=[], libs=[]):
    libpath = get_library_path(key)
    if not torch.cuda.is_available():
        raise RuntimeError(f"Torch was not built with CUDA support")
    if not shutil.which("nvcc"):
        raise RuntimeError(f"Could not find 'nvcc' in path, which is required to compile the generated CUDA code")

    # Get the version of the current GPU and generate specialized code for it.
    # TODO: In the future, we should collect the versions of all GPUs on the
    # system and compile with all of these in mind.
    def flatten(xss):
        return [x for xs in xss for x in xs]
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        include_cmd = [["-I", include] for include in includes]
        lib_cmd = [["-L", lib] for lib in libs]
        commands = [
            "-O3", "--shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
            "-x", "cu", f"{tmp.name}", "-o", f"{libpath}"
        ]
        cmd = flatten([["nvcc"], include_cmd, lib_cmd, commands])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            import uuid
            temp_file = f"{uuid.uuid4().hex}.cu"
            with open(temp_file, "w+") as f:
                f.write(source)
            stdout = r.stdout.decode('ascii')
            stderr = r.stderr.decode('ascii')
            raise RuntimeError(f"Compilation of generated CUDA code failed with exit code {r.returncode}:\nstdout:\n{stdout}\nstderr:\n{stderr}\nWrote generated code to file {temp_file}.")

def torch_to_ctype(dtype):
    if dtype == torch.int8:
        return ctypes.c_int8
    elif dtype == torch.int16:
        return ctypes.c_int16
    elif dtype == torch.int32:
        return ctypes.c_int32
    elif dtype == torch.int64:
        return ctypes.c_int64
    elif dtype == torch.float16:
        return ctypes.c_int16
    elif dtype == torch.float32:
        return ctypes.c_float
    elif dtype == torch.float64:
        return ctypes.c_double
    else:
        raise RuntimeError(f"Unknown torch dtype: {dtype}")

def get_cuda_wrapper(name, key, cache):
    libpath = get_library_path(key)
    lib = ctypes.cdll.LoadLibrary(libpath)

    # Remove the shared library if caching is not enabled
    if not cache:
        try:
            os.remove(libpath)
        except:
            pass

    def wrapper(*args):
        # Expand the arguments by making each field of a dictionary a separate argument
        def expand_arg(arg):
            if isinstance(arg, dict):
                return [v for (_, v) in sorted(arg.items())]
            else:
                return [arg]
        args = list(itertools.chain.from_iterable([expand_arg(a) for a in args]))

        # Extract the C type of each argument
        def get_ctype(arg):
            # We treat int and floats from Python as 64-bit values
            if isinstance(arg, int):
                return ctypes.c_int64
            elif isinstance(arg, float):
                return ctypes.c_double
            elif isinstance(arg, torch.Tensor):
                if arg.ndim == 0:
                    return torch_to_ctype(arg.dtype)
                else:
                    return ctypes.c_void_p
            else:
                return ctypes.c_void_p
        getattr(lib, name).argtypes = [get_ctype(arg) for arg in args]

        # Extract the pointers or values of tensor arguments before passing to CUDA
        def value_or_ptr(arg):
            if isinstance(arg, torch.Tensor):
                if arg.ndim == 0:
                    return arg.item()
                else:
                    return arg.contiguous().data_ptr()
            else:
                return arg
        def extract_args(args):
            args = list(itertools.chain.from_iterable([expand_arg(a) for a in args]))
            return [value_or_ptr(arg) for arg in args]
        ptr_args = extract_args(args)
        getattr(lib, name)(*ptr_args)
    wrapper.__name__ = name
    return wrapper
