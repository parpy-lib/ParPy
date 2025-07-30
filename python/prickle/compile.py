import ctypes
import itertools
import os
import tempfile
import torch
import shutil
import subprocess
from pathlib import Path
from .buffer import PARIR_NATIVE_PATH
from .prickle import CompileBackend

cache_path = Path(f"{os.path.expanduser('~')}/.cache/prickle")
cache_path.mkdir(parents=True, exist_ok=True)

def clear_cache():
    shutil.rmtree(f"{cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

def get_library_path(key):
    return cache_path / f"{key}-lib.so"

def is_cached(key):
    libpath = get_library_path(key)
    return os.path.isfile(libpath)

def flatten(xss):
    return [x for xs in xss for x in xs]

def build_cuda_shared_library(key, source, opts):
    libpath = get_library_path(key)

    # Get the version of the current GPU and generate specialized code for it.
    # TODO: In the future, we should collect the versions of all GPUs on the
    # system and compile with all of these in mind.
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        includes = opts.includes + [str(PARIR_NATIVE_PATH)]
        include_cmd = flatten([["-I", include] for include in includes])
        lib_cmd = flatten([["-L", lib] for lib in opts.libs])
        commands = [
            "-O3", "--shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
            "-x", "cu", tmp.name, "-o", libpath
        ]
        cmd = flatten([["nvcc"], opts.extra_flags, include_cmd, lib_cmd, commands])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            import uuid
            temp_file = f"{uuid.uuid4().hex}.cu"
            with open(temp_file, "w+") as f:
                f.write(source)
            stdout = r.stdout.decode('ascii')
            stderr = r.stderr.decode('ascii')
            raise RuntimeError(f"Compilation of generated CUDA code failed with exit code {r.returncode}:\nstdout:\n{stdout}\nstderr:\n{stderr}\nWrote generated code to file {temp_file}.")

def build_metal_shared_library(key, source, opts):
    from .buffer import try_load_metal_base_lib, PARIR_METAL_BASE_LIB_PATH
    libpath = get_library_path(key)
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        try_load_metal_base_lib()
        metal_cpp_path = os.getenv("METAL_CPP_HEADER_PATH")
        includes = opts.includes + [metal_cpp_path, str(PARIR_NATIVE_PATH)]
        frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalKit"]
        include_cmd = flatten([["-I", include] for include in includes])
        lib_cmd = flatten([["-L", lib] for lib in opts.libs])
        commands = [
            "-O3", "-shared", "-fpic", "-std=c++17", str(PARIR_METAL_BASE_LIB_PATH),
            "-x", "c++", tmp.name, "-o", str(libpath)
        ]
        cmd = flatten([["clang++"], opts.extra_flags, frameworks, include_cmd, lib_cmd, commands])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            import uuid
            temp_file = f"{uuid.uuid4().hex}.cpp"
            with open(temp_file, "w+") as f:
                f.write(source)
            stdout = r.stdout.decode('ascii')
            stderr = r.stderr.decode('ascii')
            raise RuntimeError(f"Compilation of generated Metal code failed with exit code {r.returncode}:\nstdout:\n{stdout}\nstderr:\n{stderr}\nWrote generated code to file {temp_file}.")

def build_shared_library(key, source, opts):
    if opts.backend == CompileBackend.Cuda:
        build_cuda_shared_library(key, source, opts)
    elif opts.backend == CompileBackend.Metal:
        build_metal_shared_library(key, source, opts)
    else:
        raise RuntimeError(f"Cannot build for unsupported backend {opts.backend}")

def torch_to_ctype(dtype):
    mapping = {
        torch.int8: ctypes.c_int8,
        torch.int16: ctypes.c_int16,
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.float16: ctypes.c_int16,
        torch.float32: ctypes.c_float,
        torch.float64: ctypes.c_double
    }
    if dtype in mapping:
        return mapping[dtype]
    else:
        raise RuntimeError(f"Unsupported Torch dtype: {dtype}")

def get_wrapper(name, key, opts):
    from .buffer import Buffer

    libpath = get_library_path(key)
    lib = ctypes.cdll.LoadLibrary(libpath)

    # Expand arguments such that each value stored in a dictionary is passed as a
    # separate argument.
    def expand_arg(arg):
        if isinstance(arg, dict):
            return [v for (_, v) in sorted(arg.items())]
        else:
            return [arg]

    # Return the ctypes type of an argument.
    def get_ctype(arg):
        if isinstance(arg, int):
            return ctypes.c_int64
        elif isinstance(arg, float):
            return ctypes.c_double
        elif isinstance(arg, Buffer):
            if len(arg.shape) == 0:
                return arg.dtype.to_ctype()
            else:
                return ctypes.c_void_p
        else:
            raise RuntimeError(f"Argument {arg} has unsupported type {type(arg)}")

    # Extract the pointers or values of buffer arguments.
    def value_or_ptr(arg):
        if isinstance(arg, Buffer):
            if len(arg.shape) == 0:
                return arg.numpy()
            else:
                return arg.buf
        else:
            return arg

    def wrapper(*args):
        if any([isinstance(arg, dict) for arg in args]):
            exp_args = [expand_arg(a) for a in args]
            args = [x for xs in exp_args for x in xs]
        getattr(lib, name).argtypes = [get_ctype(arg) for arg in args]
        getattr(lib, name).restype = ctypes.c_int32
        status = getattr(lib, name)(*[value_or_ptr(arg) for arg in args])
        if status != 0:
            lib.prickle_get_error_message.restype = ctypes.c_char_p
            msg = lib.prickle_get_error_message()
            raise RuntimeError(f"{msg.decode('ascii')}")
    wrapper.__name__ = name
    return wrapper
