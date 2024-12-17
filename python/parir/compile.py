import ctypes
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

def build_cuda_shared_library(key, source):
    libpath = get_library_path(key)
    if not torch.cuda.is_available():
        raise RuntimeError(f"Torch was not built with CUDA support")
    if not shutil.which("nvcc"):
        raise RuntimeError(f"Could not find 'nvcc' in path, which is required to compile the generated CUDA code")

    # Get the version of the current GPU and generate specialized code for it.
    # TODO: In the future, we should collect the versions of all GPUs on the
    # system and compile with all of these in mind.
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        r = subprocess.run(["nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC", f"-arch={arch}", "-x", "cu", tmp.name, "-o", libpath], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"Compilation of generated CUDA code failed with exit code {r.returncode}:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")

def get_cuda_wrapper(name, args, key):
    libpath = get_library_path(key)
    lib = ctypes.cdll.LoadLibrary(libpath)

    # TODO: Below code assumes int -> int64 and float -> float32, but it should
    # be possible to have other type mappings as well.
    def get_ctype(arg):
        if isinstance(arg, int):
            return ctypes.c_int64
        elif isinstance(arg, float):
            return ctypes.c_float
        else:
            return ctypes.c_void_p
    getattr(lib, name).argtypes = [get_ctype(arg) for arg in args]

    def value_or_ptr(arg):
        if isinstance(arg, torch.Tensor):
            return arg.data_ptr()
        else:
            return arg
    def wrapper(*args):
        ptr_args = [value_or_ptr(arg) for arg in args]
        getattr(lib, name)(*ptr_args)
    wrapper.__name__ = name
    return wrapper
