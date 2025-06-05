import ctypes
import itertools
import os
import tempfile
import torch
import shutil
import subprocess
from pathlib import Path
from .parir import CompileBackend

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

def flatten(xss):
    return [x for xs in xss for x in xs]

def build_cuda_shared_library(key, source, opts):
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
        include_cmd = flatten([["-I", include] for include in opts.includes])
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
    from .buffer import try_load_metal_base_lib, PARIR_METAL_PATH, PARIR_METAL_BASE_LIB_PATH
    from .state import get_metal_cpp_header_path
    libpath = get_library_path(key)
    if not torch.mps.is_available():
        raise RuntimeError(f"Torch was not built with Metal support")
    if not shutil.which("clang++"):
        raise RuntimeError(f"Could not find 'clang++' in path, which is required to compile the generated Metal code")

    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        try_load_metal_base_lib()
        metal_cpp_path = get_metal_cpp_header_path()
        if metal_cpp_path is None:
            raise RuntimeError(f"The path to the Metal C++ library must be provided \
                                 via the 'parir.set_metal_cpp_header_path' function \
                                 before using the Metal backend.")
        includes = opts.includes + [metal_cpp_path, str(PARIR_METAL_PATH)]
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

def get_cuda_wrapper(name, lib):
    def wrapper(*args):
        # Expand the arguments by making each field of a dictionary a separate argument
        def expand_arg(arg):
            if isinstance(arg, dict):
                return [v for (_, v) in sorted(arg.items())]
            else:
                return [arg]
        if any([isinstance(arg, dict) for arg in args]):
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
                    return arg.data_ptr()
            else:
                return arg
        ptr_args = [value_or_ptr(arg) for arg in args]
        getattr(lib, name)(*ptr_args)
    wrapper.__name__ = name
    return wrapper

def get_metal_wrapper(name, lib):
    def wrapper(*args):
        from .buffer import Buffer
        def get_ctype(arg):
            if isinstance(arg, int):
                return ctypes.c_int64
            elif isinstance(arg, float):
                return ctypes.c_double
            elif isinstance(arg, Buffer):
                return ctypes.c_void_p
            else:
                raise RuntimeError(f"Unsupported argument type: {arg}")
        getattr(lib, name).argtypes = [get_ctype(arg) for arg in args]
        def value_or_ptr(arg):
            if isinstance(arg, Buffer):
                return arg.buf
            else:
                return arg
        ptr_args = [value_or_ptr(arg) for arg in args]
        getattr(lib, name)(*ptr_args)
    wrapper.__name__ = name
    return wrapper

def get_wrapper(name, key, opts):
    libpath = get_library_path(key)
    lib = ctypes.cdll.LoadLibrary(libpath)
    # Remove the shared library if caching is not enabled
    if not opts.cache:
        try:
            os.remove(libpath)
        except:
            pass

    if opts.backend == CompileBackend.Cuda:
        return get_cuda_wrapper(name, lib)
    elif opts.backend == CompileBackend.Metal:
        return get_metal_wrapper(name, lib)
