import os
from pathlib import Path

cache_path = Path(f"{os.path.expanduser('~')}/.cache/prickle")
cache_path.mkdir(parents=True, exist_ok=True)

def _get_library_path(key):
    return cache_path / f"{key}-lib.so"

def _is_cached(key):
    libpath = _get_library_path(key)
    return os.path.isfile(libpath)

def _flatten(xss):
    return [x for xs in xss for x in xs]

def _report_compile_error(r, source, backend_name, temp_file, opts):
    stdout = r.stdout.decode('ascii')
    stderr = r.stderr.decode('ascii')
    msg =\
        f"Compilation of generated {backend_name} code failed with exit code {r.returncode}:\n"\
        f"Standard out:\n{stdout}\nStandard error:\n{stderr}"
    if opts.write_output:
        with open(temp_file, "w+") as f:
            f.write(source)
        msg += "\nWrote generated code to file {temp_file}."
    raise RuntimeError(msg)

def _build_cuda_shared_library(key, source, opts):
    from .buffer import PARIR_NATIVE_PATH
    import subprocess
    import tempfile
    import torch
    libpath = _get_library_path(key)

    # Get the version of the current GPU and generate specialized code for it.
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        includes = opts.includes + [str(PARIR_NATIVE_PATH)]
        include_cmd = _flatten([["-I", include] for include in includes])
        lib_cmd = _flatten([["-L", lib] for lib in opts.libs])
        commands = [
            "-O3", "--shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
            "-x", "cu", tmp.name, "-o", libpath
        ]
        cmd = _flatten([["nvcc"], opts.extra_flags, include_cmd, lib_cmd, commands])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            import uuid
            temp_file = f"{uuid.uuid4().hex}.cu"
            _report_compile_error(r, source, "CUDA", temp_file, opts)

def _build_metal_shared_library(key, source, opts):
    from .buffer import PARIR_NATIVE_PATH, PARIR_METAL_BASE_LIB_PATH
    from .buffer import compile_metal_runtime_lib
    import subprocess
    import tempfile
    libpath = _get_library_path(key)
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        compile_metal_runtime_lib()
        metal_cpp_path = os.getenv("METAL_CPP_HEADER_PATH")
        includes = opts.includes + [metal_cpp_path, str(PARIR_NATIVE_PATH)]
        frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalKit"]
        include_cmd = _flatten([["-I", include] for include in includes])
        lib_cmd = _flatten([["-L", lib] for lib in opts.libs])
        commands = [
            "-O3", "-shared", "-fpic", "-std=c++17", str(PARIR_METAL_BASE_LIB_PATH),
            "-x", "c++", tmp.name, "-o", str(libpath)
        ]
        cmd = _flatten([["clang++"], opts.extra_flags, frameworks, include_cmd, lib_cmd, commands])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            import uuid
            temp_file = f"{uuid.uuid4().hex}.cpp"
            _report_compile_error(r, source, "Metal", temp_file, opts)

def _torch_to_ctype(dtype):
    import ctypes
    import torch
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

def clear_cache():
    """
    Clears the cache of compiled shared library files.
    """
    import shutil
    shutil.rmtree(f"{cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)

def build_shared_library(key, source, opts):
    """
    Builds a shared library from the given source code for the backend
    specified in the given options. The key is used to identify the source, and
    is assumed to be unique.
    """
    from .prickle import CompileBackend
    if not _is_cached(key):
        if opts.backend == CompileBackend.Cuda:
            _build_cuda_shared_library(key, source, opts)
        elif opts.backend == CompileBackend.Metal:
            _build_metal_shared_library(key, source, opts)
        else:
            raise RuntimeError(f"Cannot build for unsupported backend {opts.backend}")

def get_wrapper(name, key, opts):
    """
    Given a key identifying a compiled shared library, this function produces a
    wrapper with which users can call the specified low-level function while
    providing arguments via Python.
    """
    from .buffer import Buffer
    from .prickle import ScalarSizes
    import ctypes

    libpath = _get_library_path(key)
    lib = ctypes.cdll.LoadLibrary(libpath)

    # Determine the sizes to use for scalar values (integers and floats) based
    # on the provided options.
    sizes = ScalarSizes(opts)

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
            return sizes.int.to_ctype()
        elif isinstance(arg, float):
            return sizes.float.to_ctype()
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
