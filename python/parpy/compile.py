import os
from pathlib import Path

cache_path = Path(f"{os.path.expanduser('~')}/.cache/parpy")
cache_path.mkdir(parents=True, exist_ok=True)

def _get_native_path(key, opts):
    from .parpy import CompileBackend
    if opts.backend == CompileBackend.Triton:
        return cache_path / f"{key}.py"
    else:
        return cache_path / f"{key}-lib.so"

def _is_cached(key, opts):
    # NOTE(larshum, 2025-12-19): Library files generated from strings should
    # never be considered cached. This resolves an odd bug in the Metal
    # backend, where loading a cached library built from a string causes a
    # segmentation fault, for unknown reason.
    if key.startswith("string_"):
        return False
    path = _get_native_path(key, opts)
    return os.path.isfile(path)

def _flatten(xss):
    return [x for xs in xss for x in xs]

def _report_compile_error(r, source, backend_name, temp_file, opts):
    stdout = r.stdout.decode('utf-8')
    stderr = r.stderr.decode('utf-8')
    msg =\
        f"Compilation of generated {backend_name} code failed with exit code {r.returncode}:\n"\
        f"Standard out:\n{stdout}\nStandard error:\n{stderr}"
    if opts.write_output:
        with open(temp_file, "w+") as f:
            f.write(source)
        msg += "\nWrote generated code to file {temp_file}."
    raise RuntimeError(msg)

def _build_cuda_shared_library(key, source, opts):
    from .runtime import PARPY_NATIVE_PATH
    import subprocess
    import tempfile
    import torch
    libpath = _get_native_path(key, opts)

    # Get the version of the current GPU and generate specialized code for it.
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        includes = opts.includes + [str(PARPY_NATIVE_PATH)]
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
    from .parpy import CompileBackend
    from .runtime import _compile_runtime_lib
    from .runtime import PARPY_NATIVE_PATH, PARPY_METAL_BASE_LIB_PATH
    import subprocess
    import tempfile
    libpath = _get_native_path(key, opts)
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "w") as f:
            f.write(source)
        _compile_runtime_lib(CompileBackend.Metal)
        metal_cpp_path = os.getenv("METAL_CPP_HEADER_PATH")
        includes = opts.includes + [metal_cpp_path, str(PARPY_NATIVE_PATH)]
        frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalKit"]
        include_cmd = _flatten([["-I", include] for include in includes])
        lib_cmd = _flatten([["-L", lib] for lib in opts.libs])
        commands = [
            "-O3", "-shared", "-fpic", "-std=c++17", str(PARPY_METAL_BASE_LIB_PATH),
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

def _build_triton_python_module(key, source, opts):
    from .parpy import CompileBackend
    from .runtime import PARPY_NATIVE_PATH
    module_path = _get_native_path(key, opts)
    # Loads a pre-defined Triton file containing simple definitions we may use
    # in the generated Triton code.
    with open(PARPY_NATIVE_PATH / "parpy_triton.py", "r") as f:
        prelude_code = f.read()
    # Prepend the prelude code to the generated code and store this in the
    # cache path.
    with open(module_path, "w+") as f:
        f.write(f"{prelude_code}\n{source}")

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
    from .parpy import CompileBackend
    if not _is_cached(key, opts):
        if opts.backend == CompileBackend.Cuda:
            _build_cuda_shared_library(key, source, opts)
        elif opts.backend == CompileBackend.Metal:
            _build_metal_shared_library(key, source, opts)
        elif opts.backend == CompileBackend.Triton:
            _build_triton_python_module(key, source, opts)
        else:
            raise RuntimeError(f"Cannot build for unsupported backend {opts.backend}")

# Extract the Ctype type of an argument based on its type. This is only
# used when calling a function compiled from a string, where we do not know
# the exact argument types beforehand (in this case, 'argtypes' is set to
# None).
def _get_ctype(sizes, arg):
    import ctypes
    from .buffer import Buffer
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

# Expand arguments such that each value stored in a dictionary is passed as a
# separate argument.
def _expand_arg(arg):
    if isinstance(arg, dict):
        return [v for (_, v) in sorted(arg.items())]
    else:
        return [arg]

def _expand_args(args):
    if any([isinstance(arg, dict) for arg in args]):
        exp_args = [_expand_arg(a) for a in args]
        return [x for xs in exp_args for x in xs]
    return args

# Extract the pointers or values of buffer arguments.
def _value_or_ptr(arg):
    from .buffer import Buffer
    if isinstance(arg, Buffer):
        if len(arg.shape) == 0:
            return arg.numpy()
        else:
            return arg._get_ptr()
    else:
        return arg

def _to_callback_argument(callback_wrapper_code, ty, vars):
    import types
    module_code = compile(callback_wrapper_code, "<callback>", "exec")
    code = [c for c in module_code.co_consts if isinstance(c, types.CodeType)][0]
    globs, _ = vars
    return ty(types.FunctionType(code, globs))

def _check_status(lib, status):
    import ctypes
    if status != 0:
        lib.parpy_get_error_message.restype = ctypes.c_char_p
        msg = lib.parpy_get_error_message()
        raise RuntimeError(f"{msg.decode('ascii')}")

def set_cuda_stream(args, opts):
    from .parpy import CompileBackend
    if opts.backend == CompileBackend.Cuda:
        import ctypes
        import torch
        # TODO(larshum, 2025-11-18): We may want the ability to customize this
        # to make it more flexible.
        stream = torch.cuda.current_stream()
        return args + [stream.cuda_stream]
    return args

def _load_python_module(key, opts):
    import importlib.util
    import sys
    module_path = _get_native_path(key, opts)
    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

def _make_python_callback(callback_str, vars):
    globs, _ = vars
    d = globs
    code = compile(callback_str, "<callback>", "exec")
    exec(code, d)
    fn = d.popitem()[1]
    def cb_wrapper(*args):
        fn(*[_value_or_ptr(arg) for arg in args])
    return cb_wrapper

def get_triton_wrapper(name, key, argtypes, vars, callbacks, opts):
    module = _load_python_module(key, opts)
    if len(callbacks) == 0:
        def wrapper(*args):
            args = _expand_args(args)
            getattr(module, name)(*args)
    else:
        callback_funs = [_make_python_callback(cb, vars) for cb in callbacks]
        def wrapper(*args):
            args = list(_expand_args(args)) + callback_funs
            getattr(module, name)(*args)
    return wrapper

def get_string_wrapper(name, key, opts):
    import ctypes
    from .parpy import CompileBackend, ScalarSizes
    if opts.backend == CompileBackend.Triton:
        return get_triton_wrapper(name, key, [], {}, [], opts)
    libpath = _get_native_path(key, opts)
    lib = ctypes.cdll.LoadLibrary(libpath)
    getattr(lib, name).restype = ctypes.c_int32
    sizes = ScalarSizes(opts)
    def wrapper(*args):
        args = _expand_args(args)
        if opts.backend == CompileBackend.Cuda:
            getattr(lib, name).argtypes = \
                [_get_ctype(sizes, arg) for arg in args] + [ctypes.c_void_p]
        else:
            getattr(lib, name).argtypes = [_get_ctype(sizes, arg) for arg in args]
        args = [_value_or_ptr(arg) for arg in args]
        args = set_cuda_stream(args, opts)
        _check_status(lib, getattr(lib, name)(*args))
    return wrapper

def get_wrapper(name, key, argtypes, vars, callbacks, opts):
    from .parpy import CompileBackend
    import ctypes
    if opts.backend == CompileBackend.Triton:
        return get_triton_wrapper(name, key, argtypes, vars, callbacks, opts)
    libpath = _get_native_path(key, opts)
    lib = ctypes.cdll.LoadLibrary(libpath)
    getattr(lib, name).restype = ctypes.c_int32
    if opts.backend == CompileBackend.Cuda:
        getattr(lib, name).argtypes = argtypes + [ctypes.c_void_p]
    else:
        getattr(lib, name).argtypes = argtypes
    def wrapper(*args):
        args = _expand_args(args)
        callback_argtypes = argtypes[len(argtypes)-len(callbacks):]
        callback_args = [
            _to_callback_argument(cb, ty, vars) for cb, ty in
            zip(callbacks, callback_argtypes)
        ]
        args = [_value_or_ptr(arg) for arg in args] + callback_args
        args = set_cuda_stream(args, opts)
        _check_status(lib, getattr(lib, name)(*args))
    return wrapper
