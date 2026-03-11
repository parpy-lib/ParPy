import ctypes
import inspect
import pathlib
import re
import subprocess
import torch
import triton
import triton.compiler as tc
from triton.backends.compiler import GPUTarget

def _get_sm_version():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

def _get_cuda_version():
    res = subprocess.run(["nvcc", "--version"], capture_output=True)
    if res.returncode == 0:
        out = res.stdout.decode("utf-8")
        r = re.search(r"release (\d+)\.(\d+)", out, re.MULTILINE)
        if r:
            return f"{r.group(1)}.{r.group(2)}"
        else:
            raise RuntimeError(f"Failed to parse CUDA version of nvcc")
    else:
        raise RuntimeError(f"Failed to run nvcc to find its CUDA version")

def _get_ptx_version():
    cuda_vers = _get_cuda_version()
    return triton.backends.nvidia.compiler.ptx_get_version(cuda_vers)

def _count_arguments_in_ptx(ptx_code):
    return len(re.findall(r"^\s+\.param", ptx_code, re.MULTILINE))

def _compile_function(kernel, signature, attrs):
    cfg = kernel.configs[0] if len(kernel.configs) == 1 else kernel.best_config
    signature = dict(zip(kernel.arg_names, signature))
    # Add all constant expressions, accessible via keyword arugments of the
    # selected configiuration, to the signature.
    for k in cfg.kwargs.keys():
        signature[k] = "constexpr"
    src = tc.ASTSource(
        fn=kernel.fn,
        signature=signature,
        constexprs=cfg.kwargs,
        attrs=attrs
    )
    options = {
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "num_ctas": cfg.num_ctas,
        "maxnreg": cfg.maxnreg,
        "ptx_version": _get_ptx_version(),
    }
    compiled = triton.compile(
        src,
        target=GPUTarget("cuda", _get_sm_version(), 32),
        options=options
    )
    cache_path = pathlib.Path(inspect.getfile(kernel.fn.fn)).parent
    with open(f"{cache_path}/{kernel.fn.__name__}.cubin", "wb+") as f:
        f.write(compiled.asm["cubin"])

    smem = compiled.metadata.shared
    argc = _count_arguments_in_ptx(compiled.asm["ptx"])
    return (smem, argc)

def _load_python_module(key, opts):
    import importlib.util
    import sys
    from .compile import _get_cache_dir
    module_path = _get_cache_dir(key) / "main.py"
    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

def _make_python_callback(callback_str, vars):
    from .compile import _value_or_ptr
    globs, _ = vars
    d = globs
    code = compile(callback_str, "<callback>", "exec")
    exec(code, d)
    fn = d.popitem()[1]
    def cb_wrapper(*args):
        fn(*[_value_or_ptr(arg) for arg in args])
    return cb_wrapper

def _insert_triton_args(args, argtypes, smem, argc, cache_dir):
    args = list(args)

    # Add the current PyTorch stream
    args += [torch.cuda.current_stream().cuda_stream]
    argtypes += [ctypes.c_void_p]

    # Add the shared memory amounts
    args += [smem.data_ptr()]
    argtypes += [ctypes.c_void_p]

    # Add the argument counts of the generated code
    args += [argc.data_ptr()]
    argtypes += [ctypes.c_void_p]

    # Add the path to the cache directory
    args += [cache_dir.encode('utf-8')]
    argtypes += [ctypes.c_char_p]

    return args, argtypes

def get_wrapper(name, key, argtypes, vars, callbacks, opts):
    from .compile import _check_status, _expand_args, _get_cache_dir, _to_callback_argument, _value_or_ptr
    module = _load_python_module(key, opts)
    if opts.triton_native:
        libpath = _get_cache_dir(key) / "main-lib.so"
        lib = ctypes.cdll.LoadLibrary(libpath)
        getattr(lib, name).restype = ctypes.c_int32
        cache_dir = str(_get_cache_dir(key))
        smem = torch.tensor([], dtype=torch.int32)
        argc = torch.tensor([], dtype=torch.int32)
        first_run = True
        def wrapper(*args):
            nonlocal argtypes, smem, argc, first_run
            args = _expand_args(args)
            if first_run:
                if len(callbacks) > 0:
                    callback_funs = [_make_python_callback(cb, vars) for cb in callbacks]
                    args = list(_expand_args(args)) + callback_funs
                res = getattr(module, name)(*args)
                if len(res[0]) > 0:
                    smem, argc = zip(*[_compile_function(k, s, {}) for k, s in zip(*res)])
                    smem = torch.tensor(smem, dtype=torch.int32)
                    argc = torch.tensor(argc, dtype=torch.int32)
                _, argtypes = _insert_triton_args(args, argtypes, smem, argc, cache_dir)
                getattr(lib, name).argtypes = argtypes
                first_run = False
            else:
                args = [_value_or_ptr(arg) for arg in args]
                if len(callbacks) > 0:
                    callback_argtypes = argtypes[len(argtypes)-len(callbacks):-4]
                    callback_args = [
                        _to_callback_argument(cb, ty, vars) for cb, ty in
                        zip(callbacks, callback_argtypes)
                    ]
                    args = args + callback_args
                args, _ = _insert_triton_args(args, argtypes, smem, argc, cache_dir)
                _check_status(lib, getattr(lib, name)(*args))
    else:
        callback_funs = [_make_python_callback(cb, vars) for cb in callbacks]
        def wrapper(*args):
            args = _expand_args(args)
            if len(callbacks) > 0:
                args = list(args) + callback_funs
            getattr(module, name)(*args)
    return wrapper
