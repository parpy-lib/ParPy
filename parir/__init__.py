from . import key, parir

from pathlib import Path
import os
cache_path = Path(f"{os.path.expanduser('~')}/.cache/parir")
cache_path.mkdir(parents=True, exist_ok=True)

compiled_cache = {}

def compile_to_binary(cpp, cu_host, cu_dev, name):
    # Use Torch to produce binary code from the output code
    import torch.utils.cpp_extension
    if len(cu_dev) == 0:
        cpp_sources = [f"{cu_host}\n{cpp}"]
        cuda_sources = []
    else:
        cpp_sources = [cpp]
        cuda_sources = [f"{cu_dev}\n{cu_host}"]
    module = torch.utils.cpp_extension.load_inline(
        name = name,
        cpp_sources = cpp_sources,
        cuda_sources = cuda_sources
    )

    return getattr(module, name)

def compile_function(ir_ast, args, kwargs, fn, key):
    # Look for the parallelize keyword argument, which is used to control how
    # the for-loops of the function should be parallelized. If it is not
    # specified, we return the original function without performing any extra
    # work.
    if "parallelize" not in kwargs or kwargs["parallelize"] is None:
        return fn
    par = kwargs["parallelize"]
    del kwargs["parallelize"]
    if not isinstance(par, dict):
        raise RuntimeError("The parallelization argument must be a dictionary")
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unknown keyword arguments: {kwargs}")

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    cpp, cu_host, cu_dev = parir.compile_ir(ir_ast, args, par)

    return compile_to_binary(cpp, cu_host, cu_dev, fn.__name__)

def convert_python_function_to_ir(fn):
    import ast as python_ast
    import inspect
    filepath = inspect.getfile(fn)
    src, fst_line = inspect.getsourcelines(fn)
    ast = python_ast.parse("".join(src))
    return parir.python_to_ir(ast, filepath, fst_line-1)

def jit(fn):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    ir_ast = convert_python_function_to_ir(fn)

    def inner(*args, **kwargs):
        k = key.generate_function_key(fn, args, kwargs)
        if not k in compiled_cache:
            compiled_cache[k] = compile_function(ir_ast, args, kwargs, fn, key)
        compiled_cache[k](*args)
    inner.__name__ = fn.__name__
    return inner
