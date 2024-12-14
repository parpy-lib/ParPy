import ast as python_ast
import inspect
import torch.utils.cpp_extension

from . import key, parir

compiled_cache = {}

def compile_function(ir_ast, args, kwargs, fn):
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

    # Use Torch to produce binary code from the output code
    if len(cu_dev) == 0:
        cpp_sources = [f"{cu_host}\n{cpp}"]
        cuda_sources = []
    else:
        cpp_sources = [cpp]
        cuda_sources = [f"{cu_dev}\n{cu_host}"]
    module = torch.utils.cpp_extension.load_inline(
        name = fn.__name__,
        cpp_sources = cpp_sources,
        cuda_sources = cuda_sources
    )

    return getattr(module, fn.__name__)

def jit(fn):
    # When we first reach a function, we compile it to the IR AST with little
    # type information. Then, when the function is actually called, we use the
    # provided function arguments and the parallelization arguments to produce
    # a complete IR AST before compiling this to low-level code.
    src = inspect.getsource(fn)
    ast = python_ast.parse(src)
    ir_ast = parir.python_to_ir(ast)

    def inner(*args, **kwargs):
        k = key.generate_function_key(fn, args, kwargs)
        if not k in compiled_cache:
            compiled_cache[k] = compile_function(ir_ast, args, kwargs, fn)
        compiled_cache[k](*args)
    inner.__name__ = fn.__name__
    return inner
