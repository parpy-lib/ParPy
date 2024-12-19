from . import compile, key, parir
from .compile import clear_cache
from .parir import ParKind

def convert_python_function_to_ir(fn):
    import ast as python_ast
    import inspect
    filepath = inspect.getfile(fn)
    src, fst_line = inspect.getsourcelines(fn)
    ast = python_ast.parse("".join(src))
    return parir.python_to_ir(ast, filepath, fst_line-1)

def compile_function(ir_ast, args, kwargs, fn, key):
    # Look for the parallelize keyword argument, which is used to control how
    # the for-loops of the function should be parallelized. If it is not
    # specified, we return the original function without performing any extra
    # work.
    if "parallelize" not in kwargs or kwargs["parallelize"] is None:
        return fn
    elif isinstance(kwargs["parallelize"], dict):
        par = kwargs["parallelize"]
    else:
        raise RuntimeError("The parallelization argument must be a dictionary")
    del kwargs["parallelize"]

    # The cache argument is used to specify whether the compiled shared library
    # should be cached or not. By default, we cache the result.
    if "cache" not in kwargs or kwargs["cache"] is None:
        cache = True
    elif isinstance(kwargs["cache"], bool):
        cache = kwargs["cache"]
    else:
        raise RuntimeError("The cache argument must be a boolean")
    del kwargs["cache"]

    # Any keyword arguments remaining after processing the known ones above are
    # considered unknown.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unknown keyword arguments: {kwargs}")

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    if not cache or not compile.is_cached(key):
        # TODO: build differently if we have no device code
        cu_host, cu_dev = parir.compile_ir(ir_ast, args, par)
        cu_source = f"{cu_dev}\n{cu_host}"
        compile.build_cuda_shared_library(key, cu_source)

    return compile.get_cuda_wrapper(fn.__name__, args, key)

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    ir_ast = convert_python_function_to_ir(fun)

    def inner(*args, **kwargs):
        k = key.generate_function_key(fun, args, kwargs)
        compile_function(ir_ast, args, kwargs, fun, k)(*args)
    inner.__name__ = fun.__name__
    return inner
