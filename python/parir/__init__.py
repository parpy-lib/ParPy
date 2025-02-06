from . import compile, key, parir
from .compile import clear_cache
from .operators import *
from .parir import ParKind

ir_asts = {}

def convert_python_function_to_ir(fn):
    import ast as python_ast
    import inspect
    import textwrap
    filepath = inspect.getfile(fn)
    src, fst_line = inspect.getsourcelines(fn)

    # The Python AST parser requires properly indented code. Therefore, we make
    # sure to dedent it properly in case it is a nested function, including
    # removing any documentation strings that may prevent proper dedentation.
    src = textwrap.dedent("".join(src))
    if inspect.getdoc(fn) is not None:
        src = inspect.cleandoc(src)

    ast = python_ast.parse(src)
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
        del kwargs["parallelize"]
    else:
        raise RuntimeError("The parallelization argument must be a dictionary")

    # The cache argument is used to specify whether the compiled shared library
    # should be cached or not. By default, we cache the result.
    if "cache" not in kwargs or kwargs["cache"] is None:
        cache = True
    elif isinstance(kwargs["cache"], bool):
        cache = kwargs["cache"]
        del kwargs["cache"]
    else:
        raise RuntimeError("The cache argument must be a boolean")

    # Any keyword arguments remaining after processing the known ones above are
    # considered unknown.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unknown keyword arguments: {kwargs}")

    # Return the Python function as is if no parallelization was specified.
    if len(par) == 0:
        return fn

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    if not cache or not compile.is_cached(key):
        code = parir.compile_ir(ir_ast, args, par)
        compile.build_cuda_shared_library(key, code)

    # Return a CUDA wrapper which ensures the arguments are passed correctly on
    # to the exposed shared library function.
    return compile.get_cuda_wrapper(fn.__name__, key)

def compile_string(fun_name, code):
    k = "string_" + key.generate_code_key(code)
    if not compile.is_cached(k):
        compile.build_cuda_shared_library(k, code)
    return compile.get_cuda_wrapper(fun_name, k)

def print_compiled(fun, args, par):
    """
    Compile the provided Python function with respect to the given function
    arguments and parallelization arguments. Returns the resulting CUDA C++
    code.
    """
    if fun in ir_asts:
        ir_ast = ir_asts[fun]
    else:
        ir_ast = convert_python_function_to_ir(fun)
    if par is None:
        raise RuntimeError("Parallel specification must be provided")
    return parir.compile_ir(ir_ast, args, par)

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
    ir_asts[inner] = ir_ast
    inner.__name__ = fun.__name__
    return inner
