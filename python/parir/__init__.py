from . import compile, key, parir
from .compile import clear_cache
from .operators import *

ir_asts = {}

def threads(n):
    from .parir import ParKind
    return ParKind.GpuThreads(n)

def reduce():
    from .parir import ParKind
    return ParKind.GpuReduction()

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
    def check_kwarg(key, default_value, expected_ty):
        if key not in kwargs or kwargs[key] is None:
            return default_value
        elif isinstance(kwargs[key], expected_ty):
            v = kwargs[key]
            del kwargs[key]
            return v
        else:
            raise RuntimeError(f"The keyword argument {key} should be of type {ty}")

    # Extract provided keyword arguments if available and ensure they have the
    # correct types.
    par = check_kwarg("parallelize", {}, dict)
    cache = check_kwarg("cache", True, bool)
    seq = check_kwarg("seq", False, bool)

    # Any keyword arguments remaining after processing the known ones above are
    # considered unknown.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unknown keyword arguments: {kwargs}")

    # If the user explicitly requests sequential execution by setting the 'seq'
    # keyword argument to True, we return the original Python function. This is
    # useful for debugging, as it avoids the need to remove the function
    # decorator.
    if seq:
        return fn

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    if not cache or not compile.is_cached(key):
        code = parir.compile_ir(ir_ast, args, par)
        compile.build_cuda_shared_library(key, code)

    # Return a CUDA wrapper which ensures the arguments are passed correctly on
    # to the exposed shared library function.
    return compile.get_cuda_wrapper(fn.__name__, key, cache)

def compile_string(fun_name, code, includes=[], libs=[]):
    k = "string_" + key.generate_code_key(code)
    compile.build_cuda_shared_library(k, code, includes, libs)
    return compile.get_cuda_wrapper(fun_name, k, False)

def print_compiled(fun, args, par=None):
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
        par = {}
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
