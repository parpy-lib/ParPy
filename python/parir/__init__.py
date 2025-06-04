from . import compile, key, parir, validate
from .compile import clear_cache
from .operators import *
from .parir import parallelize
from .parir import CompileBackend, CompileOptions

ir_asts = {}
fun_cache = {}

def threads(n):
    from .parir import LoopPar
    return LoopPar().threads(n)

def reduce():
    from .parir import LoopPar
    return LoopPar().reduce()

def convert_python_function_to_ir(fn):
    import ast as python_ast
    import builtins
    import inspect
    import itertools
    import textwrap
    filepath = inspect.getfile(fn)
    src, fst_line = inspect.getsourcelines(fn)

    # The Python AST parser requires properly indented code. Therefore, we make
    # sure to dedent it properly in case it is a nested function, including
    # removing any documentation strings that may prevent proper dedentation.
    col_ofs = builtins.sum(1 for _ in itertools.takewhile(str.isspace, src[0]))
    src = textwrap.dedent("".join(src))
    if inspect.getdoc(fn) is not None:
        src = inspect.cleandoc(src)

    # Parse the Python AST
    ast = python_ast.parse(src)

    # Convert the Python representation of the AST to a Python-like
    # representation in the compiler. As part of this step, we inline any
    # references to previously parsed functions.
    ir_ast_map = {k.__name__: v for k, v in ir_asts.items()}
    return parir.python_to_ir(ast, filepath, fst_line-1, col_ofs, ir_ast_map)

def check_kwarg(kwargs, key, default_value, expected_ty):
    if key not in kwargs or kwargs[key] is None:
        return default_value
    elif isinstance(kwargs[key], expected_ty):
        v = kwargs[key]
        del kwargs[key]
        return v
    else:
        raise RuntimeError(f"The keyword argument {key} should be of type {ty}")

def check_kwargs(kwargs):
    default_opts = parir.CompileOptions()
    opts = check_kwarg(kwargs, "opts", default_opts, type(default_opts))

    # If the compiler is given any other keyword arguments than those specified
    # above, it reports an error specifying which keyword arguments were not
    # supported.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unsupported keyword arguments: {kwargs}")

    return opts

def compile_function(ir_ast, args, opts, fn):
    # If the user explicitly requests sequential execution by setting the 'seq'
    # keyword argument to True, we return the original Python function. This is
    # useful for debugging, as it avoids the need to remove the function
    # decorator.
    if opts.seq:
        return fn

    # If we recently generated a wrapper for this function, we do a quick
    # lookup based on the name and signature of the function to immediately
    # return the wrapper function.
    quick_key = key.generate_quick_function_key(ir_ast, args, opts)
    if opts.cache and quick_key in fun_cache:
        return fun_cache[quick_key]

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    full_key = key.generate_function_key(quick_key)
    if not opts.cache or not compile.is_cached(full_key):
        code = parir.compile_ir(ir_ast, args, opts)
        compile.build_shared_library(full_key, code, opts)

    # Return a wrapper function which ensures the arguments are passed
    # correctly on to the exposed shared library function.
    wrap_fn = compile.get_wrapper(fn.__name__, full_key, opts)
    fun_cache[quick_key] = wrap_fn
    return wrap_fn

def compile_string(fun_name, code, opts=parir.CompileOptions()):
    k = "string_" + key.generate_code_key(code)
    compile.build_shared_library(k, code, opts)
    opts.cache = False
    return compile.get_wrapper(fun_name, k, opts)

def print_compiled(fun, args, opts=parir.CompileOptions()):
    """
    Compile the provided Python function with respect to the given function
    arguments and parallelization arguments. Returns the resulting CUDA C++
    code.
    """
    if fun in ir_asts:
        ir_ast = ir_asts[fun]
    else:
        ir_ast = convert_python_function_to_ir(fun)
    return parir.compile_ir(ir_ast, args, opts)

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    ir_ast = convert_python_function_to_ir(fun)

    def inner(*args, **kwargs):
        opts = check_kwargs(kwargs)
        args = validate.check_arguments(args, opts)
        compile_function(ir_ast, args, opts, fun)(*args)
    ir_asts[inner] = ir_ast
    inner.__name__ = fun.__name__
    return inner
