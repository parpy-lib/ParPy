from . import backend, buffer, compile, key, parir, validate
from .buffer import sync
from .compile import clear_cache
from .operators import *
from .state import *
from .parir import par, seq
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

def compile_function(ir_ast, args, opts):
    # Extract the name of the main function in the IR AST.
    name = parir.get_ir_function_name(ir_ast)

    # Generate a key based on the IR AST, the function arguments, and the
    # compile options. If this key is found in the cache, we have already
    # compiled the function in this way before, so we return the cached wrapper
    # function.
    fast_cache_key = key.generate_fast_cache_key(ir_ast, args, opts)
    if fast_cache_key in fun_cache:
        return fun_cache[fast_cache_key]

    # Generate the code based on the provided IR AST, arguments and compilation
    # options.
    ir_ast_map = {k.__name__: v for k, v in ir_asts.items()}
    code, unsymb_code = parir.compile_ir(ir_ast, args, opts, ir_ast_map)

    # If the shared library corresponding to the generated code does not exist,
    # we run the underlying compiler to produce a shared library.
    cache_key = key.generate_function_key(unsymb_code, opts)
    if not compile.is_cached(cache_key):
        compile.build_shared_library(cache_key, code, opts)

    # Return a wrapper function which ensures the arguments are correctly
    # passed to the exposed shared library function.
    wrap_fn = compile.get_wrapper(name, cache_key, opts)
    fun_cache[fast_cache_key] = wrap_fn
    return wrap_fn

def run_callbacks(callbacks, opts):
    if len(callbacks) > 0:
        sync(opts.backend)
        for cb in callbacks:
            cb()

def compile_string(fun_name, code, opts=parir.CompileOptions()):
    opts = backend.resolve(opts, True)
    cache_key = "string_" + key.generate_function_key(code, opts)
    compile.build_shared_library(cache_key, code, opts)
    fn = compile.get_wrapper(fun_name, cache_key, opts)
    def inner(*args):
        callbacks, args = validate.check_arguments(args, opts, True)
        fn(*args)
        run_callbacks(callbacks, opts)
    inner.__name__ = fun_name
    return inner

def print_compiled(fun, args, opts=parir.CompileOptions()):
    """
    Compile the provided Python function with respect to the given function
    arguments and parallelization arguments. Returns the resulting CUDA C++
    code.
    """
    opts = backend.resolve(opts, False)
    if fun in ir_asts:
        ir_ast = ir_asts[fun]
    else:
        ir_ast = convert_python_function_to_ir(fun)
    _, args = validate.check_arguments(args, opts, False)
    ir_ast_map = {k.__name__: v for k, v in ir_asts.items()}
    code, _ = parir.compile_ir(ir_ast, args, opts, ir_ast_map)
    return code

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    ir_ast = convert_python_function_to_ir(fun)

    def inner(*args, **kwargs):
        opts = backend.resolve(check_kwargs(kwargs), True)
        callbacks, args = validate.check_arguments(args, opts, True)
        # If the user explicitly requests sequential execution by setting the 'seq'
        # keyword argument to True, we call the original Python function.
        if opts.seq:
            fun(*args)
        else:
            compile_function(ir_ast, args, opts)(*args)
        run_callbacks(callbacks, opts)
    ir_asts[inner] = ir_ast
    inner.__name__ = fun.__name__
    return inner
