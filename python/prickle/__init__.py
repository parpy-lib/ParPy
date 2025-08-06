from .prickle import par, seq, CompileBackend, CompileOptions
from . import backend, buffer, types
from .buffer import sync
from .operators import *

_ir_asts = {}
_ext_decls = {}
_ext_tops = {}
_fun_cache = {}

def _get_tops(backend):
    ast_tops = {k.__name__: v for k, v in _ir_asts.items()}
    if backend is not None:
        if backend in _ext_decls:
            ext_tops = _ext_decls[backend]
        else:
            ext_tops = {}
    else:
        ext_tops = _ext_tops
    return {**ast_tops, **ext_tops}

def _convert_python_function_to_ir(fn):
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
    top_map = _get_tops(None)
    return prickle.python_to_ir(ast, filepath, fst_line-1, col_ofs, top_map)

def _check_kwarg(kwargs, key, default_value, expected_ty):
    if key not in kwargs or kwargs[key] is None:
        return default_value
    elif isinstance(kwargs[key], expected_ty):
        v = kwargs[key]
        del kwargs[key]
        return v
    else:
        raise RuntimeError(f"The keyword argument {key} should be of type {ty}")

def _check_kwargs(kwargs):
    default_opts = prickle.CompileOptions()
    opts = _check_kwarg(kwargs, "opts", default_opts, type(default_opts))

    # If the compiler is given any other keyword arguments than those specified
    # above, it reports an error specifying which keyword arguments were not
    # supported.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unsupported keyword arguments: {kwargs}")

    return opts

def _compile_function(ir_ast, args, opts):
    from .compile import build_shared_library, get_wrapper
    from .key import generate_fast_cache_key, generate_function_key

    # Extract the name of the main function in the IR AST.
    name = prickle.get_function_name(ir_ast)

    # Generate a key based on the IR AST, the function arguments, and the
    # compile options. If this key is found in the cache, we have already
    # compiled the function in this way before, so we return the cached wrapper
    # function.
    fast_cache_key = generate_fast_cache_key(ir_ast, args, opts)
    if fast_cache_key in _fun_cache:
        return _fun_cache[fast_cache_key]

    # Generate the code based on the provided IR AST, arguments and compilation
    # options.
    top_map = _get_tops(opts.backend)
    code, unsymb_code = prickle.compile_ir(ir_ast, args, opts, top_map)

    # If the shared library corresponding to the generated code does not exist,
    # we run the underlying compiler to produce a shared library.
    cache_key = generate_function_key(unsymb_code, opts)
    build_shared_library(cache_key, code, opts)

    # Return a wrapper function which ensures the arguments are correctly
    # passed to the exposed shared library function.
    wrap_fn = get_wrapper(name, cache_key, opts)
    _fun_cache[fast_cache_key] = wrap_fn
    return wrap_fn

def _run_callbacks(callbacks, opts):
    if len(callbacks) > 0:
        sync(opts.backend)
        for cb in callbacks:
            cb()

def threads(n):
    """
    Produces a LoopPar object (used in parallel specifications) representing a
    parallel operation over `n` threads.
    """
    from .prickle import LoopPar
    return LoopPar().threads(n)

def reduce():
    """
    Produces a LoopPar object (used in parallel specifications) representing a
    parallelizable reduction.
    """
    from .prickle import LoopPar
    return LoopPar().reduce()

def clear_cache():
    """
    Clears the cached shared library files as well as the local function cache.
    """
    from .compile import clear_cache
    clear_cache()
    _fun_cache = {}

def declare_external(py_name, ext_name, params, res_ty, header, backend):
    """
    Declares external functions accessible from JIT-compiled functions.
    """
    from .prickle import make_external_declaration
    import inspect
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    if hasattr(caller, "positions"):
        p = caller.positions
        i = (caller.filename, p.lineno, p.col_offset, p.end_lineno, p.end_col_offset)
    else:
        i = None
    ext_decl = make_external_declaration(py_name, ext_name, params, res_ty, header, i)
    if not backend in _ext_decls:
        _ext_decls[backend] = {}
    _ext_decls[backend][py_name] = ext_decl
    _ext_tops[py_name] = ext_decl

def compile_string(fun_name, code, opts=prickle.CompileOptions()):
    """
    Compiles the code provided as a string and returns a wrapper to the
    entry point function with the specified name.
    """
    from .compile import build_shared_library, get_wrapper
    from .key import generate_function_key
    from .validate import check_arguments
    opts = backend.resolve_backend(opts, True)
    cache_key = "string_" + generate_function_key(code, opts)
    build_shared_library(cache_key, code, opts)
    fn = get_wrapper(fun_name, cache_key, opts)
    def inner(*args):
        callbacks, args = check_arguments(args, opts, True)
        fn(*args)
        _run_callbacks(callbacks, opts)
    inner.__name__ = fun_name
    return inner

def print_compiled(fun, args, opts=prickle.CompileOptions()):
    """
    Compile the provided Python function with respect to the given function
    arguments and parallelization arguments. Returns the resulting CUDA C++
    code.
    """
    from .validate import check_arguments
    opts = backend.resolve_backend(opts, False)
    if fun in _ir_asts:
        ir_ast = _ir_asts[fun]
    else:
        ir_ast = _convert_python_function_to_ir(fun)
    _, args = check_arguments(args, opts, False)
    top_map = _get_tops(opts.backend)
    code, _ = prickle.compile_ir(ir_ast, args, opts, top_map)
    return code

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    from .validate import check_arguments
    ir_ast = _convert_python_function_to_ir(fun)

    def inner(*args, **kwargs):
        opts = backend.resolve_backend(_check_kwargs(kwargs), True)
        callbacks, args = check_arguments(args, opts, True)
        # If the user explicitly requests sequential execution by setting the 'seq'
        # keyword argument to True, we call the original Python function.
        if opts.seq:
            fun(*args)
        else:
            _compile_function(ir_ast, args, opts)(*args)
        _run_callbacks(callbacks, opts)
    _ir_asts[inner] = ir_ast
    inner.__name__ = fun.__name__
    return inner
