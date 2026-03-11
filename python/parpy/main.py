from . import backend, parpy

from .parpy import par, CompileBackend, CompileOptions, ElemSize, Target

_ir_asts = {}
_ext_decls = {}
_ext_tops = {}
_callbacks = {}
_fun_cache = {}

def _qualified_name(fn):
    import inspect
    return f"{inspect.getmodule(fn).__name__}.{fn.__name__}"

def _get_tops(backend):
    ast_tops = {_qualified_name(k): v for k, v in _ir_asts.items()}
    if backend is not None:
        if backend in _ext_decls:
            ext_tops = _ext_decls[backend]
        else:
            ext_tops = {}
    else:
        ext_tops = _ext_tops
    return {**ast_tops, **ext_tops, **_callbacks}

def _get_source_code(fn):
    import inspect
    import itertools
    import textwrap
    src, fst_line = inspect.getsourcelines(fn)

    # The Python AST parser requires properly indented code. Therefore, we make
    # sure to dedent it properly in case it is a nested function, including
    # removing any documentation strings that may prevent proper dedentation.
    col_ofs = sum(1 for _ in itertools.takewhile(str.isspace, src[0]))
    src = textwrap.dedent("".join(src))
    if inspect.getdoc(fn) is not None:
        src = inspect.cleandoc(src)
    return src, fst_line-1, col_ofs

def _parse_function(fn):
    import ast as python_ast
    import inspect
    import itertools
    filepath = inspect.getfile(fn)
    src, line_ofs, col_ofs = _get_source_code(fn)
    ast = python_ast.parse(src)
    info = (filepath, line_ofs, col_ofs)
    return ast, info

def _convert_python_function_to_ir(fn, vars):
    ast, info = _parse_function(fn)

    # Convert the Python representation of the AST to a Python-like
    # representation in the compiler. As part of this step, we inline any
    # references to previously parsed functions.
    top_map = _get_tops(None)
    return parpy.python_to_ir(ast, info, top_map, vars)

def _check_kwarg(kwargs, key, expected_ty, fun_name):
    if key not in kwargs or kwargs[key] is None:
        raise RuntimeError(f"Missing required keyword argument {key} in call to {fun_name}")
    elif isinstance(kwargs[key], expected_ty):
        v = kwargs[key]
        del kwargs[key]
        return v
    else:
        raise RuntimeError(f"The keyword argument {key} should be of type {ty}")

def _validate_external_type(target, par):
    from .parpy import CompileBackend, Target
    if target == Target.Host and par.is_parallel():
        raise RuntimeError(f"Host externals cannot be parallel")

def _declare_callback(fn, vars):
    ast, info = _parse_function(fn)
    return parpy.declare_callback(ast, info, vars)

def _declare_external(fn, ext_name, target, header, parallelize, vars):
    ast, info = _parse_function(fn)
    return parpy.declare_external(ast, info, ext_name, target, header, parallelize, vars)

def _check_kwargs(kwargs, fun_name):
    default_opts = CompileOptions()
    opts = _check_kwarg(kwargs, "opts", type(default_opts), fun_name)

    # If the compiler is given any other keyword arguments than those specified
    # above, it reports an error specifying which keyword arguments were not
    # supported.
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unsupported keyword arguments: {kwargs}")

    return opts

def _compile_function(qualified_name, ir_ast, args, vars, opts):
    from .compile import build_shared_library, get_wrapper
    from .key import _generate_fast_cache_key, _generate_function_key

    # Quickly generate a key based on the qualified name of the function, the
    # arguments passed to it, and the compilation options. If the key is found
    # in the cache, it was already compiled using the same specialization.
    fast_cache_key = _generate_fast_cache_key(qualified_name, args, opts)
    if fast_cache_key in _fun_cache:
        return _fun_cache[fast_cache_key]

    # Generate the code based on the provided IR AST, arguments and compilation
    # options.
    top_map = _get_tops(opts.backend)
    argtypes, callbacks, code, unsymb_code = parpy.compile_ir(ir_ast, args, opts, top_map)

    # If the shared library corresponding to the generated code does not exist,
    # we run the underlying compiler to produce a shared library.
    cache_key = _generate_function_key(unsymb_code, opts)
    build_shared_library(cache_key, code, opts)

    # Extract the name of the main function in the IR AST.
    name = parpy.get_function_name(ir_ast)

    # Return a wrapper function which ensures the arguments are correctly
    # passed to the exposed shared library function.
    wrap_fn = get_wrapper(name, cache_key, argtypes, vars, callbacks, opts)
    _fun_cache[fast_cache_key] = wrap_fn
    return wrap_fn

def threads(n):
    """
    Produces a LoopPar object (used in parallel specifications) representing a
    parallel operation over `n` threads.
    """
    return parpy.LoopPar().threads(n)

def par_reduction():
    """
    Produces a LoopPar object (used in parallel specifications) representing a
    parallelizable reduction.
    """
    return parpy.LoopPar().par_reduction()

def unroll():
    """
    Produces a LoopPar object indicating that the associated loop(s) should be
    unrolled.
    """
    return parpy.LoopPar().unroll()

def clear_cache():
    """
    Clears the cached shared library files as well as the local function cache.
    """
    from .compile import clear_cache
    global _fun_cache
    clear_cache()
    _fun_cache = {}

def default_backend():
    """
    Returns the default backend used by the compiler. If no backends are
    available, or more than one is available, this function returns 'None'.
    """
    from .backend import _resolve_backend
    opts = CompileOptions()
    opts.verbose_backend_resolution = False
    try:
        opts = _resolve_backend(opts, False)
        return opts.backend
    except:
        return None

def compile_string(fun_name, code, opts=CompileOptions()):
    """
    Compiles the code provided as a string and returns a wrapper to the
    entry point function with the specified name.
    """
    from .compile import build_shared_library, get_string_wrapper
    from .key import _generate_function_key
    from .validate import check_arguments
    import functools
    opts = backend._resolve_backend(opts, True)
    cache_key = "string_" + _generate_function_key(code, opts)
    build_shared_library(cache_key, code, opts)
    fn = get_string_wrapper(fun_name, cache_key, opts)
    @functools.wraps(fn)
    def inner(*args):
        args = check_arguments(args, opts, True)
        fn(*args)
    return inner

def print_compiled(fun, args, opts=CompileOptions()):
    """
    Compile the provided Python function with respect to the given function
    arguments and parallelization arguments. Returns a string representing the
    code generated by the backend.
    """
    from .validate import check_arguments
    import inspect
    opts = backend._resolve_backend(opts, False)
    if fun in _ir_asts:
        ir_ast = _ir_asts[fun]
    else:
        globs = inspect.currentframe().f_back.f_globals
        locs = inspect.currentframe().f_back.f_locals
        vars = (globs, locs)
        ir_ast = _convert_python_function_to_ir(fun, vars)
    args = check_arguments(args, opts, False)
    top_map = _get_tops(opts.backend)
    _, _, code, _ = parpy.compile_ir(ir_ast, args, opts, top_map)
    return code[0]

def callback(fn):
    import inspect
    globs = inspect.currentframe().f_back.f_globals
    locs = inspect.currentframe().f_back.f_locals
    vars = (globs, locs)
    _callbacks[_qualified_name(fn)] = _declare_callback(fn, vars)
    return fn

def external(ext_name, backend, target, header=None, parallelize=parpy.LoopPar()):
    """
    Decorator used to indicate that the associated function refers to an
    externally defined function.
    """
    import inspect
    globs = inspect.currentframe().f_back.f_globals
    locs = inspect.currentframe().f_back.f_locals
    vars = (globs, locs)
    def external_wrap(fn):
        import functools
        _validate_external_type(target, parallelize)
        ext_decl = _declare_external(fn, ext_name, target, header, parallelize, vars)
        if not backend in _ext_decls:
            _ext_decls[backend] = {}
        _ext_decls[backend][_qualified_name(fn)] = ext_decl
        _ext_tops[_qualified_name(fn)] = ext_decl
        return fn
    return external_wrap

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    from .validate import check_arguments
    import functools
    import inspect
    globs = inspect.currentframe().f_back.f_globals
    locs = inspect.currentframe().f_back.f_locals
    vars = (globs, locs)
    qualified_name = _qualified_name(fun)
    ir_ast = _convert_python_function_to_ir(fun, vars)

    @functools.wraps(fun)
    def inner(*args, **kwargs):
        opts = backend._resolve_backend(_check_kwargs(kwargs, fun.__name__), True)
        args = check_arguments(args, opts, True)
        fn = _compile_function(qualified_name, ir_ast, args, vars, opts)
        fn(*args)
    _ir_asts[inner] = ir_ast
    return inner
