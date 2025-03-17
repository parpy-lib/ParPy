from . import compile, key, parir
from .compile import clear_cache
from .operators import *

ir_asts = {}
fun_cache = {}

def threads(n):
    from .parir import ParKind
    return ParKind.GpuThreads(n)

def reduce():
    from .parir import ParKind
    return ParKind.GpuReduction()

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

def compile_function(ir_ast, args, kwargs, fn):
    # Extract provided keyword arguments if available and ensure they have the
    # correct types.
    par = check_kwarg(kwargs, "parallelize", {}, dict)
    cache = check_kwarg(kwargs, "cache", True, bool)
    seq = check_kwarg(kwargs, "seq", False, bool)
    debug = check_kwarg(kwargs, "debug", False, bool)

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

    # If we recently generated a CUDA wrapper for this function, we do a quick
    # lookup based on the name and signature of the function to immediately
    # return the wrapper function.
    quick_key = key.generate_quick_function_key(ir_ast, args, par)
    if cache and quick_key in fun_cache:
        return fun_cache[quick_key]

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    full_key = key.generate_function_key(quick_key)
    if not cache or not compile.is_cached(full_key):
        code = parir.compile_ir(ir_ast, args, par, debug)
        compile.build_cuda_shared_library(full_key, code)

    # Return a CUDA wrapper which ensures the arguments are passed correctly on
    # to the exposed shared library function.
    wrap_fn = compile.get_cuda_wrapper(fn.__name__, full_key, cache)
    fun_cache[quick_key] = wrap_fn
    return wrap_fn

def compile_string(fun_name, code, includes=[], libs=[], extra_flags=[]):
    k = "string_" + key.generate_code_key(code)
    compile.build_cuda_shared_library(k, code, includes, libs, extra_flags)
    return compile.get_cuda_wrapper(fun_name, k, False)

def print_compiled(fun, args, par=None, debug=False):
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
    return parir.compile_ir(ir_ast, args, par, debug)

# Validate all arguments, ensuring that they have a supported type and that
# all tensor data is contiguous and allocated on the GPU.
def validate_arguments(args, kwargs):
    seq = check_kwarg(kwargs, "seq", False, bool)
    kwargs["seq"] = seq
    def check_arg(arg, i, in_dict):
        import torch
        if isinstance(arg, int) or isinstance(arg, float):
            return arg
        elif isinstance(arg, dict):
            if in_dict:
                raise RuntimeError(f"Argument {i} cannot be a nested dictionary")
            for k, v in arg.items():
                if isinstance(k, str):
                    v = check_arg(v, f"{i}[\"{k}\"]", True)
                else:
                    raise RuntimeError(f"Argument {i} has a non-string key")
            return arg
        elif isinstance(arg, torch.Tensor):
            # If we are to run the code sequentially (in the Python
            # interpreter), it doesn't matter where the tensors are
            # allocated or whether they are contiguous.
            if seq:
                return arg
            if arg.ndim > 0 and arg.get_device() != torch.cuda.current_device():
                msg = [
                    f"The data of tensor in argument {i} is on device ",
                    f"{arg.get_device()}, while it was expected to be on device ",
                    f"{torch.cuda.current_device()}"
                ]
                raise RuntimeError("".join(msg))
            elif not arg.is_contiguous():
                raise RuntimeError(f"Argument {i} contains non-contiguous data")
            return arg
        elif hasattr(arg, "__cuda_array_interface__"):
            # If the argument implements the CUDA array interface, such as a
            # CuPy array, we convert it to Torch and validate this, for
            # simplicity.
            return check_arg(torch.as_tensor(arg), i, in_dict)
        else:
            raise RuntimeError(f"Argument {i} has unsupported type {type(arg)}")
    return [check_arg(arg, f"#{i+1}", False) for (i, arg) in enumerate(args)]

def jit(fun):
    """
    Prepares the provided function for JIT-compilation. Initially, the Python
    function is parsed and translated to an untyped IR AST once. Then, for each
    time the function is used, the IR AST is JIT compiled based on the types of
    the provided arguments.
    """
    ir_ast = convert_python_function_to_ir(fun)

    def inner(*args, **kwargs):
        args = validate_arguments(args, kwargs)
        compile_function(ir_ast, args, kwargs, fun)(*args)
    ir_asts[inner] = ir_ast
    inner.__name__ = fun.__name__
    return inner
