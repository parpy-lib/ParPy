import ast as python_ast
import enum
import inspect
import torch

from . import key, ir, parir
from .parir import ParKind, ParSpec

compiled_cache = {}

def compile_function(fn, args, kwargs):
    # Look for the parallelize keyword argument, which is used to control how
    # the for-loops of the function should be parallelized.
    par = kwargs["parallelize"] if "parallelize" in kwargs and kwargs["parallelize"] is not None else []
    if not isinstance(par, list):
        print(par)
        raise RuntimeError("The parallelization argument provided via the 'parallelize' keyword argument must be a list")

    # Convert the provided function to an AST.
    src = inspect.getsource(fn)
    ast = python_ast.parse(src)

    # Immediately compiles the provided Python AST to low-level parallelized
    # code, based on the provided function and parallelization arguments. We
    # pass along the Python AST module for quick access from Rust.
    code = parir.compile_python_ast(ast, args, par, python_ast)

    # TODO: Produce binary code and return a function using this...
    return fn

def jit(fn):
    def inner(*args, **kwargs):
        k = key.generate_function_key(fn, args, kwargs)
        if not k in compiled_cache:
            compiled_cache[k] = compile_function(fn, args, kwargs)
        compiled_cache[k](*args)
    inner.__name__ = fn.__name__
    return inner
