import ast as python_ast
import enum
import inspect
import torch

from . import key, ir, parir

compiled_cache = {}

def compile_function(ir_ast, args, kwargs, fn):
    # Look for the parallelize keyword argument, which is used to control how
    # the for-loops of the function should be parallelized.
    if "parallelize" in kwargs:
        par = kwargs["parallelize"]
        if par is None:
            par = {}
        del kwargs["parallelize"]
    else:
        par = {}
    if not isinstance(par, dict):
        raise RuntimeError("The parallelization argument must be a dictionary")
    if len(kwargs) > 0:
        raise RuntimeError(f"Received unknown keyword arguments: {kwargs}")

    # Compiles the IR AST using type information of the provided arguments and
    # the parallelization settings to determine how to generate parallel
    # low-level code.
    code = parir.compile_ir(ir_ast, args, par)
    print(code)

    # TODO: Produce binary code and return a function using this...
    return fn

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
