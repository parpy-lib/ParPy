# This file ensures that features that are not yet supported by the compiler
# are correctly reported as errors already when parsing the Python function.

import numpy as np
import parpy
import pytest
import torch

from common import *

def test_while_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def while_fun(x, y, N):
            i = 0
            while i < N:
                y[i] = x[i]
                i += 1
            else:
                y[i] = 0.0
    assert e_info.match(r".*lines 16-20.*")

def test_for_else_rejected():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def for_else(x, y, N):
            for i in range(N):
                y[i] = x[i]
            else:
                y[0] += 1
    assert e_info.match(r".*lines 27-30.*")

def test_with_unsupported_context():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def with_context():
            with 5:
                pass
    assert e_info.match(r".*lines 37-38.*")

def test_with_as():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def with_as():
            with parpy.gpu as x:
                a = x + 1
    assert e_info.match(r".*lines 45-46.*")

def test_dict_with_non_string_keys():
    @parpy.jit
    def dict_arg(a):
        with parpy.gpu:
            a["x"] = a["y"]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 'y': 4, 3: 5}, opts=parpy.par({}))
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")

def test_dict_with_int_key():
    @parpy.jit
    def dict_arg(a):
        with parpy.gpu:
            a["x"] = a[2]

    with pytest.raises(RuntimeError) as e_info:
        dict_arg({'x': 2, 2: 4}, opts=parpy.par({}))
    assert e_info.match(r"(.*non-string key.*)|(Found no enabled GPU backends.*)")

def test_invalid_max_thread_blocks_per_cluster():
    opts = parpy.CompileOptions()
    with pytest.raises(RuntimeError) as e_info:
        opts.max_thread_blocks_per_cluster = 3
    assert e_info.match(r".*number of thread blocks per cluster must be a power of two.*")

def test_empty_return():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def empty_return():
            return
    assert e_info.match(r"Empty return statements are not supported")

def test_return_in_main_function():
    enabled_backends = [b for b in compiler_backends if parpy.backend.is_enabled(b)]
    if len(enabled_backends) == 0:
        pytest.skip("No available backends to use for compilation")
    else:
        @parpy.jit
        def f_return(x):
            with parpy.gpu:
                y = parpy.reduce.sum(x[:])
                return y
        backend = enabled_backends[0]
        with pytest.raises(TypeError) as e_info:
            f_return(np.ndarray(10), opts=par_opts(backend, {}))
        assert e_info.match(r"Main function f_return cannot return a value")

def test_call_without_compiler_options():
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def fun_with_args(x, y):
            with parpy.gpu:
                x[:] += y[:]
        x = np.ndarray((10,), dtype=np.float32)
        y = np.ndarray((10,), dtype=np.float32)
        fun_with_args(x, y)
    assert e_info.match("Missing required keyword argument opts in call to fun_with_args")

@pytest.mark.parametrize('backend', compiler_backends)
def test_unbound_shape_variable(backend):
    N = parpy.types.shape_var()
    M = parpy.types.shape_var()
    with pytest.raises(NameError) as e_info:
        @parpy.jit
        def shape_var_in_loop_bound_zero_init(x: parpy.types.buffer(parpy.types.F32, [M])):
            for i in range(N):
                x[i] = 0.0
    assert e_info.match(r"Found reference to unused shape variable N")

@pytest.mark.parametrize('backend', compiler_backends)
def test_unbound_type_variable(backend):
    ty1 = parpy.types.type_var()
    ty2 = parpy.types.type_var()
    N = parpy.types.shape_var()

    @parpy.jit
    def unbound_type_var_conversion(x: parpy.types.buffer(ty1, [N])):
        with parpy.gpu:
            x[0] = parpy.builtin.convert(1.0, ty2)

    x = np.ndarray((1,), np.float32)
    opts = par_opts(backend, {})
    with pytest.raises(RuntimeError) as e_info:
        parpy.print_compiled(unbound_type_var_conversion, [x], opts)
    assert e_info.match("Found unresolved type variable")

@pytest.mark.parametrize('backend', compiler_backends)
def test_inlining_call_expr(backend):
    @parpy.jit
    def add_func(x, y):
        return x + y
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def invalid_call_inlining():
            with parpy.gpu:
                x = parpy.builtin.inline(add_func(2, 3))
    assert e_info.match("Expression cannot be inlined")
