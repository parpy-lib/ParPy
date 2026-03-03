import numpy as np
import parpy
import pytest

from common import *

@pytest.mark.parametrize('backend', compiler_backends)
def test_annotate_int_param(backend):
    def helper():
        @parpy.jit
        def annot_int_param(x: parpy.types.I32, y):
            with parpy.gpu:
                y[:] += x
        x = 1
        y = np.ndarray((10,), dtype=np.int32)
        annot_int_param(x, y, opts=par_opts(backend, {}))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_scalar_int_argument_expected_float(backend):
    def helper():
        @parpy.jit
        def annot_float_param(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = 1
        y = np.ndarray((10,), dtype=np.int32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_argument_expected_scalar(backend):
    def helper():
        @parpy.jit
        def annot_float_param_2(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = np.ndarray((10,), dtype=np.float32)
        y = np.ndarray((10,), dtype=np.float32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param_2(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_argument_wrong_element_type(backend):
    def helper():
        @parpy.jit
        def annot_float_param_3(x: parpy.types.F32, y):
            with parpy.gpu:
                y[:] += x
        x = np.ndarray((10,), dtype=np.int32)
        y = np.ndarray((10,), dtype=np.float32)
        with pytest.raises(TypeError) as e_info:
            annot_float_param_3(x, y, opts=par_opts(backend, {}))
        assert e_info.match("Parameter x was annotated with type parpy.types.F32 which is incompatible with argument type")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_parameter_annotation(backend):
    def helper():
        N = parpy.types.shape_var()
        @parpy.jit
        def annot_buffer_param(x: parpy.types.buffer(parpy.types.F32, (N,))):
            with parpy.gpu:
                x[:] += 1.0
        x = np.ndarray((10,), dtype=np.float32)
        annot_buffer_param(x, opts=par_opts(backend, {}))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_annotation_shape_equality(backend):
    def helper():
        N = parpy.types.shape_var()
        @parpy.jit
        def annot_add_elemwise_inplace(
            x: parpy.types.buffer(parpy.types.F32, [N]),
            y: parpy.types.buffer(parpy.types.F32, [N])
        ):
            parpy.label('N')
            x[:] += y[:]
        x = np.ndarray((10,), dtype=np.float32)
        y = np.ones_like(x)
        z = np.ndarray((20,), dtype=np.float32)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        annot_add_elemwise_inplace(x, y, opts=opts)
        with pytest.raises(TypeError) as e_info:
            annot_add_elemwise_inplace(x, z, opts=opts)
        assert e_info.match("Parameter y was annotated with type.*[.*].*which is incompatible with argument type.*[20].*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_called_function_annotation_shape_constraint(backend):
    N = parpy.types.shape_var()

    @parpy.jit
    def annot_add_elemwise_inplace_helper(
        x: parpy.types.buffer(parpy.types.F32, [N]),
        y: parpy.types.buffer(parpy.types.F32, [N])
    ) -> parpy.types.I32:
        parpy.label('N')
        x[:] += y[:]
        return 0

    @parpy.jit
    def annot_add_outer(a, b, N):
        parpy.label('N')
        for i in range(N):
            n = annot_add_elemwise_inplace_helper(a[i], b[i])
    x = np.ndarray((10,10), dtype=np.float32)
    y = np.ndarray((10,20), dtype=np.float32)
    opts = par_opts(backend, {'N': parpy.threads(128)})
    with pytest.raises(TypeError) as e_info:
        parpy.print_compiled(annot_add_outer, [x, y, 10], opts)
    assert e_info.match("Parameter y was annotated with type.*incompatible with.*")

@pytest.mark.parametrize('backend', compiler_backends)
def test_called_function_scalar_coercion(backend):
    def helper():
        @parpy.jit
        def base_func_add_scalars(x: parpy.types.I32, y: parpy.types.I32) -> parpy.types.I32:
            return x + y
        @parpy.jit
        def calling_func_add_scalars(x, y):
            parpy.label('N')
            x[:] = base_func_add_scalars(x[:], y[:])
        x = np.random.randint(0, 100, size=(10,), dtype=np.int64)
        y = np.random.randint(0, 100, size=(10,), dtype=np.int64)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        calling_func_add_scalars(x, y, opts=opts)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_shape_implicit_labeling(backend):
    def helper():
        import parpy.types as pt
        N = pt.shape_var()

        @parpy.jit
        def implicit_labels_add_elemwise(x: pt.buffer(pt.F32, [N]), y: pt.buffer(pt.F32, [N])):
            x[:N] += y[:N]

        x = np.ndarray((128,), dtype=np.float32)
        y = np.ndarray((128,), dtype=np.float32)
        opts = par_opts(backend, {'N': parpy.threads(128)})
        opts.implicit_shape_labels = True
        implicit_labels_add_elemwise(x, y, opts=opts)

        # If we disable implicit shape labels, the compilation should fail
        # because there is no parallelism (as no label was inserted).
        opts.implicit_shape_labels = False
        with pytest.raises(RuntimeError) as e_info:
            implicit_labels_add_elemwise(x, y, opts=opts)
        assert e_info.match(r"The function .* does not contain any parallelism")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_type_variable_scalar_inference(backend):
    sz = parpy.types.type_var()

    @parpy.jit
    def add_type_variable(x: sz, y: sz) -> sz:
        return x + y

    @parpy.jit
    def call_type_var_func(x):
        parpy.label('N')
        x[:] = add_type_variable(x[:], x[:])

    x = np.ndarray((10,), dtype=np.float32)
    opts = par_opts(backend, {'N': parpy.threads(32)})
    code = parpy.print_compiled(call_type_var_func, [x], opts)
    assert len(code) != 0

@pytest.mark.parametrize('backend', compiler_backends)
def test_type_variable_contradiction(backend):
    sz = parpy.types.type_var()

    @parpy.jit
    def add_type_variable2(x: sz, y: sz) -> sz:
        return x + y

    @parpy.jit
    def call_type_var_func2(x, y):
        parpy.label('N')
        x[:] = add_type_variable2(x[:], y[:])

    x = np.ndarray((10,), dtype=np.float32)
    y = np.ndarray((10,), dtype=np.int32)
    opts = par_opts(backend, {'N': parpy.threads(32)})
    with pytest.raises(TypeError) as e_info:
        parpy.print_compiled(call_type_var_func2, [x, y], opts)
    assert e_info.match("Parameter y was annotated with type .* which is incompatible with .*")

@pytest.mark.parametrize('backend', compiler_backends)
def test_type_variable_multiple_instantiations(backend):
    sz = parpy.types.type_var()

    @parpy.jit
    def add_type_variable3(x: sz, y: sz) -> sz:
        return x + y

    @parpy.jit
    def multi_instantiation_type_vars(x, y):
        with parpy.gpu:
            x[:] = add_type_variable3(x[:], x[:])
            y[:] = add_type_variable3(y[:], y[:])

    x = np.ndarray((10,), dtype=np.float32)
    y = np.ndarray((10,), dtype=np.int32)
    opts = par_opts(backend, {})
    code = parpy.print_compiled(multi_instantiation_type_vars, [x, y], opts)
    assert len(code) != 0

@pytest.mark.parametrize('backend', compiler_backends)
def test_type_variable_in_buffer(backend):
    N = parpy.types.shape_var()
    sz = parpy.types.type_var()

    @parpy.jit
    def add_elemwise_type_variables(
        x: parpy.types.buffer(sz, [N]),
        y: parpy.types.buffer(sz, [N])
    ):
        parpy.label('N')
        x[:] = x[:] + y[:]

    x = np.ndarray((10,), dtype=np.float32)
    y = np.ndarray((10,), dtype=np.float32)
    opts = par_opts(backend, {'N': parpy.threads(32)})
    code = parpy.print_compiled(add_elemwise_type_variables, [x, y], opts)
    assert len(code) != 0

@pytest.mark.parametrize('backend', compiler_backends)
def test_type_variable_in_buffer_contradiction(backend):
    N = parpy.types.shape_var()
    sz = parpy.types.type_var()

    @parpy.jit
    def add_elemwise_type_variables2(
        x: parpy.types.buffer(sz, [N]),
        y: parpy.types.buffer(sz, [N])
    ):
        parpy.label('N')
        x[:] = x[:] + y[:]

    x = np.ndarray((10,), dtype=np.float32)
    y = np.ndarray((10,), dtype=np.int32)
    opts = par_opts(backend, {'N': parpy.threads(32)})
    with pytest.raises(TypeError) as e_info:
        parpy.print_compiled(add_elemwise_type_variables2, [x, y], opts)
    assert e_info.match(r"Parameter y was annotated with type .* incompatible with.*")

T = parpy.types.type_var()
N = parpy.types.shape_var()

@parpy.jit
def elemwise_add_inplace(
        x: parpy.types.buffer(T, [N]),
        y: parpy.types.buffer(T, [parpy.types.literal(10)])):
    x[:N] = x[:N] + y[:N]

@pytest.mark.parametrize('backend', compiler_backends)
def test_literal_shape_variable(backend):
    def helper():
        T = parpy.types.type_var()
        N = parpy.types.shape_var()

        # OK if x and y contains 10 elements each
        p = {'N': parpy.threads(32)}
        x = torch.zeros(10, dtype=torch.float32)
        y = torch.randn_like(x)
        elemwise_add_inplace(x, y, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_literal_shape_variable_contradictory_arguments(backend):
    # If y contains 20 elements, we have a contradiction in the types
    with pytest.raises(TypeError) as e_info:
        x = torch.zeros(20, dtype=torch.float32)
        y = torch.randn_like(x)
        opts = par_opts(backend, {'N': parpy.threads(32)})
        parpy.print_compiled(elemwise_add_inplace, [x, y], opts)
    assert e_info.match(r"incompatible with argument type")

@pytest.mark.parametrize('backend', compiler_backends)
def test_literal_shape_variable_contradiction(backend):
    T = parpy.types.type_var()
    N = parpy.types.shape_var()

    # Contradictory use of shape literals - this will fail for any shapes of
    # the arguments A and B.
    @parpy.jit
    def matrix_elemwise_add(
            a: parpy.types.buffer(T, [N, parpy.types.literal(12)]),
            b: parpy.types.buffer(T, [parpy.types.literal(7), N]),
            c: parpy.types.buffer(T, [N, N])):
        c[:N,:N] = a[:,:] + b[:,:]

    with pytest.raises(TypeError) as e_info:
        a = torch.randn(12, 12, dtype=torch.float32)
        b = torch.randn_like(a)
        c = torch.zeros_like(a)
        opts = par_opts(backend, {'N': parpy.threads(32)})
        parpy.print_compiled(matrix_elemwise_add, [a, b, c], opts)
    assert e_info.match(r"incompatible with argument type")
