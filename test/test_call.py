import parpy
import parpy.types
import pytest
import torch

from common import *

@parpy.jit
def parpy_add_call(x: parpy.types.F32, y: parpy.types.F32):
    return x + y

@parpy.jit
def parpy_mul_call(x: parpy.types.F32, y: parpy.types.F32):
    return x * y

@parpy.jit
def parpy_add_mul(x: parpy.types.F32, y: parpy.types.F32, z: parpy.types.F32):
    return parpy_mul_call(parpy_add_call(x, y), z)

@parpy.jit
def parpy_add_direct(x, y, N):
    parpy.label('N')
    y[:] = parpy_add_call(x[:], y[:])

@parpy.jit
def parpy_add_mul_nested(x, y, N):
    parpy.label('N')
    y[:] = parpy_add_mul(x[:], y[:], x[:])

@parpy.jit
def parpy_sum_call(x, y):
    with parpy.gpu:
        y[0] = parpy.reduce.sum(x[:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_direct_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': parpy.threads(10)}
        parpy_add_direct(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': parpy.threads(10)}
        parpy_add_mul_nested(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x**2, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_sum_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros(1)
        p = {'N': parpy.threads(10)}
        parpy_sum_call(x, y, opts=par_opts(backend, p))
        assert torch.allclose(torch.sum(x), y)
    run_if_backend_is_enabled(backend, helper)

@parpy.jit
def add_inplace(x, y, M):
    parpy.label("1d")
    y[:] += x[:]

@parpy.jit
def add_2d_inplace(x, y, N, M):
    parpy.label("2d")
    for i in range(N):
        parpy.builtin.inline(add_inplace(x[i], y[i], M))

@parpy.jit
def add_2d_inplace_no_inline(x, y, N, M):
    parpy.label('2d')
    for i in range(N):
        add_inplace(x[i], y[i], M)

@parpy.jit
def add_2d_inplace_x2(x, y, z, w, N, M):
    parpy.label("2d")
    for i in range(N):
        parpy.builtin.inline(add_inplace(x[i], y[i], M))
        parpy.builtin.inline(add_inplace(z[i], w[i], M))

@parpy.jit
def add_3d_inplace(x, y, N, M, K):
    parpy.label('3d')
    for i in range(N):
        parpy.builtin.inline(add_2d_inplace(x[i], y[i], M, K))

@pytest.mark.parametrize('backend', compiler_backends)
def test_call(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        p = {'2d': parpy.threads(10), '1d': parpy.threads(15)}
        add_2d_inplace(x, y, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_stmt_no_inlining(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        p = {'2d': parpy.threads(10)}
        add_2d_inplace_no_inline(x, y, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_different_types(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        z = torch.randint(0, 10, (10, 15), dtype=torch.int32)
        w = torch.zeros_like(z)
        p = {'2d': parpy.threads(10), '1d': parpy.threads(15)}
        add_2d_inplace_x2(x, y, z, w, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
        assert torch.allclose(z, w)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_dependency(backend):
    def helper():
        x = torch.randn(10, 20, 30)
        y = torch.zeros_like(x)
        p = {
            '3d': parpy.threads(10),
            '2d': parpy.threads(20),
            '1d': parpy.threads(30)
        }
        add_3d_inplace(x, y, 10, 20, 30, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

def test_call_non_decorated_function_fails():
    # This function is intentionally not decorated with '@parpy.jit'
    def non_decorated_add(x, y, M):
        for i in range(M):
            y[i] += x[i]
    with pytest.raises(RuntimeError) as e_info:
        @parpy.jit
        def add_2d(x, y, N, M):
            parpy.label('N')
            for i in range(N):
                non_decorated_add(x[i], y[i], M)
    assert e_info.match(r"Call to unknown ParPy function .*non_decorated_add.*")

def test_recursive_call_fails():
    with pytest.raises(NameError) as e_info:
        @parpy.jit
        def reset(x, i):
            with parpy.gpu:
                if i > 0:
                    x[i] = 0.0
                    reset(x, i-1)
    assert e_info.match(r"name 'reset' is not defined")

def test_external_declaration():
    import parpy.types as types
    parpy.clear_cache()

    @parpy.external("powf", parpy.CompileBackend.Cuda, parpy.Target.Device)
    def pow(x: types.F32, y: types.F32) -> types.F32:
        return x ** y

def call_external_helper(backend, fn):
    x = torch.tensor(2.0, dtype=torch.float32)
    y = torch.zeros(1, dtype=torch.float32)
    fn(x, y, opts=par_opts(backend, {}))
    assert torch.allclose(y, torch.sqrt(x))

def call_external_helper_cuda():
    import parpy.types as types
    ext_name = "sqrtf"
    header = None
    backend = parpy.CompileBackend.Cuda

    @parpy.external(ext_name, backend, parpy.Target.Device, header=header)
    def sqrt_ext(x: types.F32) -> types.F32:
        return np.sqrt(x)
    @parpy.jit
    def ext_sqrt(x, y):
        with parpy.gpu:
            y[0] = sqrt_ext(x)
    call_external_helper(backend, ext_sqrt)

def call_external_helper_metal():
    import parpy.types as types
    ext_name = "metal::sqrt"
    header = "<metal_math>"
    backend = parpy.CompileBackend.Metal

    @parpy.external(ext_name, backend, parpy.Target.Device, header=header)
    def sqrt_ext(x: types.F32) -> types.F32:
        return np.sqrt(x)
    @parpy.jit
    def ext_sqrt(x, y):
        with parpy.gpu:
            y[0] = sqrt_ext(x)
    call_external_helper(backend, ext_sqrt)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_external(backend):
    def helper():
        if backend == parpy.CompileBackend.Cuda:
            call_external_helper_cuda()
        elif backend == parpy.CompileBackend.Metal:
            call_external_helper_metal()
    run_if_backend_is_enabled(backend, helper)

def select_distinct_element(x, l):
    for y in l:
        if x != y:
            return y
    raise RuntimeError(f"Could not find a distinct element from {x} in list {l}")

@pytest.mark.parametrize('backend', compiler_backends)
def test_invalid_backend_call(backend):
    import parpy.types as types
    def helper():
        other_backend = select_distinct_element(backend, compiler_backends)
        res_ty = types.I32

        @parpy.external("_zero", other_backend, parpy.Target.Device)
        def zero() -> types.I32:
            return 0
        with pytest.raises(TypeError) as e_info:
            @parpy.jit
            def f(x):
                with parpy.gpu:
                    x[:] = zero()
            x = torch.zeros(10, dtype=torch.int32)
            f(x, opts=par_opts(backend, {}))
        assert e_info.match(r"Call to unknown function .*zero.*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_invalid_parameter_type_annotation(backend):
    with pytest.raises(RuntimeError) as e_info:
        @parpy.external("dummy", backend, parpy.Target.Device)
        def dummy(x: int) -> parpy.types.I32:
            return x
    assert e_info.match("Unsupported parameter type annotation")

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_other_module_function(backend):
    def helper():
        import test_static_eq
        @parpy.jit
        def set_value_based_on_backend(x):
            parpy.builtin.inline(test_static_eq.parpy_backend_set_value(x))
        x = torch.zeros((1,), dtype=torch.int32)
        x[0] = 10
        set_value_based_on_backend(x, opts=par_opts(backend, {}))
        if backend == parpy.CompileBackend.Cuda:
            assert x[0] == 0
        elif backend == parpy.CompileBackend.Metal:
            assert x[0] == 1
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_host_external(backend):
    def helper():
        if backend == parpy.CompileBackend.Cuda:
            header = "<cuda_utils.h>"
        elif backend == parpy.CompileBackend.Metal:
            header = "<metal_utils.h>"
        else:
            print(f"Unsupported backend {backend}")

        @parpy.external("add_host", backend, parpy.Target.Host, header=header)
        def add_host(x: parpy.types.F32, y: parpy.types.F32) -> parpy.types.F32:
            return x + y
        @parpy.jit
        def call_func(x, y):
            z = add_host(x, 1.0)
            with parpy.gpu:
                for i in range(10):
                    y[i] = z
        y = torch.zeros((10,), dtype=torch.float32)
        opts = par_opts(backend, {})
        opts.includes += ['test/code']
        call_func(2.5, y, opts=opts)
        assert torch.allclose(y, torch.full((10,), 3.5))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_host_and_device_codegen(backend):
    @parpy.external("host_dummy", backend, parpy.Target.Host)
    def host_func(x: parpy.types.F32, y: parpy.types.F32) -> parpy.types.F32:
        return x + y
    @parpy.external("device_dummy", backend, parpy.Target.Device)
    def device_func(x: parpy.types.F32, y: parpy.types.F32) -> parpy.types.F32:
        return x + y

    # Invalid definition - we cannot mix use of host and device functions.
    @parpy.jit
    def func1(x, y, z):
        return host_func(x, device_func(y, z))

    @parpy.jit
    def func2(x, y, z):
        return x + host_func(y, z)

    @parpy.jit
    def func3(x, y, z):
        return device_func(x, y) + z

    a = 1.0
    b = 1.5
    c = 2.0
    d = 2.5
    N, M = 10, 15
    x = torch.zeros((N, M), dtype=torch.float32)
    opts = par_opts(backend, {'N': parpy.threads(N), 'M': parpy.threads(M)})

    N = parpy.types.shape_var()
    M = parpy.types.shape_var()

    # Invalid call, we use a function that involves both host and device code...
    @parpy.jit
    def entry1(x: parpy.types.buffer(parpy.types.F32, [N, M]), a, b, c, d):
        for i in range(N):
            for j in range(M):
                x[i, j] = func2(a, b, c)
    with pytest.raises(RuntimeError) as e_info:
        parpy.print_compiled(entry1, [x, a, b, c, d], opts=opts)
    assert e_info.match("Host function returning a non-void value")

    # Valid call, no errors expected here
    @parpy.jit
    def entry2(x: parpy.types.buffer(parpy.types.F32, [N, M]), a, b, c, d):
        for i in range(N):
            for j in range(M):
                x[i, j] = func3(x[i, j], c, d)
    code = parpy.print_compiled(entry2, [x, a, b, c, d], opts=opts)
    assert len(code) != 0
