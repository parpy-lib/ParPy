import prickle
import prickle.types
import pytest
import torch

from common import *

@prickle.jit
def prickle_add(x: prickle.types.F32, y: prickle.types.F32):
    return x + y

@prickle.jit
def prickle_mul(x: prickle.types.F32, y: prickle.types.F32):
    return x * y

@prickle.jit
def prickle_add_mul(x: prickle.types.F32, y: prickle.types.F32, z: prickle.types.F32):
    return prickle_mul(prickle_add(x, y), z)

@prickle.jit
def prickle_add_direct(x, y, N):
    prickle.label('N')
    y[:] = prickle_add(x[:], y[:])

@prickle.jit
def prickle_add_mul_nested(x, y, N):
    prickle.label('N')
    y[:] = prickle_add_mul(x[:], y[:], x[:])

@prickle.jit
def prickle_sum_call(x, y):
    with prickle.gpu:
        y[0] = prickle.sum(x[:])

@pytest.mark.parametrize('backend', compiler_backends)
def test_direct_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': prickle.threads(10)}
        prickle_add_direct(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_nested_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros_like(x)
        p = {'N': prickle.threads(10)}
        prickle_add_mul_nested(x, y, 10, opts=par_opts(backend, p))
        assert torch.allclose(x**2, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_sum_call_expr(backend):
    def helper():
        x = torch.randn(10)
        y = torch.zeros(1)
        p = {'N': prickle.threads(10)}
        prickle_sum_call(x, y, opts=par_opts(backend, p))
        assert torch.allclose(torch.sum(x), y)
    run_if_backend_is_enabled(backend, helper)

@prickle.jit
def add_inplace(x, y, M):
    prickle.label("1d")
    y[:] += x[:]

@prickle.jit
def add_2d_inplace(x, y, N, M):
    prickle.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)

@prickle.jit
def add_2d_inplace_x2(x, y, z, w, N, M):
    prickle.label("2d")
    for i in range(N):
        add_inplace(x[i], y[i], M)
        add_inplace(z[i], w[i], M)

@prickle.jit
def add_3d_inplace(x, y, N, M, K):
    prickle.label('3d')
    for i in range(N):
        add_2d_inplace(x[i], y[i], M, K)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        p = {'2d': prickle.threads(10), '1d': prickle.threads(15)}
        add_2d_inplace(x, y, 10, 15, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_different_types(backend):
    def helper():
        x = torch.randn(10, 15)
        y = torch.zeros_like(x)
        z = torch.randint(0, 10, (10, 15), dtype=torch.int32)
        w = torch.zeros_like(z)
        p = {'2d': prickle.threads(10), '1d': prickle.threads(15)}
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
            '3d': prickle.threads(10),
            '2d': prickle.threads(20),
            '1d': prickle.threads(30)
        }
        add_3d_inplace(x, y, 10, 20, 30, opts=par_opts(backend, p))
        assert torch.allclose(x, y)
    run_if_backend_is_enabled(backend, helper)

def test_call_non_decorated_function_fails():
    # This function is intentionally not decorated with '@prickle.jit'
    def non_decorated_add(x, y, M):
        for i in range(M):
            y[i] += x[i]
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def add_2d(x, y, N, M):
            prickle.label('N')
            for i in range(N):
                non_decorated_add(x[i], y[i], M)
    assert e_info.match(r".*unknown function non_decorated_add.*")

def test_recursive_call_fails():
    with pytest.raises(RuntimeError) as e_info:
        @prickle.jit
        def reset(x, i):
            with prickle.gpu:
                if i > 0:
                    x[i] = 0.0
                    reset(x, i-1)
    assert e_info.match(r".*unknown function reset.*")

def test_external_declaration():
    import prickle.types as types
    prickle.clear_cache()

    @prickle.external("powf", prickle.CompileBackend.Cuda, prickle.Target.Device)
    def pow(x: types.F32, y: types.F32) -> types.F32:
        return x ** y
    assert len(prickle._ext_decls) == 1

def call_external_helper(backend, fn):
    x = torch.tensor(2.0, dtype=torch.float32)
    y = torch.zeros(1, dtype=torch.float32)
    fn(x, y, opts=par_opts(backend, {}))
    assert torch.allclose(y, torch.sqrt(x))

def call_external_helper_cuda():
    @prickle.jit
    def ext_sqrt(x, y):
        with prickle.gpu:
            y[0] = sqrt_ext(x)
    call_external_helper(prickle.CompileBackend.Cuda, ext_sqrt)

def call_external_helper_metal():
    @prickle.jit
    def ext_sqrt(x, y):
        with prickle.gpu:
            y[0] = sqrt_ext(x)
    call_external_helper(prickle.CompileBackend.Metal, ext_sqrt)

@pytest.mark.parametrize('backend', compiler_backends)
def test_call_external(backend):
    import prickle.types as types
    def helper():
        if backend == prickle.CompileBackend.Cuda:
            ext_name = "sqrtf"
            header = None
        elif backend == prickle.CompileBackend.Metal:
            ext_name = "metal::sqrt"
            header = "<metal_math>"
        else:
            raise RuntimeError(f"Unsupported backend {backend}")

        @prickle.external(ext_name, backend, prickle.Target.Device, header=header)
        def sqrt_ext(x: types.F32) -> types.F32:
            return np.sqrt(x)
        if backend == prickle.CompileBackend.Cuda:
            call_external_helper_cuda()
        elif backend == prickle.CompileBackend.Metal:
            call_external_helper_metal()
    run_if_backend_is_enabled(backend, helper)

def select_distinct_element(x, l):
    for y in l:
        if x != y:
            return y
    raise RuntimeError(f"Could not find a distinct element from {x} in list {l}")

@pytest.mark.parametrize('backend', compiler_backends)
def test_invalid_backend_call(backend):
    import prickle.types as types
    def helper():
        other_backend = select_distinct_element(backend, compiler_backends)
        res_ty = types.I32

        @prickle.external("_zero", other_backend, prickle.Target.Device)
        def zero() -> types.I32:
            return 0
        with pytest.raises(RuntimeError) as e_info:
            @prickle.jit
            def f(x):
                with prickle.gpu:
                    x[:] = zero()
            x = torch.zeros(10, dtype=torch.int32)
            f(x, opts=par_opts(backend, {}))
        assert e_info.match(r"Call to unknown function zero.*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_invalid_parameter_type_annotation(backend):
    with pytest.raises(RuntimeError) as e_info:
        @prickle.external("dummy", backend, prickle.Target.Device)
        def dummy(x: int) -> prickle.types.I32:
            return x
    assert e_info.match("Unsupported parameter type annotation")

@pytest.mark.parametrize('backend', compiler_backends)
def test_invalid_return_type_annotation(backend):
    with pytest.raises(RuntimeError) as e_info:
        @prickle.external("dummy", backend, prickle.Target.Device)
        def dummy(x: prickle.types.I32) -> int:
            return x
    assert e_info.match("Unsupported return type annotation on external function")
