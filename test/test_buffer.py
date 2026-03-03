import numpy as np
import parpy
import torch

from common import *

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_empty(backend):
    def helper():
        shape = (20, 10, 32)
        b = parpy.buffer.empty(shape, parpy.types.F32, backend)
        n = b.numpy()
        assert n.shape == shape
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_empty_like(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.empty(shape, parpy.types.F32, backend)
        b2 = parpy.buffer.empty_like(b1)
        assert b1.shape == b2.shape
        assert b1._get_ptr() != b2._get_ptr()
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_zeros(backend):
    def helper():
        shape = (20, 10, 32)
        b = parpy.buffer.zeros(shape, parpy.types.F32, backend)
        n = b.numpy()
        assert np.allclose(n, np.zeros(shape, dtype=np.float32))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_zeros_like(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.empty(shape, parpy.types.F32, backend)
        b2 = parpy.buffer.zeros_like(b1)
        assert b1.shape == b2.shape
        assert b1._get_ptr() != b2._get_ptr()
        assert np.allclose(b2.numpy(), np.zeros(shape, dtype=np.float32))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_from_numpy_array(backend):
    def helper():
        shape = (20, 10, 32)
        a = np.ndarray(shape, dtype=np.float32)
        b = parpy.buffer.from_array(a, backend)
        assert b.shape == shape
        assert b.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.F32)
        assert b.buf.src is a
    run_if_backend_is_enabled(backend, helper)

def test_buffer_from_array_none_backend():
    shape = (20, 10, 32)
    a = np.ndarray(shape, dtype=np.float32)
    b = parpy.buffer.from_array(a, None)
    assert b.shape == shape
    assert b.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.F32)

def test_buffer_from_array_invalid_backend():
    a = np.ndarray((10,), dtype=np.float32)
    with pytest.raises(ValueError) as e_info:
        b = parpy.buffer.from_array(a, 5)
    assert e_info.match("Cannot convert to buffer of unknown backend 5")

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_singleton_to_float(backend):
    def helper():
        a = np.array(2.5, dtype=np.float32)
        b = parpy.buffer.from_array(a, backend)
        assert float(b) == 2.5
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_singleton_to_int(backend):
    def helper():
        a = np.array(2, dtype=np.int32)
        b = parpy.buffer.from_array(a, backend)
        assert int(b) == 2
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_singleton_to_bool(backend):
    def helper():
        a = torch.tensor(True, dtype=torch.bool)
        b = parpy.buffer.from_array(a, backend)
        assert bool(b)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_size(backend):
    def helper():
        shape = (20, 10, 32)
        dtype = parpy.types.I16
        b = parpy.buffer.empty(shape, dtype, backend)
        assert dtype.size() == 2
        assert b.size() == 20 * 10 * 32 * 2
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_to_numpy(backend):
    def helper():
        shape = (20, 10, 32)
        dtype = parpy.types.I16
        b = parpy.buffer.zeros(shape, dtype, backend)
        a = b.numpy()
        assert a.shape == shape
        assert a.dtype == np.int16
        assert np.allclose(a, 0)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_to_torch(backend):
    def helper():
        shape = (20, 10, 32)
        dtype = parpy.types.I16
        b = parpy.buffer.zeros(shape, dtype, backend)
        a = b.torch()
        assert a.shape == shape
        assert a.dtype == torch.int16
        assert torch.allclose(a, torch.zeros(shape, dtype=torch.int16, device=a.device))
        if backend in [parpy.CompileBackend.Cuda, parpy.CompileBackend.Triton]:
            assert a.device == torch.device('cuda', 0)
        else:
            assert a.device == torch.device('cpu')
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_invalid_reshape(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.zeros(shape, parpy.types.F32, backend)
        with pytest.raises(ValueError) as e_info:
            b1.reshape(20, 10, 32, 2)
        assert e_info.match("Cannot reshape buffer of shape.*to.*")
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_int_sign_conversion(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.zeros(shape, parpy.types.I32, backend)
        b2 = b1.with_type(parpy.types.U32)
        assert b1.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.I32)
        assert b2.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.U32)
        assert b1.size() == b2.size()
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_convert_int_to_float_type(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.zeros(shape, parpy.types.I32, backend)
        b2 = b1.with_type(parpy.types.F64)
        assert b1.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.I32)
        assert b2.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.F64)
        assert b1._get_ptr() != b2._get_ptr()
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_back_to_back_conversion(backend):
    def helper():
        shape = (20, 10, 32)
        a = np.random.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        c = b.numpy()
        assert np.allclose(a, c)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_indexing(backend):
    def helper():
        shape = (20,)
        a = np.random.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        assert b[12] == a[12]
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_multidim_indexing_elem(backend):
    def helper():
        shape = (20, 10, 32)
        a = np.random.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        print(b[12,9,14])
        print(a[12,9,14])
        assert b[12, 9, 14] == a[12, 9, 14]
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_multidim_partial_indexing(backend):
    def helper():
        import numpy as np
        shape = (20, 10, 32)
        a = np.random.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        assert np.allclose(b[12].numpy(), a[12])
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_multidim_partial_indexing_torch(backend):
    def helper():
        import torch
        shape = (20, 10, 32)
        a = torch.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        if backend in [parpy.CompileBackend.Cuda, parpy.CompileBackend.Triton]:
            assert torch.allclose(b[12].torch(), a[12].cuda())
        else:
            assert torch.allclose(b[12].torch(), a[12])
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_indexing_out_of_bounds(backend):
    def helper():
        shape = (20, 10, 32)
        a = np.random.randn(*shape)
        b = parpy.buffer.from_array(a, backend)
        with pytest.raises(IndexError):
            b[21]
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_call_with_indexed_argument(backend):
    def helper():
        N = parpy.types.shape_var()
        @parpy.jit
        def elemwise_add_one(x: parpy.types.buffer(parpy.types.F32, [N])):
            for i in range(N):
                x[i] += 1.0
        shape = (2, 32)
        a = np.random.randn(*shape).astype(np.float32)
        b = parpy.buffer.from_array(a, backend)
        elemwise_add_one(b[1], opts=par_opts(backend, {'N': parpy.threads(32)}))
        a[1,:] += 1.0
        assert np.allclose(a, b.numpy(), atol=1e-5)
    run_if_backend_is_enabled(backend, helper)
