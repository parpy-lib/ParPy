import parpy

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
def test_buffer_zeros(backend):
    def helper():
        import numpy as np
        shape = (20, 10, 32)
        b = parpy.buffer.zeros(shape, parpy.types.F32, backend)
        n = b.numpy()
        assert np.allclose(n, np.zeros(shape, dtype=np.float32))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_reshape_refcount(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.zeros(shape, parpy.types.F32, backend)
        assert b1.refcount[0] == 1
        b2 = b1.reshape(200, 32)
        assert b1.shape == shape
        assert b2.shape == (200, 32)
        assert b1.refcount[0] == 2 and b2.refcount[0] == 2
        del b2
        assert b1.refcount[0] == 1
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
        assert b1.buf != b2.buf
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_convert_int_to_float_type(backend):
    def helper():
        shape = (20, 10, 32)
        b1 = parpy.buffer.zeros(shape, parpy.types.I32, backend)
        b2 = b1.with_type(parpy.types.F64)
        assert b1.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.I32)
        assert b2.dtype == parpy.buffer.DataType.from_elem_size(parpy.types.F64)
        assert b1.buf != b2.buf
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_back_to_back_conversion(backend):
    def helper():
        import numpy as np
        shape = (20, 10, 32)
        a = np.random.randn(*shape)
        b = parpy.buffer.Buffer.from_array(a, backend)
        c = b.numpy()
        assert np.allclose(a, c)
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.parametrize('backend', compiler_backends)
def test_buffer_torch_ref(backend):
    def helper():
        import numpy as np
        import torch
        shape = (20, 10, 32)
        a = parpy.buffer.zeros(shape, parpy.types.F32, backend)
        b = a.torch_ref()
        assert b.dtype == torch.float32
        if backend == parpy.CompileBackend.Cuda:
            assert b.device == torch.device('cuda', index=0)
        else:
            assert b.device == torch.device('cpu')
        b[0] = 5.0
        v = a.numpy()
        assert np.allclose(v[0], 5.0)
    run_if_backend_is_enabled(backend, helper)
