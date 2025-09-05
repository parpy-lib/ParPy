import pathlib
from .parpy import CompileBackend, DataType, ElemSize
from .runtime import _compile_runtime_lib

def _check_errors(lib, rescode):
    if rescode != 0:
        msg = lib.parpy_get_error_message().decode('ascii')
        raise RuntimeError(f"Runtime library error: {msg} (code={rescode})")

def _check_not_nullptr(lib, resptr):
    if resptr == 0:
        raise RuntimeError(f"Runtime library error: {lib.parpy_get_error_message()}")
    return resptr

def _check_array_interface(intf):
    shape = intf["shape"]
    dtype = DataType(intf["typestr"])

    # We require the data pointer to be provided as part of the interface.
    if "data" in intf:
        data, ro = intf["data"]
        if ro == True:
            raise ValueError(f"Cannot construct buffer from read-only memory")
    else:
        raise ValueError(f"Buffer protocol not supported")

    # We require data to be laid out contiguously in memory
    if "strides" in intf and intf["strides"] is not None:
        raise ValueError(f"Buffers must only operate on contiguous memory")

    return shape, dtype, data

def _to_array_interface(ptr, dtype, shape):
    return {
        'data': (ptr, False),
        'strides': None,
        'typestr': str(dtype),
        'shape': shape,
        'version': 3
    }

def _extract_array_interface(a, allow_cuda=False):
    if allow_cuda and hasattr(a, "__cuda_array_interface__"):
        return _check_array_interface(a.__cuda_array_interface__)
    elif hasattr(a, "__array_interface__"):
        return _check_array_interface(a.__array_interface__)
    elif hasattr(a, "__array__"):
        return _check_array_interface(a.__array__().__array_interface__)
    else:
        raise ValueError("Failed to extract array interface")

def _resolve_dtype(dtype):
    """
    Resolves the provided dtype - provided to allow users to construct buffers
    using the more easily accessible types defined in the 'parpy.types' module.
    """
    if isinstance(dtype, ElemSize):
        return DataType.from_elem_size(dtype)
    else:
        return dtype

def _size(shape, dtype):
    sz = dtype.size()
    for dim in shape:
        sz *= dim
    return sz

def _alloc_data(shape, dtype, lib):
    nbytes = _size(shape, dtype)
    return _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))

def sync(backend):
    """
    Synchronizes the CPU and the target device by waiting until all running
    kernels complete.
    """
    lib = _compile_runtime_lib(backend)
    _check_errors(lib, lib.parpy_sync())

def empty(shape, dtype, backend):
    dtype = _resolve_dtype(dtype)
    if backend == CompileBackend.Cuda:
        import torch
        t = torch.empty(*shape, dtype=dtype.to_torch(), device='cuda')
        return CudaBuffer(t, shape, dtype)
    else:
        lib = _compile_runtime_lib(backend)
        nbytes = _size(shape, dtype)
        buf = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        return MetalBuffer(buf, shape, dtype)

def empty_like(b):
    return empty(b.shape, b.dtype, b.backend)

def zeros(shape, dtype, backend):
    dtype = _resolve_dtype(dtype)
    if backend == CompileBackend.Cuda:
        import torch
        t = torch.zeros(*shape, dtype=dtype.to_torch(), device='cuda')
        return CudaBuffer(t, shape, dtype)
    else:
        b = empty(shape, dtype, backend)
        lib = _compile_runtime_lib(backend)
        _check_errors(lib, lib.parpy_memset(b.buf, b.size(), 0))
        return b

def zeros_like(b):
    return zeros(b.shape, b.dtype, b.backend)

def from_array(t, backend):
    if backend is None:
        return DummyBuffer._from_array(t)
    elif backend == CompileBackend.Cuda:
        return CudaBuffer._from_array(t)
    elif backend == CompileBackend.Metal:
        return MetalBuffer._from_array(t)
    else:
        raise ValueError(f"Cannot convert to buffer of unknown backend {backend}")

class Buffer:
    def __init__(self, buf, shape, dtype, src=None, refcount=None):
        if type(self) is Buffer:
            raise RuntimeError(f"Cannot construct instance of base Buffer class")

        self.buf = buf
        self.shape = shape
        self.dtype = _resolve_dtype(dtype)
        self.src = src

        if refcount is None:
            self.refcount = [1]
        else:
            self.refcount = refcount
            self.refcount[0] += 1

    def __del__(self):
        self.refcount[0] -= 1
        if self.refcount[0] == 0:
            if self.src is not None:
                _, _, src_ptr = _extract_array_interface(self.src)
            else:
                src_ptr = None

            self._deconstruct(src_ptr)

    def __float__(self):
        if len(self.shape) == 0:
            return float(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to float")

    def __int__(self):
        if len(self.shape) == 0:
            return int(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to int")

    def __bool__(self):
        if len(self.shape) == 0:
            return bool(self.numpy().item())
        else:
            raise ValueError(f"Cannot convert buffer of shape {self.shape} to bool")

    def __index__(self):
        if len(self.shape) == 0:
            if self.dtype.is_integer():
                return self.__int__()
        raise ValueError(f"Cannot use buffer of shape {self.shape} and type {self.dtype} as index")

    def _deconstruct(self, src_ptr):
        pass

    def _get_ptr(self):
        return self.buf

    def sync(self):
        sync(self.backend)

    def size(self):
        return _size(self.shape, self.dtype)

    def numpy(self):
        import numpy as np
        return np.asarray(self)

    def torch(self):
        import torch
        return torch.as_tensor(self.numpy())

    def copy(self):
        raise RuntimeError(f"Cannot instantiate base Buffer class")

    def reshape(self, *dims):
        import math
        curr_sz = math.prod(self.shape)
        new_shape = tuple(dims)
        new_sz = math.prod(new_shape)
        if curr_sz == new_sz:
            return type(self)(self.buf, new_shape, self.dtype, self.src, self.refcount)
        else:
            raise ValueError(f"Cannot reshape buffer of shape {self.shape} to {new_shape}")

    def with_type(self, new_dtype):
        raise RuntimeError(f"Cannot instantiate base Buffer class")

class DummyBuffer(Buffer):
    def __init__(self, buf, shape, dtype, src=None, refcount=None):
        super().__init__(buf, shape, dtype, src, refcount)
        arr_intf = _to_array_interface(self.buf, self.dtype, self.shape)
        self.__array_interface__ = arr_intf

    @classmethod
    def _make_view(buf, shape, dtype, src, refcount):
        return DummyBuffer(buf, shape, dtype, src, refcount)

    def _from_array(t):
        try:
            shape, dtype, data_ptr = _extract_array_interface(t, allow_cuda=True)
        except ValueError:
            raise RuntimeError(f"Cannot convert argument {t} to CPU buffer")
        return DummyBuffer(data_ptr, shape, dtype)

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            return DummyBuffer(self.buf, self.shape, new_dtype, self.src, self.refcount)
        else:
            raise ValueError(f"Found unsupported data type: {type(new_dtype)}")

class CudaBuffer(Buffer):
    def __init__(self, buf, shape, dtype, src=None, refcount=None):
        super().__init__(buf, shape, dtype, src, refcount)
        self.__cuda_array_interface__ = buf.__cuda_array_interface__
        self.backend = CompileBackend.Cuda
        self.lib = _compile_runtime_lib(self.backend)

    def _deconstruct(self, src_ptr):
        if src_ptr is not None:
            _, _, buf_ptr = _extract_array_interface(self.buf, allow_cuda=True)
            nbytes = _size(self.shape, self.dtype)
            _check_errors(self.lib, self.lib.parpy_memcpy(src_ptr, buf_ptr, self.size(), 2))
            self.sync()

    def _from_array(t):
        try:
            shape, dtype, data_ptr = _extract_array_interface(t, allow_cuda=True)
            if hasattr(t, "__cuda_array_interface__"):
                return CudaBuffer(t, shape, dtype)
        except ValueError:
            raise ValueError(f"Cannot convert argument {t} to a CUDA buffer")

        import torch
        if len(shape) == 0:
            data = torch.empty((), dtype=dtype.to_torch(), device='cuda')
        else:
            data = torch.empty(*shape, dtype=dtype.to_torch(), device='cuda')
        _, _, ptr = _extract_array_interface(data, allow_cuda=True)
        lib = _compile_runtime_lib(CompileBackend.Cuda)
        _check_errors(lib, lib.parpy_memcpy(ptr, data_ptr, _size(shape, dtype), 1))
        return CudaBuffer(data, shape, dtype, src=t)

    def _get_ptr(self):
        _, _, ptr = _extract_array_interface(self.buf, allow_cuda=True)
        return ptr

    def numpy(self):
        import numpy as np
        return np.asarray(self.buf.cpu())

    def torch(self):
        return self.buf

    def copy(self):
        data = self.buf.detach().clone()
        return CudaBuffer(data, self.shape, self.dtype)

    def reshape(self, *dims):
        b = super().reshape(*dims)
        b.buf = b.buf.reshape(b.shape)
        return b

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            if self.dtype.size() == new_dtype.size():
                return CudaBuffer(self.buf, self.shape, new_dtype, self.src, self.refcount)
            else:
                t = self.buf.detach().clone().to(new_dtype.to_torch())
                return CudaBuffer(t, self.shape, new_dtype)
        else:
            raise ValueError(f"Found unsupported data type: {type(new_dtype)}")

class MetalBuffer(Buffer):
    def __init__(self, buf, shape, dtype, src=None, refcount=None):
        super().__init__(buf, shape, dtype, src, refcount)
        self.backend = CompileBackend.Metal
        self.lib = _compile_runtime_lib(self.backend)
        ptr = self.lib.parpy_ptr_buffer(self.buf)
        arr_intf = _to_array_interface(ptr, self.dtype, self.shape)
        self.__array_interface__ = arr_intf

    def _deconstruct(self, src_ptr):
        self.sync()
        if src_ptr is not None:
            _check_errors(self.lib, self.lib.parpy_memcpy(src_ptr, self.buf, self.size(), 2))
        _check_errors(self.lib, self.lib.parpy_free_buffer(self.buf))

    def _from_array(t):
        try:
            shape, dtype, data_ptr = _extract_array_interface(t)
        except ValueError:
            raise ValueError(f"Cannot convert argument {t} to a Metal buffer")

        lib = _compile_runtime_lib(CompileBackend.Metal)
        nbytes = _size(shape, dtype)
        buf = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        _check_errors(lib, lib.parpy_memcpy(buf, data_ptr, nbytes, 1))
        return MetalBuffer(buf, shape, dtype, src=t)

    def _get_ptr(self):
        return self.buf

    def numpy(self):
        import numpy as np
        a = np.ndarray(self.shape, dtype=self.dtype.to_numpy())
        _, _, data_ptr = _check_array_interface(a.__array_interface__)
        self.sync()
        _check_errors(self.lib, self.lib.parpy_memcpy(data_ptr, self.buf, self.size(), 2))
        return a

    def copy(self):
        b = empty(self.shape, self.dtype, self.backend)
        self.sync()
        _check_errors(self.lib, self.lib.parpy_memcpy(b.buf, self.buf, self.size(), 3))
        return b

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            if self.dtype.size() == new_dtype.size():
                return MetalBuffer(self.buf, self.shape, new_dtype, self.src, self.refcount)
            else:
                import numpy as np
                t = np.asarray(self).astype(dtype=new_dtype.to_numpy())
                return MetalBuffer._from_array(t)
        else:
            raise ValueError(f"Found unsupported data type: {type(new_dtype)}")
