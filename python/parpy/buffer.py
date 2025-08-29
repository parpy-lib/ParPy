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

def sync(backend):
    """
    Synchronizes the CPU and the target device by waiting until all running
    kernels complete.
    """
    lib = _compile_runtime_lib(backend)
    _check_errors(lib, lib.parpy_sync())

def empty(shape, dtype, backend):
    dtype = _resolve_dtype(dtype)
    lib = _compile_runtime_lib(backend)
    ptr = Buffer._alloc_data(shape, dtype, lib)
    return Buffer(ptr, shape, dtype, backend=backend)

def empty_like(b):
    return empty(b.shape, b.dtype, b.backend)

def zeros(shape, dtype, backend):
    b = empty(shape, dtype, backend)
    lib = _compile_runtime_lib(backend)
    _check_errors(lib, lib.parpy_memset(b.buf, b.size(), 0))
    return b

def zeros_like(b):
    return zeros(shape, dtype, backend)

class Buffer:
    def __init__(self, buf, shape, dtype, backend=None, src=None, refcount=None):
        if refcount is None:
            self.refcount = [1]
        else:
            self.refcount = refcount
            self.refcount[0] += 1

        self.buf = buf
        self.shape = tuple(shape)
        self.dtype = _resolve_dtype(dtype)
        self.backend = backend
        self.src = src

        if self.backend is None:
            arr_intf = _to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        elif self.backend == CompileBackend.Cuda:
            cuda_intf = _to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__cuda_array_interface__", cuda_intf)
        elif self.backend == CompileBackend.Metal:
            lib = _compile_runtime_lib(self.backend)
            self.ptr = lib.parpy_ptr_buffer(self.buf)
            arr_intf = _to_array_interface(self.ptr, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        else:
            raise RuntimeError(f"Unsupported compiler backend {backend}")

    def __del__(self):
        self.refcount[0] -= 1
        if self.refcount[0] == 0:
            nbytes = self.dtype.size()
            for sh in self.shape:
                nbytes *= sh
            if self.src is not None:
                allow_cuda = self.backend == CompileBackend.Cuda
                _, _, src_ptr = _extract_array_interface(self.src, allow_cuda=allow_cuda)
            else:
                src_ptr = None
            if self.backend == CompileBackend.Cuda:
                lib = _compile_runtime_lib(self.backend)
                if src_ptr is not None:
                    _check_errors(lib, lib.parpy_memcpy(src_ptr, self.buf, nbytes, 2))
                    _check_errors(lib, lib.sync())
                else:
                    _check_errors(lib, lib.parpy_free_buffer(self.buf))
            elif self.backend == CompileBackend.Metal:
                lib = _compile_runtime_lib(self.backend)
                _check_errors(lib, lib.sync())
                if src_ptr is not None:
                    _check_errors(lib, lib.parpy_memcpy(src_ptr, self.buf, nbytes, 2))
                _check_errors(lib, lib.parpy_free_buffer(self.buf))

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

    def sync(self):
        sync(self.backend)

    def _alloc_data(shape, dtype, lib):
        nbytes = _size(shape, dtype)
        return _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))

    def _from_array_cpu(t):
        # For the dummy backend, we just need any pointer to construct the
        # Buffer, so we use a CUDA pointer if this is available to ensure no
        # copying is performed (this Buffer is only used for validation
        # purposes, it should never be dereferenced).
        try:
            shape, dtype, data_ptr = _extract_array_interface(t, allow_cuda=True)
        except ValueError:
            raise RuntimeError(f"Cannot convert argument {t} to CPU buffer")
        return Buffer(data_ptr, shape, dtype, None)

    def _from_array_cuda(t):
        # If the provided argument defines the __cuda_array_interface__, we can
        # construct the buffer without copying data. Otherwise, we allocate a
        # new buffer based on the provided data.
        try:
            shape, dtype, data_ptr = _extract_array_interface(t, allow_cuda=True)
            if hasattr(t, "__cuda_array_interface__"):
                return Buffer(data_ptr, shape, dtype, CompileBackend.Cuda, src=t)
        except ValueError:
            raise RuntimeError(f"Cannot convert argument {t} to CUDA buffer")

        lib = _compile_runtime_lib(CompileBackend.Cuda)
        ptr = Buffer._alloc_data(shape, dtype, lib)
        _check_errors(lib, lib.parpy_memcpy(ptr, data_ptr, _size(shape, dtype), 1))
        _check_errors(lib, lib.sync())
        return Buffer(ptr, shape, dtype, CompileBackend.Cuda, src=t)

    def _from_array_metal(t):
        try:
            shape, dtype, data_ptr = _extract_array_interface(t)
        except ValueError:
            raise RuntimeError(f"Cannot convert argument {t} to Metal buffer")

        lib = _compile_runtime_lib(CompileBackend.Metal)
        buf = Buffer._alloc_data(shape, dtype, lib)
        _check_errors(lib, lib.parpy_memcpy(buf, data_ptr, _size(shape, dtype), 1))
        _check_errors(lib, lib.sync())
        return Buffer(buf, shape, dtype, CompileBackend.Metal, src=t)

    def from_array(t, backend):
        if backend is None:
            return Buffer._from_array_cpu(t)
        if backend == CompileBackend.Cuda:
            return Buffer._from_array_cuda(t)
        elif backend == CompileBackend.Metal:
            return Buffer._from_array_metal(t)
        else:
            raise RuntimeError(f"Unsupported buffer backend {backend}")

    def size(self):
        return _size(self.shape, self.dtype)

    def numpy(self):
        import numpy as np
        if self.backend is None:
            return np.asarray(self)
        else:
            a = np.ndarray(self.shape, dtype=self.dtype.to_numpy())
            _, _, data_ptr = _check_array_interface(a.__array_interface__)
            lib = _compile_runtime_lib(self.backend)
            _check_errors(lib, lib.parpy_memcpy(data_ptr, self.buf, self.size(), 2))
            _check_errors(lib, lib.sync())
            return a

    def torch_ref(self):
        """
        Constructs a reference to the data in torch when using the CUDA
        backend. Otherwise, it produces a tensor containing a copy of the data
        in the original buffer.
        """
        import torch
        if self.backend is None:
            return torch.as_tensor(self.numpy())
        elif self.backend == CompileBackend.Cuda:
            return torch.as_tensor(self, device='cuda')
        elif self.backend == CompileBackend.Metal:
            return torch.as_tensor(self.numpy())
        else:
            raise RuntimeError(f"Unsupported buffer backend {self.backend}")

    def reshape(self, *dims):
        import math
        curr_sz = math.prod(self.shape)
        new_shape = tuple(dims)
        new_sz = math.prod(new_shape)
        if curr_sz == new_sz:
            return Buffer(self.buf, new_shape, self.dtype, backend=self.backend, src=self.src, refcount=self.refcount)
        else:
            raise ValueError(f"Cannot reshape buffer of shape {self.shape} to {new_shape}")

    def with_type(self, new_dtype):
        new_dtype = _resolve_dtype(new_dtype)
        if isinstance(new_dtype, DataType):
            b = empty(self.shape, new_dtype, self.backend)
            lib = _compile_runtime_lib(self.backend)
            _check_errors(lib, lib.parpy_memcpy(b.buf, self.buf, self.size(), 3))
            _check_errors(lib, lib.sync())
            return b
        else:
            raise ValueError(f"Found unsupported data type: {type(new_dtype)}")
