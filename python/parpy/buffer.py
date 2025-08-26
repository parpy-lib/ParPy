import pathlib
from .parpy import CompileBackend, DataType
from .runtime import compile_runtime_lib

def _check_errors(lib, rescode):
    if rescode != 0:
        msg = lib.parpy_get_error_message().decode('ascii')
        raise RuntimeError(f"Error in runtime library: {msg} (code={rescode})")

def _check_not_nullptr(lib, resptr):
    if resptr == 0:
        raise RuntimeError(f"{lib.parpy_get_error_message()}")
    return resptr

def _check_array_interface(intf):
    shape = intf["shape"]
    dtype = DataType(intf["typestr"])

    # We require the data pointer to be provided as part of the interface.
    if "data" in intf:
        data, ro = intf["data"]
        if ro == True:
            raise RuntimeError(f"Cannot construct buffer from read-only memory")
    else:
        raise RuntimeError(f"Buffer protocol not supported")

    # We require data to be laid out contiguously in memory
    if "strides" in intf and intf["strides"] is not None:
        raise RuntimeError(f"Buffers must only operate on contiguous memory")

    return shape, dtype, data

def _to_array_interface(ptr, dtype, shape):
    return {
        'data': (ptr, False),
        'strides': None,
        'typestr': str(dtype),
        'shape': shape,
        'version': 3
    }

def sync(backend):
    """
    Synchronizes the CPU and the target device by waiting until all running
    kernels complete.
    """
    lib = compile_runtime_lib(backend)
    _check_errors(lib, lib.parpy_sync())

class Buffer:
    def __init__(self, buf, shape, dtype, backend=None, src_ptr=None):
        self.buf = buf
        self.shape = shape
        self.dtype = dtype
        self.backend = backend
        self.src_ptr = src_ptr

        if self.backend is None:
            arr_intf = _to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        elif self.backend == CompileBackend.Cuda:
            cuda_intf = _to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__cuda_array_interface__", cuda_intf)
        elif self.backend == CompileBackend.Metal:
            compile_metal_runtime_lib()
            self.ptr = metal_lib.parpy_ptr_buffer(self.buf)
            arr_intf = _to_array_interface(self.ptr, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        else:
            raise RuntimeError(f"Unsupported compiler backend {backend}")

    def __del__(self):
        if self.buf is not None:
            self.cleanup()

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

    def cleanup(self):
        nbytes = self.dtype.size()
        for sh in self.shape:
            nbytes *= sh
        lib = compile_runtime_lib(self.backend)
        if self.backend == CompileBackend.Cuda:
            if self.src_ptr is not None:
                _check_errors(lib, lib.parpy_memcpy(self.src_ptr, self.buf, nbytes, 2))
                _check_errors(lib, lib.parpy_free_buffer(self.buf))
        elif self.backend == CompileBackend.Metal:
            # Need to wait for kernels to complete before we copy data.
            _check_errors(lib, lib.sync())
            if self.src_ptr is not None:
                _check_errors(lib, lib.parpy_memcpy(self.src_ptr, self.ptr, nbytes, 2))
            _check_errors(lib, lib.parpy_free_buffer(self.buf))

    def _from_array_cpu(t):
        # For the dummy backend, we just need any pointer to construct the
        # Buffer, so we use a CUDA pointer if this is available to ensure no
        # copying is performed (this Buffer is only used for validation
        # purposes, it should never be dereferenced).
        if hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = _check_array_interface(t.__cuda_array_interface__)
        elif hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CPU buffer")

        return Buffer(data_ptr, shape, dtype)

    def _from_array_cuda(t):
        from functools import reduce
        from operator import mul
        # If the provided argument defines the __cuda_array_interface__, we can
        # construct the buffer without copying data. Otherwise, we allocate a
        # new buffer based on the provided data.
        if hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = _check_array_interface(t.__cuda_array_interface__)
            return Buffer(data_ptr, shape, dtype, CompileBackend.Cuda)
        elif hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CUDA buffer")

        nbytes = reduce(mul, shape, 1) * dtype.size()
        lib = compile_runtime_lib(CompileBackend.Cuda)
        ptr = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        _check_errors(lib, lib.parpy_memcpy(ptr, data_ptr, nbytes, 1))
        return Buffer(ptr, shape, dtype, CompileBackend.Cuda, data_ptr)

    def _from_array_metal(t):
        from functools import reduce
        from operator import mul
        if hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = _check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to Metal buffer")

        lib = compile_runtime_lib(CompileBackend.Metal)
        nbytes = reduce(mul, shape, 1) * dtype.size()
        buf = _check_not_nullptr(lib, lib.parpy_alloc_buffer(nbytes))
        ptr = lib.parpy_ptr_buffer(buf)
        _check_errors(lib, lib.parpy_memcpy(ptr, data_ptr, nbytes, 1))
        return Buffer(buf, shape, dtype, CompileBackend.Metal, data_ptr)

    def from_array(t, backend=None):
        if backend is None:
            return Buffer._from_array_cpu(t)
        if backend == CompileBackend.Cuda:
            return Buffer._from_array_cuda(t)
        elif backend == CompileBackend.Metal:
            return Buffer._from_array_metal(t)
        else:
            raise RuntimeError(f"Unsupported buffer backend {backend}")

    def numpy(self):
        from functools import reduce
        import numpy as np
        from operator import mul
        if self.backend is None:
            return np.asarray(self)
        elif self.backend == CompileBackend.Cuda:
            a = np.ndarray(self.shape, dtype=self.dtype.to_numpy())
            shape, dtype, data_ptr = _check_array_interface(a.__array_interface__)
            nbytes = reduce(mul, shape, 1) * dtype.size()
            lib = compile_runtime_lib(self.backend)
            _check_errors(lib, lib.parpy_memcpy(data_ptr, self.buf, nbytes, 2))
            return a
        elif self.backend == CompileBackend.Metal:
            self.sync()
            return np.asarray(self)
        else:
            raise RuntimeError(f"Unsupported buffer backend {self.backend}")
