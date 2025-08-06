import pathlib
from .prickle import CompileBackend, DataType

DEFAULT_METAL_COMMAND_QUEUE_SIZE = 64
PARIR_NATIVE_PATH = pathlib.Path(__file__).parent / "native"
PARIR_METAL_BASE_LIB_PATH = PARIR_NATIVE_PATH / "prickle_metal_lib.so"

metal_lib = None

# Inspired by the approach used in the cuda-python API documentation.
def _check_cuda_errors(result):
    if result[0].value:
        from cuda.bindings import runtime
        _, s = runtime.cudaGetErrorString(result[0])
        raise RuntimeError(f"CUDA error {s} (code={result[0].value})")
    else:
        return result[1:]

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

def compile_metal_runtime_lib():
    """
    Compiles the Metal runtime library if it has not already been compiled. The
    Metal runtime library provides convenient wrappers handling details such as
    the launch of compute kernels and memory allocations. A runtime exception
    is raised if the function is called when the Metal backend is disabled.
    """
    from .backend import is_enabled
    import ctypes
    import shutil
    import subprocess
    import os
    global metal_lib
    if metal_lib is None:
        if not is_enabled(CompileBackend.Metal):
            raise RuntimeError("Cannot build Metal runtime library when the Metal backend is disabled")
        libpath = PARIR_METAL_BASE_LIB_PATH
        src_path = libpath.parent / "prickle_metal.cpp"
        # We only need to build the library if the file does not exist or if
        # the source file was modified after the library was last built.
        if not libpath.exists() or os.path.getmtime(libpath) < os.path.getmtime(src_path):
            metal_cpp_path = os.getenv("METAL_CPP_HEADER_PATH")
            if metal_cpp_path is None:
                raise RuntimeError("The path to the Metal C++ library must be provided " +
                                   "via the 'METAL_CPP_HEADER_PATH' variable before " +
                                   "using the Metal backend.")
            frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalKit"]
            cmd = ["clang++", "-std=c++17", "-O3", "-shared", "-fpic",
                   "-I", metal_cpp_path] + frameworks + [src_path, "-o", libpath]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                stdout = r.stdout.decode('ascii')
                stderr = r.stderr.decode('ascii')
                raise RuntimeError("Compilation of the Metal base library failed.\n" +
                                  f"stdout:\n{stdout}\nstderr:\n{stderr}")
        lib = ctypes.cdll.LoadLibrary(libpath)
        lib.prickle_init.argtypes = [ctypes.c_int64]
        lib.prickle_sync.argtypes = []
        lib.prickle_alloc_buffer.argtypes = [ctypes.c_int64]
        lib.prickle_alloc_buffer.restype = ctypes.c_void_p
        lib.prickle_ptr_buffer.argtypes = [ctypes.c_void_p]
        lib.prickle_ptr_buffer.restype = ctypes.c_void_p
        lib.prickle_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        lib.prickle_free_buffer.argtypes = [ctypes.c_void_p]
        lib.prickle_init(DEFAULT_METAL_COMMAND_QUEUE_SIZE)
        metal_lib = lib

def sync(backend):
    """
    Synchronizes the CPU and the target device by waiting until all running
    kernels complete.
    """
    if backend == CompileBackend.Cuda:
        from cuda.bindings import runtime
        _check_cuda_errors(runtime.cudaDeviceSynchronize())
    elif backend == CompileBackend.Metal:
        compile_metal_runtime_lib()
        metal_lib.prickle_sync()
    else:
        raise RuntimeError(f"Called sync on unsupported compiler backend {backend}")

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
            self.ptr = metal_lib.prickle_ptr_buffer(self.buf)
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
        from functools import reduce
        from operator import mul
        nbytes = reduce(mul, self.shape, 1) * self.dtype.size()
        if self.backend == CompileBackend.Cuda:
            try:
                from cuda.bindings import runtime
            except ImportError:
                # If we cannot import the library the program is about to quit.
                # In this case, the memory will be deallocated on exit anyway.
                return
            if self.src_ptr is not None:
                _check_cuda_errors(runtime.cudaMemcpyAsync(self.src_ptr, self.buf, nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, 0))
                _check_cuda_errors(runtime.cudaFreeAsync(self.buf, 0))
                self.buf = None
        elif self.backend == CompileBackend.Metal:
            if self.buf is not None:
                # Need to wait for kernels to complete before we copy data.
                self.sync()
                if self.src_ptr is not None:
                    metal_lib.prickle_memcpy(self.src_ptr, self.ptr, nbytes)
                metal_lib.prickle_free_buffer(self.buf)
                self.ptr = None
                self.buf = None

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
        from cuda.bindings import runtime
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
        [ptr] = _check_cuda_errors(runtime.cudaMallocAsync(nbytes, 0))
        _check_cuda_errors(runtime.cudaMemcpyAsync(ptr, data_ptr, nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, 0))
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

        compile_metal_runtime_lib()
        nbytes = reduce(mul, shape, 1) * dtype.size()
        buf = metal_lib.prickle_alloc_buffer(nbytes)
        ptr = metal_lib.prickle_ptr_buffer(buf)
        metal_lib.prickle_memcpy(ptr, data_ptr, nbytes)
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
            from cuda.bindings import runtime
            a = np.ndarray(self.shape, dtype=self.dtype.to_numpy())
            shape, dtype, data_ptr = _check_array_interface(a.__array_interface__)
            nbytes = reduce(mul, shape, 1) * dtype.size()
            _check_cuda_errors(runtime.cudaMemcpy(data_ptr, self.buf, nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            return a
        elif self.backend == CompileBackend.Metal:
            self.sync()
            return np.asarray(self)
        else:
            raise RuntimeError(f"Unsupported buffer backend {self.backend}")
