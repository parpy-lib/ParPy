import ctypes
import pathlib
import shutil
import subprocess
import numpy as np
from operator import mul
from functools import reduce
import os
from .parir import CompileBackend
import parir.state

DEFAULT_METAL_COMMAND_QUEUE_SIZE = 64
PARIR_METAL_PATH = pathlib.Path(__file__).parent / "native"
PARIR_METAL_BASE_LIB_PATH = PARIR_METAL_PATH / "parir_metal_lib.so"

metal_lib = None

# Inspired by the approach used in the cuda-python API documentation.
def check_cuda_errors(result):
    if result[0].value:
        from cuda.bindings import runtime
        _, s = runtime.cudaGetErrorString(result[0])
        raise RuntimeError(f"CUDA error {s} (code={result[0].value})")
    else:
        return result[1:]

def try_load_metal_base_lib():
    global metal_lib
    if metal_lib is None:
        if not shutil.which("clang++"):
            return None
        libpath = PARIR_METAL_BASE_LIB_PATH
        src_path = libpath.parent / "parir_metal.cpp"
        # We only need to build the library if the file does not exist or if
        # the source file was modified after the library was last built.
        if not libpath.exists() or os.path.getmtime(libpath) < os.path.getmtime(src_path):
            metal_cpp_path = parir.state.get_metal_cpp_header_path()
            if metal_cpp_path is None:
                raise RuntimeError(f"The path to the Metal C++ library must be provided \
                                     via the 'parir.set_metal_cpp_header_path' function \
                                     before using the Metal backend.")
            frameworks = ["-framework", "Metal", "-framework", "Foundation", "-framework", "MetalKit"]
            cmd = ["clang++", "-std=c++17", "-O3", "-shared", "-fpic",
                   "-I", metal_cpp_path] + frameworks + [src_path, "-o", libpath]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                stdout = r.stdout.decode('ascii')
                stderr = r.stderr.decode('ascii')
                raise RuntimeError(f"Compilation of the Metal base library failed.\
                        \nstdout:\n{stdout}\nstderr:\n{stderr}")
        lib = ctypes.cdll.LoadLibrary(libpath)
        lib.parir_init.argtypes = [ctypes.c_int64]
        lib.parir_sync.argtypes = []
        lib.parir_alloc_buffer.argtypes = [ctypes.c_int64]
        lib.parir_alloc_buffer.restype = ctypes.c_void_p
        lib.parir_ptr_buffer.argtypes = [ctypes.c_void_p]
        lib.parir_ptr_buffer.restype = ctypes.c_void_p
        lib.parir_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        lib.parir_free_buffer.argtypes = [ctypes.c_void_p]
        lib.parir_init(DEFAULT_METAL_COMMAND_QUEUE_SIZE)
        metal_lib = lib

ty_bool = 0
int8 = 1
int16 = 2
int32 = 3
int64 = 4
float16 = 5
float32 = 6
float64 = 7
typemap = {
    ty_bool: "b1",
    int8: "i1",
    int16: "i2",
    int32: "i4",
    int64: "i8",
    float16: "f2",
    float32: "f4",
    float64: "f8",
}
np_typemap = {
    ty_bool: bool,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
}
ctypemap = {
    ty_bool: ctypes.c_bool,
    int8: ctypes.c_int8,
    int16: ctypes.c_int16,
    int32: ctypes.c_int32,
    int64: ctypes.c_int64,
    float16: ctypes.c_int16,
    float32: ctypes.c_float,
    float64: ctypes.c_double,
}

def select_type(tystr):
    for k, v in typemap.items():
        if v == tystr:
            return k
    raise RuntimeError(f"Found unsupported type of type string {tystr}")

def lookup_type(ty, m):
    if ty in m:
        return m[ty]
    else:
        raise RuntimeError(f"Found unknown type {ty}")

def print_type(ty):
    return lookup_type(ty, typemap)

class BufferDtype:
    def __init__(self, typestr):
        [bo, ty, itemsz] = typestr
        self.bo = bo
        self.ty = select_type(ty + itemsz)
        self.itemsz = int(itemsz)

    def __str__(self):
        return f"{self.bo}{print_type(self.ty)}"

    def to_typestr(self):
        return self.bo + print_type(self.ty)

    def size(self):
        return self.itemsz

    def to_numpy(self):
        return lookup_type(self.ty, np_typemap)

    def to_ctype(self):
        return lookup_type(self.ty, ctypemap)

    def is_integer(self):
        return self.ty in [int8, int16, int32, int64]

    def is_float(self):
        return self.ty in [float16, float32, float64]

def check_array_interface(intf):
    shape = intf["shape"]
    dtype = BufferDtype(intf["typestr"])

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

def to_array_interface(ptr, dtype, shape):
    return {
        'data': (ptr, False),
        'strides': None,
        'typestr': dtype.to_typestr(),
        'shape': shape,
        'version': 3
    }

def sync(backend):
    if backend == CompileBackend.Cuda:
        from cuda.bindings import runtime
        check_cuda_errors(runtime.cudaDeviceSynchronize())
    elif backend == CompileBackend.Metal:
        try_load_metal_base_lib()
        metal_lib.parir_sync()
    else:
        raise RuntimeError(f"Called sync on unsupported compiler backend {backend}")

class Buffer:
    def __init__(self, buf, shape, dtype, backend, src_ptr=None):
        self.buf = buf
        self.shape = shape
        self.dtype = dtype
        self.backend = backend
        self.src_ptr = src_ptr

        if self.backend == CompileBackend.Dummy:
            arr_intf = to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        elif self.backend == CompileBackend.Cuda:
            cuda_intf = to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__cuda_array_interface__", cuda_intf)
        elif self.backend == CompileBackend.Metal:
            try_load_metal_base_lib()
            self.ptr = metal_lib.parir_ptr_buffer(self.buf)
            arr_intf = to_array_interface(self.ptr, self.dtype, self.shape)
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
        nbytes = reduce(mul, self.shape, 1) * self.dtype.size()
        if self.backend == CompileBackend.Cuda:
            try:
                from cuda.bindings import runtime
            except ImportError:
                # If we cannot import the library the program is about to quit.
                # In this case, the memory will be deallocated on exit anyway.
                return
            if self.src_ptr is not None:
                check_cuda_errors(runtime.cudaMemcpyAsync(self.src_ptr, self.buf, nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, 0))
                check_cuda_errors(runtime.cudaFreeAsync(self.buf, 0))
                self.buf = None
        elif self.backend == CompileBackend.Metal:
            if self.buf is not None:
                # Need to wait for kernels to complete before we copy data.
                self.sync()
                if self.src_ptr is not None:
                    metal_lib.parir_memcpy(self.src_ptr, self.ptr, nbytes)
                metal_lib.parir_free_buffer(self.buf)
                self.buf = None

    def from_array_cpu(t):
        # For the dummy backend, we just need any pointer to construct the
        # Buffer, so we use a CUDA pointer if this is available to ensure no
        # copying is performed (this Buffer is only used for validation
        # purposes, it should never be dereferenced).
        if hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__cuda_array_interface__)
        elif hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CPU buffer")

        return Buffer(data_ptr, shape, dtype, CompileBackend.Dummy)

    def from_array_cuda(t):
        # If the provided argument defines the __cuda_array_interface__, we can
        # construct the buffer without copying data. Otherwise, we allocate a
        # new buffer based on the provided data.
        if hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__cuda_array_interface__)
            return Buffer(data_ptr, shape, dtype, CompileBackend.Cuda)
        elif hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CUDA buffer")

        from cuda.bindings import runtime
        nbytes = reduce(mul, shape, 1) * dtype.size()
        [ptr] = check_cuda_errors(runtime.cudaMallocAsync(nbytes, 0))
        check_cuda_errors(runtime.cudaMemcpyAsync(ptr, data_ptr, nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, 0))
        return Buffer(ptr, shape, dtype, CompileBackend.Cuda, data_ptr)

    def from_array_metal(t):
        if hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = check_array_interface(t.__array__().__array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to Metal buffer")

        try_load_metal_base_lib()
        nbytes = reduce(mul, shape, 1) * dtype.size()
        buf = metal_lib.parir_alloc_buffer(nbytes)
        ptr = metal_lib.parir_ptr_buffer(buf)
        metal_lib.parir_memcpy(ptr, data_ptr, nbytes)
        return Buffer(buf, shape, dtype, CompileBackend.Metal, data_ptr)

    def from_array(t, backend):
        if backend == CompileBackend.Dummy:
            return Buffer.from_array_cpu(t)
        if backend == CompileBackend.Cuda:
            return Buffer.from_array_cuda(t)
        elif backend == CompileBackend.Metal:
            return Buffer.from_array_metal(t)
        else:
            raise RuntimeError(f"Unsupported buffer backend {backend}")

    def numpy(self):
        if self.backend == CompileBackend.Dummy:
            return np.asarray(self)
        elif self.backend == CompileBackend.Cuda:
            from cuda.bindings import runtime
            a = np.ndarray(self.shape, dtype=self.dtype.to_numpy())
            shape, dtype, data_ptr = check_array_interface(a.__array_interface__)
            nbytes = reduce(mul, shape, 1) * dtype.size()
            check_cuda_errors(runtime.cudaMemcpy(data_ptr, self.buf, nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            return a
        elif self.backend == CompileBackend.Metal:
            self.sync()
            return np.asarray(self)
        else:
            raise RuntimeError(f"Unsupported buffer backend {self.backend}")
