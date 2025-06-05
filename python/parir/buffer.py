import ctypes
import pathlib
import shutil
import subprocess
from operator import mul
from functools import reduce
import os
from .parir import CompileBackend
import parir.state

DEFAULT_METAL_COMMAND_QUEUE_SIZE = 64
PARIR_METAL_PATH = pathlib.Path(__file__).parent / "native"
PARIR_METAL_BASE_LIB_PATH = PARIR_METAL_PATH / "parir_metal_lib.so"

metal_lib = None

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

bool = 0
int8 = 1
int16 = 2
int32 = 3
int64 = 4
float16 = 5
float32 = 6
float64 = 7
typemap = {
    bool: "b1",
    int8: "i1",
    int16: "i2",
    int32: "i4",
    int64: "i8",
    float16: "f2",
    float32: "f4",
    float64: "f8",
}

def select_type(tystr):
    for k, v in typemap.items():
        if v == tystr:
            return k
    raise RuntimeError(f"Found unsupported type of type string {tystr}")

def print_type(ty):
    if ty in typemap:
        return typemap[ty]
    else:
        raise RuntimeError(f"Found unknown type {ty}")

class BufferDtype:
    def __init__(self, typestr):
        [bo, ty, itemsz] = typestr
        self.bo = bo
        self.ty = select_type(ty + itemsz)
        self.itemsz = int(itemsz)

    def to_typestr(self):
        return self.bo + print_type(self.ty)

    def size(self):
        return self.itemsz

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

class Buffer:
    def __init__(self, buf, shape, dtype, backend, src_ptr=None):
        self.buf = buf
        self.shape = shape
        self.dtype = dtype
        self.backend = backend
        self.src_ptr = src_ptr
        if self.backend is None:
            arr_intf = to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        elif self.backend == CompileBackend.Cuda:
            cuda_intf = to_array_interface(self.buf, self.dtype, self.shape)
            setattr(self, "__cuda_array_interface__", cuda_arr_intf)
        elif self.backend == CompileBackend.Metal:
            self.lib = metal_lib
            self.ptr = self.lib.parir_ptr_buffer(self.buf)
            arr_intf = to_array_interface(self.ptr, self.dtype, self.shape)
            setattr(self, "__array_interface__", arr_intf)
        else:
            raise RuntimeError(f"Unsupported compiler backend {backend}")

    def __del__(self):
        if self.buf is not None:
            self.cleanup()

    def cleanup(self):
        nbytes = reduce(mul, self.shape, 1) * self.dtype.size()
        if self.backend == CompileBackend.Cuda:
            if self.src_ptr is not None:
                from cuda.bindings import runtime
                err = runtime.cudaMemcpy(self.src_ptr, self.buf, nbytes, runtime.cudaMemcpyKind(4))
        if self.backend == CompileBackend.Metal:
            if self.buf is not None:
                # Need to wait for kernels to complete before we copy data.
                self.sync()
                if self.src_ptr is not None:
                    self.lib.parir_memcpy(self.src_ptr, self.ptr, nbytes)
                self.lib.parir_free_buffer(self.buf)
                self.buf = None

    def sync(self):
        if self.backend == CompileBackend.Metal:
            self.lib.sync()

    def from_array_cpu(t):
        if hasattr(t, "__array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__array_interface__)
        elif hasattr(t, "__array__"):
            shape, dtype, data_ptr = check_array_interface(t.__array__().__array_interface__)
        elif hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = check_array_interface(t.__cuda_array_interface__)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CPU buffer")

        return Buffer(data_ptr, shape, dtype, None)

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
        err, ptr = runtime.cudaMalloc(nbytes)
        err = runtime.cudaMemcpy(ptr, data_ptr, nbytes, runtime.cudaMemcpyKind(4))
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
        if backend is None:
            return Buffer.from_array_cpu(t)
        if backend == CompileBackend.Cuda:
            return Buffer.from_array_cuda(t)
        elif backend == CompileBackend.Metal:
            return Buffer.from_array_metal(t)
        else:
            raise RuntimeError(f"Unsupported buffer backend {backend}")

    def numpy(self):
        self._sync()
        return np.asarray(self)
