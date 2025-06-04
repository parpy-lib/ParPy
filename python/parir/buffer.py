import ctypes
import pathlib
import shutil
import subprocess
from operator import mul
from functools import reduce
from .parir import CompileBackend
import parir.state

DEFAULT_METAL_COMMAND_QUEUE_SIZE = 64

metal_lib = None

def try_load_metal_base_lib():
    global metal_lib
    if metal_lib is None:
        # If the library file already exists, and it is more recent than the
        # corresponding source file, we could skip the build to save time.
        libpath = pathlib.Path(f"{__file__}").parent / "native" / "parir_metal_lib.so"
        if not shutil.which("clang++"):
            return None
        src_path = libpath.parent / "parir_metal.cpp"
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

int8 = 0
int16 = 1
int32 = 2
int64 = 3
float16 = 4
float32 = 5
float64 = 6
typemap = {
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
    def __init__(self, buf, shape, dtype, backend):
        self.buf = buf
        self.shape = shape
        self.dtype = dtype
        self.backend = backend
        if self.backend == CompileBackend.Cuda:
            cuda_intf = to_array_interface(self.buf, self.dtype, shape)
            setattr(self, "__cuda_array_interface__", cuda_arr_intf)
        elif self.backend == CompileBackend.Metal:
            try_load_metal_base_lib()
            self.ptr = metal_lib.parir_ptr_buffer(buf)
            arr_intf = to_array_interface(self.ptr, self.dtype, shape)
            setattr(self, "__array_interface__", arr_intf)
        else:
            raise RuntimeError(f"Unsupported compiler backend {backend}")

    def __del__(self):
        if self.backend == CompileBackend.Metal and self.buf is not None:
            metal_lib.parir_free_buffer(self.buf)

    def sync(self):
        if self.backend == CompileBackend.Metal:
            try_load_metal_base_lib()
            metal_lib.sync()

    def from_array_cuda(t):
        if hasattr(t, "__cuda_array_interface__"):
            shape, dtype, data_ptr = check_array_interface(getattr(t, "__cuda_array_interface__"))
            return Buffer(data_ptr, shape, dtype, CompileBackend.Cuda)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to CUDA buffer")

    def from_array_metal(t):
        if hasattr(t, "__array_interface__"):
            try_load_metal_base_lib()
            shape, dtype, data_ptr = check_array_interface(getattr(t, "__array_interface__"))
            nbytes = reduce(mul, shape, 1) * dtype.size()
            buf = metal_lib.parir_alloc_buffer(nbytes)
            ptr = metal_lib.parir_ptr_buffer(buf)
            metal_lib.parir_memcpy(ptr, data_ptr, nbytes)
            return Buffer(buf, shape, dtype, CompileBackend.Metal)
        else:
            raise RuntimeError(f"Cannot convert argument {t} to Metal buffer")

    def numpy(self):
        self._sync()
        return np.asarray(self)
