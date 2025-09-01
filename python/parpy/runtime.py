from .parpy import CompileBackend
import pathlib

DEFAULT_METAL_COMMAND_QUEUE_SIZE = 64
PARPY_NATIVE_PATH = pathlib.Path(__file__).parent / "native"
PARPY_CUDA_BASE_LIB_PATH = PARPY_NATIVE_PATH / "parpy_cuda_lib.so"
PARPY_METAL_BASE_LIB_PATH = PARPY_NATIVE_PATH / "parpy_metal_lib.so"

libs = {}

def init_library(libpath):
    import ctypes
    lib = ctypes.cdll.LoadLibrary(libpath)
    lib.parpy_sync.argtypes = []
    lib.parpy_sync.restype = ctypes.c_int32
    lib.parpy_alloc_buffer.argtypes = [ctypes.c_int64]
    lib.parpy_alloc_buffer.restype = ctypes.c_void_p
    lib.parpy_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    lib.parpy_memcpy.restype = ctypes.c_int32
    lib.parpy_memset.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int8]
    lib.parpy_memset.restype = ctypes.c_int32
    lib.parpy_free_buffer.argtypes = [ctypes.c_void_p]
    lib.parpy_free_buffer.restype = ctypes.c_int32
    lib.parpy_get_error_message.argtypes = []
    lib.parpy_get_error_message.restype = ctypes.c_char_p
    return lib


def _compile_cuda_runtime_lib():
    import os
    import subprocess
    libpath = PARPY_CUDA_BASE_LIB_PATH
    src_path = libpath.parent / "parpy_cuda.cpp"
    if not libpath.exists() or os.path.getmtime(libpath) < os.path.getmtime(src_path):
        cmd = [
            "nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC", "-x", "cu",
            src_path, "-o", libpath
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            stdout = r.stdout.decode('ascii')
            stderr = r.stderr.decode('ascii')
            raise RuntimeError("Compilation of the CUDA runtime library failed.\n"
                              f"stdout:\n{stdout}\nstderr:\n{stderr}")

    return init_library(libpath)

def _compile_metal_runtime_lib():
    import ctypes
    import os
    import subprocess
    libpath = PARPY_METAL_BASE_LIB_PATH
    src_path = libpath.parent / "parpy_metal.cpp"
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

    lib = init_library(libpath)
    lib.parpy_init.argtypes = [ctypes.c_int64]
    lib.parpy_ptr_buffer.argtypes = [ctypes.c_void_p]
    lib.parpy_ptr_buffer.restype = ctypes.c_void_p
    lib.parpy_init(DEFAULT_METAL_COMMAND_QUEUE_SIZE)
    return lib

def _compile_runtime_lib(backend):
    """
    Compiles the runtime library for a backend unless it has already been
    compiled.
    """
    global libs
    if not backend in libs:
        from parpy.backend import is_enabled
        if not is_enabled(backend):
            raise RuntimeError(f"Cannot build runtime library for {backend} as it is disabled")
        if backend == CompileBackend.Cuda:
            libs[backend] = _compile_cuda_runtime_lib()
            return libs[backend]
        elif backend == CompileBackend.Metal:
            libs[backend] = _compile_metal_runtime_lib()
            return libs[backend]
        else:
            raise RuntimeError(f"Failed to compile runtime library for backend {backend}")
    else:
        return libs[backend]
