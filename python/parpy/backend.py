from .parpy import CompileBackend, CompileOptions
import os
import shutil
import torch

backends = [
    CompileBackend.Cuda,
    CompileBackend.Metal,
    CompileBackend.Triton,
]

def _assert_cuda_is_enabled():
    if not torch.cuda.is_available():
        raise RuntimeError(f"Torch was not built with CUDA support")
    if not shutil.which("nvcc"):
        raise RuntimeError(f"Could not find 'nvcc' in path - it is required to build the generated CUDA C++ code")

def _assert_metal_is_enabled():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Torch was not built with Metal support")
    if os.getenv("METAL_CPP_HEADER_PATH") is None:
        raise RuntimeError("The path to the Metal-cpp headers must be provided " +
                           "to use the Metal backend. The headers are available " +
                           "at https://developer.apple.com/metal/cpp/. The path " +
                           "is specified using the 'METAL_CPP_HEADER_PATH' " +
                           "environment variable.")

def _assert_triton_is_enabled():
    try:
        import triton
    except:
        raise RuntimeError(f"Failed to import Triton")

def is_enabled(backend, verbose=False):
    """
    Determines whether the specified backend is enabled or not on the current
    device. The `verbose` flag can be set to `True` to enable detailed output.
    """
    try:
        if backend == CompileBackend.Cuda:
            _assert_cuda_is_enabled()
        elif backend == CompileBackend.Metal:
            _assert_metal_is_enabled()
        elif backend == CompileBackend.Triton:
            _assert_triton_is_enabled()
        else:
            raise RuntimeError(f"Unsupported backend {backend}")
        return True
    except RuntimeError as e:
        if verbose:
            print(f"Backend {backend} is not enabled: {e}")
        return False

# Determine the list of available backend once, so we do not have to do this
# every time we want to resolve the available backends.
available = [b for b in backends if is_enabled(b, False)]

def _resolve_backend(opts, strict):
    """
    If the provided options specify the backend as `CompileBackend.Auto`, this
    function attempts to resolve it by finding a uniquely supported backend. If
    none or multiple are supported, this function fails, in which case users
    have to explicitly specify the target backend.
    """
    if opts.verbose_backend_resolution:
        [b for b in backends if is_enabled(b, True)]
    if opts.backend == CompileBackend.Auto:
        if len(available) == 0:
            raise RuntimeError("Found no enabled GPU backends. For detailed " +
                               "information on why this is, enable the " +
                               "'verbose_backend_resolution' flag in the " +
                               "compiler options.")
        else:
            opts.backend = available[0]
            return opts
    elif strict and opts.backend not in available:
        raise RuntimeError(f"Specified backend {opts.backend} is not available. For " +
                            "more information, enable the 'verbose_backend_resolution' " +
                            "flag in the compiler options.")
    else:
        return opts

def set_default(backend):
    """
    Sets the provided backend as the default choice for the compiler.
    """
    global available
    if backend is not None and backend in available:
        # If the backend is one of the enabled backends, we put it in the front
        # of the list of available backends. This makes it the default choice
        # in the implementation of _resolve_backend.
        available.remove(backend)
        available.insert(0, backend)
