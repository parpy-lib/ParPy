from .prickle import CompileBackend, CompileOptions
import prickle.state
import shutil
import torch

backends = [
    CompileBackend.Cuda,
    CompileBackend.Metal,
]

def assert_cuda_is_enabled():
    if not torch.cuda.is_available():
        raise RuntimeError(f"Torch was not built with CUDA support")
    if not shutil.which("nvcc"):
        raise RuntimeError(f"Could not find 'nvcc' in path - it is required to build the generated CUDA C++ code")

def assert_metal_is_enabled():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Torch was not built with Metal support")
    if prickle.state.get_metal_cpp_header_path() is None:
        raise RuntimeError("The path to the Metal-cpp headers must be provided " +
                           "to use the Metal backend. The headers are available " +
                           "at https://developer.apple.com/metal/cpp/. The path " +
                           "is specified using the 'METAL_CPP_HEADER_PATH' " +
                           "environment variable.")

def is_enabled(backend, verbose=False):
    try:
        if backend == CompileBackend.Cuda:
            assert_cuda_is_enabled()
        elif backend == CompileBackend.Metal:
            assert_metal_is_enabled()
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

# If the provided options specify the backend as 'Auto', this function attempts
# to resolve it. The result depends on the number of available backends. If
# exactly one backend is available, the options are updated to use this backend
# and returned. Otherwise, if none or multiple backends are available, this
# function raises an error reporting that the automatic selection failed.
def resolve(opts, strict):
    if opts.verbose_backend_resolution:
        [b for b in backends if is_enabled(b, True)]
    if opts.backend == CompileBackend.Auto:
        if len(available) == 1:
            opts.backend = available[0]
            return opts
        elif len(available) == 0:
            raise RuntimeError("Found no enabled GPU backends. For detailed " +
                               "information on why this is, enable the " +
                               "'verbose_backend_resolution' flag in the " +
                               "compiler options.")
        else:
            raise RuntimeError(f"Found multiple supported GPU backends: {available}. " +
                                "Please explicitly specify which backend to use " +
                                "by setting the 'backend' field of the 'opts' + "
                                "argument to the desired backend.")
    elif strict and opts.backend not in available:
        raise RuntimeError(f"Specified backend {backend} is not available. For " +
                            "more information, enable the 'verbose_backend_resolution' " +
                            "flag in the compiler options.")
    else:
        return opts
