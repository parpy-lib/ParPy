import os
import parir
import pytest
import torch

# Explicitly clear the cache before running tests. This is important, as the
# caching assumes the compiler is fixed. If the compiler is updated, we have to
# clear the cache to ensure it runs.
parir.clear_cache()

# Use all backends declared in the library
compiler_backends = parir.backend.backends

# If the Metal backend is determined to be available by Torch, but the
# Metal-cpp header is missing, we report an error to alert the user that they
# need to provide this path (otherwise all compilation tests for Metal would be
# skipped).
if torch.backends.mps.is_available() and parir.get_metal_cpp_header_path() is None:
    raise RuntimeError(f"The Metal backend is available, but the path to the " +
                        "Metal-cpp header was not set using the " +
                        "METAL_CPP_HEADER_PATH environment variable.")

def run_if_backend_is_enabled(backend, fn):
    if parir.backend.is_enabled(backend):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return fn()
    else:
        pytest.skip(f"{backend} is not enabled")

# In this file, we define short-hand functions for specifying the compile
# options to be passed to the JIT compiler. The 'seq_opts' function ensures the
# code runs sequentially in the Python interpreter, while the 'par_opts'
# function runs with the given parallelization specification and (importantly)
# disables caching to prevent bugs in tests.

def seq_opts(backend):
    opts = parir.CompileOptions()
    opts.backend = backend
    opts.seq = True
    return opts

def par_opts(backend, p):
    opts = parir.CompileOptions()
    opts.backend = backend
    opts.parallelize = p
    return opts
