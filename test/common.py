import os
import prickle
import pytest
import shutil
import torch
import warnings

# Explicitly clear the cache before running tests. This is important, as the
# caching assumes the compiler is fixed. If the compiler is updated, we have to
# clear the cache to ensure it runs.
prickle.clear_cache()

# Use all backends declared in the library
compiler_backends = prickle.backend.backends

# If the Metal backend is available according to PyTorch and the Metal-cpp
# header is missing, we report that the Metal backend is currently disabled and
# what they have to do to enable it.
if torch.backends.mps.is_available() and os.getenv("METAL_CPP_HEADER_PATH") is None:
    msg = "Metal is available on this machine, but the Metal-cpp library " +\
          "could not be found. Please download the Metal-cpp headers and run:\n" +\
          "  export METAL_CPP_HEADER_PATH=/path/to/metal-cpp\n" +\
          "to enable the Metal backend."
    warnings.warn(msg, category=RuntimeWarning)

if torch.cuda.is_available() and not shutil.which("nvcc"):
    msg = "CUDA is available on this machine, but the Nvidia CUDA compiler " +\
          "(nvcc) could not be found. Please ensure 'nvcc' is included in " +\
          "the path to enable the CUDA backend."
    warnings.warn(msg, category=RuntimeWarning)

def run_if_backend_is_enabled(backend, fn):
    if prickle.backend.is_enabled(backend):
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
    opts = prickle.CompileOptions()
    opts.backend = backend
    opts.seq = True
    return opts

def par_opts(backend, p):
    opts = prickle.CompileOptions()
    opts.backend = backend
    opts.parallelize = p
    return opts
