import os
import parpy
import pathlib
import pytest
import re
import shutil
import subprocess
import torch
import warnings

# Explicitly clear the cache before running tests. This is important, as the
# caching assumes the compiler is fixed. If the compiler is updated, we have to
# clear the cache to ensure it runs.
parpy.clear_cache()

# Use all backends declared in the library
compiler_backends = parpy.backend.backends

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
    if parpy.backend.is_enabled(backend):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return fn()
    else:
        pytest.skip(f"{backend} is not enabled")

def run_if_clusters_are_enabled(backend, fn):
    if backend == parpy.CompileBackend.Cuda and parpy.backend.is_enabled(backend):
        major, minor = torch.cuda.get_device_capability()
        if major < 9:
            pytest.skip("Thread block clusters require CUDA compute capability 9.0 "
                       f"or higher (found {major}.{minor})")
    else:
        pytest.skip(f"Thread block clusters are not supported in the {backend} backend")
    run_if_backend_is_enabled(backend, fn)

def get_cuda_version():
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        r = subprocess.run(["nvcc", "--version"], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to extract CUDA version from nvcc")
        r = re.search(r"release (\d+)\.(\d+)", r.stdout.decode('utf-8'), re.MULTILINE)
        if r:
            return int(r.group(1)), int(r.group(2))
        else:
            raise RuntimeError("Failed to parse CUDA version output from nvcc")
    else:
        raise RuntimeError("Could not find nvcc, needed to determine CUDA backend")

# In this file, we define short-hand functions for specifying the compile
# options to be passed to the JIT compiler. The 'par_opts' function runs with
# the given parallelization specification and (importantly) disables caching to
# prevent bugs in tests.

def par_opts(backend, p):
    opts = parpy.CompileOptions()
    opts.backend = backend
    opts.parallelize = p
    opts.verbose_backend_resolution = True
    opts.debug_print = True
    opts.debug_callbacks = True
    return opts
