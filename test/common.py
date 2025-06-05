import os
import parir
import pytest
import torch

# We produce lists of the backends that should be tested. The compilation
# backends include the backends for which the target machine should support
# end-to-end compilation. The codegen backends include all compiler backends.

metal_cpp_header_path = os.getenv("METAL_CPP_HEADER_PATH")
if metal_cpp_header_path is not None:
    parir.set_metal_cpp_header_path(metal_cpp_header_path)

compiler_backends = [
    parir.CompileBackend.Cuda,
    parir.CompileBackend.Metal,
]

def compiler_backend_is_enabled(backend):
    if backend == parir.CompileBackend.Cuda:
        return torch.cuda.is_available()
    elif backend == parir.CompileBackend.Metal:
        return torch.mps.is_available() and metal_cpp_header_path is not None

def run_if_backend_is_enabled(backend, fn):
    if compiler_backend_is_enabled(backend):
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
    opts.cache = False
    return opts
