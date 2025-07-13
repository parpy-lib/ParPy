import prickle
import pytest
import torch

from common import *

torch.manual_seed(1234)

@prickle.jit
def add_slices(x, y, out):
    prickle.label('N')
    out[:] = x[:] + y[:]

@prickle.jit
def add_slices_2d(x, y, out):
    prickle.label('N')
    prickle.label('M')
    out[:,:] = x[:,:] + y[:,:]

@prickle.jit
def mul_discontinuous_2d(x, y, out):
    prickle.label('N')
    prickle.label('M')
    out[:,:] = x[:,0,:] * y[0,:,:]

@prickle.jit
def matmul_slice(a, b, M, N, c):
    for i in range(M):
        prickle.label('N')
        for j in range(N):
            prickle.label('K')
            c[i,j] = prickle.sum(a[i,:] * b[:,j])

@prickle.jit
def jacobi_1d(nsteps, A, B):
    for t in range(1, nsteps):
        prickle.label('N')
        B[1:-1] = (A[:-2] + A[1:-1] + A[2:]) / 3.0
        prickle.label('N')
        A[1:-1] = (B[:-2] + B[1:-1] + B[2:]) / 3.0

@prickle.jit
def slice_assignment(x):
    prickle.label('N')
    prickle.label('M')
    x[:,:] = 0.0

@prickle.jit
def slice_multi_dim_sum(x, out):
    with prickle.gpu:
        prickle.label('N')
        out[0] = prickle.sum(x[:,:])

@prickle.jit
def slice_multi_dim_interspersed_sum(x, out):
    with prickle.gpu:
        prickle.label('N')
        out[0] = prickle.sum(x[:,0,:,2])

@prickle.jit
def slice_assign_to_new_var(x):
    with prickle.gpu:
        y = x[:]

@prickle.jit
def slice_assign_invalid_dims(x, y):
    with prickle.gpu:
        y[:] = prickle.sum(x[:,:,:], axis=0)

@prickle.jit
def slice_reduce_incompatible_shapes(x, y, out):
    with prickle.gpu:
        out[0] = prickle.sum(x[:,:] * y[:], axis=0)

@prickle.jit
def slice_reduce_in_loop(x, y, N, out):
    prickle.label('N')
    for i in range(N):
        prickle.label('M')
        out[i] = prickle.sum(x[i,:] * y[:,i])

@prickle.jit
def slice_invalid_reduce_assignment(x, y, z, N):
    prickle.label('N')
    for i in range(N):
        x[i,:] = prickle.min(y[i,:] + z[:,i])

@prickle.jit
def slice_invalid_dims(x, y, N):
    prickle.label('N')
    for i in range(N):
        x[i,:] = y[i,:,:]

@prickle.jit
def slice_in_range(x, y):
    prickle.label('N')
    for i in range(prickle.sum(y[:])):
        x[i] = i

@prickle.jit
def temp_slices(x, y, z):
    prickle.label('N')
    x[:] = (y[1:] + z[:-1])[1:-1]

def run_slicing_test(compile_only, spec, backend):
    def clone_arg(arg):
        if isinstance(arg, torch.Tensor):
            return arg.detach().clone()
        else:
            return arg
    if len(spec) == 4:
        fn, args, err, err_msg = spec
    else:
        fn, args = spec
        err = None
        err_msg = None
    p = {
        'N': prickle.threads(32),
        'M': prickle.threads(32),
        'K': prickle.threads(32)
    }
    if compile_only:
        if err:
            with pytest.raises(err) as e_info:
                s = prickle.print_compiled(fn, args, par_opts(backend, p))
            assert e_info.match(err_msg)
        else:
            s = prickle.print_compiled(fn, args, par_opts(backend, p))
            assert len(s) != 0
    else:
        if err:
            with pytest.raises(err) as e_info:
                fn(*args, opts=par_opts(backend, p))
            assert e_info.match(err_msg)
        else:
            seq_args = [clone_arg(arg) for arg in args]
            fn(*args, opts=par_opts(backend, p))
            fn(*seq_args, opts=seq_opts(backend))
            assert torch.allclose(args[-1], seq_args[-1], atol=1e-5)

def tensor(*shapes):
    return torch.randn(shapes, dtype=torch.float32)

fun_specs = [
    (add_slices, [tensor(10), tensor(10), tensor(10)]),
    (add_slices_2d, [tensor(10, 15), tensor(10, 15), tensor(10, 15)]),
    (mul_discontinuous_2d, [tensor(10, 4, 15), tensor(5, 10, 15), tensor(10, 15)]),
    (matmul_slice, [tensor(10, 15), tensor(15, 20), 10, 20, tensor(10, 20)]),
    (jacobi_1d, [5, tensor(10), tensor(10)]),
    (slice_assignment, [tensor(10, 10)]),
    (slice_multi_dim_sum, [tensor(12, 14), tensor(1)]),
    (slice_multi_dim_interspersed_sum, [tensor(8, 10, 12, 14), tensor(1)]),
    (slice_assign_to_new_var, [tensor(10)], RuntimeError, r".*Slice statements cannot have more slice dim.*"),
    ( slice_assign_invalid_dims, [tensor(10, 12, 14), tensor(12)]
    , TypeError, r".*incompatible shapes.*" ),
    ( slice_reduce_incompatible_shapes, [tensor(10, 12), tensor(10), tensor(1)]
    , TypeError, r".*incompatible shapes.*" ),
    ( slice_reduce_in_loop, [tensor(10, 10), tensor(10, 10), 10, tensor(10)]),
    ( slice_invalid_reduce_assignment
    , [tensor(10, 10), tensor(10, 10), tensor(10, 10), 10]
    , RuntimeError, r"When reducing along all dimensions,.*" ),
    ( slice_invalid_dims, [tensor(10, 10), tensor(10, 10), 10]
    , TypeError, r"Indexing with 3 dimensions on tensor of shape .*" ),
    ( slice_in_range, [tensor(10), tensor(10)], RuntimeError
    , r"Slice expressions are only allowed in assignment.*"),
    ( temp_slices, [tensor(7), tensor(10), tensor(10)], RuntimeError
    , r"Target of slice must be a variable." )
]

@pytest.mark.parametrize('spec', fun_specs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_run_slicing(spec, backend):
    run_if_backend_is_enabled(backend, lambda: run_slicing_test(False, spec, backend))

@pytest.mark.parametrize('spec', fun_specs)
@pytest.mark.parametrize('backend', compiler_backends)
def test_compile_slicing(spec, backend):
    run_if_backend_is_enabled(backend, lambda: run_slicing_test(True, spec, backend))
