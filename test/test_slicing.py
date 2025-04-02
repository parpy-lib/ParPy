import parir
import pytest
import torch

torch.manual_seed(1234)

@parir.jit
def add_slices(x, y, out):
    parir.label('N')
    out[:] = x[:] + y[:]

@parir.jit
def add_slices_2d(x, y, out):
    parir.label('N')
    parir.label('M')
    out[:,:] = x[:,:] + y[:,:]

@parir.jit
def mul_discontinuous_2d(x, y, out):
    parir.label('N')
    parir.label('M')
    out[:,:] = x[:,0,:] * y[0,:,:]

@parir.jit
def matmul_slice(a, b, M, N, c):
    for i in range(M):
        parir.label('N')
        for j in range(N):
            parir.label('K')
            c[i,j] = parir.sum(a[i,:] * b[:,j])

@parir.jit
def jacobi_1d(nsteps, A, B):
    for t in range(1, nsteps):
        parir.label('N')
        B[1:-1] = (A[:-2] + A[1:-1] + A[2:]) / 3.0
        parir.label('N')
        A[1:-1] = (B[:-2] + B[1:-1] + B[2:]) / 3.0

@parir.jit
def slice_assignment(x):
    parir.label('N')
    parir.label('M')
    x[:,:] = 0.0

@parir.jit
def slice_multi_dim_sum(x, out):
    with parir.gpu:
        parir.label('N')
        out[0] = parir.sum(x[:,:])

@parir.jit
def slice_multi_dim_interspersed_sum(x, out):
    with parir.gpu:
        parir.label('N')
        out[0] = parir.sum(x[:,0,:,2])

@parir.jit
def slice_assign_to_new_var(x):
    with parir.gpu:
        y = x[:]

@parir.jit
def slice_assign_invalid_dims(x, y):
    with parir.gpu:
        y[:] = parir.sum(x[:,:,:], axis=0)

@parir.jit
def slice_reduce_incompatible_shapes(x, y, out):
    with parir.gpu:
        out[0] = parir.sum(x[:,:] * y[:], axis=0)

@parir.jit
def slice_reduce_in_loop(x, y, N, out):
    parir.label('N')
    for i in range(N):
        parir.label('M')
        out[i] = parir.sum(x[i,:] * y[:,i])

@parir.jit
def slice_invalid_reduce_assignment(x, y, z, N):
    parir.label('N')
    for i in range(N):
        x[i,:] = parir.min(y[i,:] + z[:,i])

@parir.jit
def slice_invalid_dims(x, y, N):
    parir.label('N')
    for i in range(N):
        x[i,:] = y[i,:,:]

@parir.jit
def slice_in_range(x, y):
    parir.label('N')
    for i in range(parir.sum(y[:])):
        x[i] = i

@parir.jit
def temp_slices(x, y, z):
    parir.label('N')
    x[:] = (y[1:] + z[:-1])[1:-1]

def run_slicing_test(compile_only, spec):
    def move_to_device(arg, device):
        if isinstance(arg, torch.Tensor):
            return arg.to(device)
        else:
            return arg
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
    device = torch.device('cpu') if compile_only else torch.device('cuda')
    args = [move_to_device(arg, device) for arg in args]
    p = {
        'N': [parir.threads(32)],
        'M': [parir.threads(32)],
        'K': [parir.threads(32)]
    }
    if compile_only:
        if err:
            with pytest.raises(err) as e_info:
                s = parir.print_compiled(fn, args, p)
            assert e_info.match(err_msg)
        else:
            s = parir.print_compiled(fn, args, p)
            assert len(s) != 0
    else:
        if err:
            with pytest.raises(err) as e_info:
                fn(*args, parallelize=p, cache=False)
            assert e_info.match(err_msg)
        else:
            seq_args = [clone_arg(arg) for arg in args]
            fn(*args, parallelize=p, cache=False)
            fn(*seq_args, seq=True)
            assert torch.allclose(args[-1], seq_args[-1], atol=1e-5)

def tensor(*shapes):
    return torch.randn(shapes, dtype=torch.float64)

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

def slice_assign_invalid_dims(x, y):
    y[:] = parir.sum(x[:,:,:], axis=0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
@pytest.mark.parametrize('spec', fun_specs)
def test_run_slicing(spec):
    run_slicing_test(False, spec)

@pytest.mark.parametrize('spec', fun_specs)
def test_compile_slicing(spec):
    run_slicing_test(True, spec)
