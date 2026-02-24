import parpy
from parpy.types import F64, I32, I64
import pathlib
import torch

backend = parpy.default_backend()

if backend == parpy.CompileBackend.Metal:
    @parpy.external("metal::popcount", parpy.CompileBackend.Metal, parpy.Target.Device, header="<metal_stdlib>")
    def popcount(v: I32) -> I32:
        # Naive implementation in Python
        counts = v & 1
        for i in range(1, 32):
            counts += (v >> i) & 1
        return counts

    N = parpy.types.shape_var()

    @parpy.jit
    def sum_bitcount(
            x: parpy.types.buffer(I32, [N]),
            out: parpy.types.buffer(I32, [parpy.types.literal(1)])):
        out[0] = parpy.reduce.sum(popcount(x[:N]))

    N = 132
    x = torch.randint(0, 2**31-1, (N,), dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int32)
    opts = parpy.par({'N': parpy.threads(32)})
    sum_bitcount(x, out, opts=opts)
    assert out[0] == sum([popcount(v) for v in x])
elif backend == parpy.CompileBackend.Cuda:
    M = parpy.types.shape_var()
    N = parpy.types.shape_var()
    K = parpy.types.shape_var()

    @parpy.external("cublas_gemm_f64", parpy.CompileBackend.Cuda, parpy.Target.Host, header="<cublas_wrapper.h>")
    def cublas_gemm_f64(
            M: I64,
            N: I64,
            K: I64,
            alpha: F64,
            A: parpy.types.buffer(F64, [M, K]),
            B: parpy.types.buffer(F64, [K, N]),
            beta: F64,
            C: parpy.types.buffer(F64, [M, N])):
        pass

    @parpy.jit
    def cublas_gemm(
            alpha: F64,
            A: parpy.types.buffer(F64, [M, K]),
            B: parpy.types.buffer(F64, [K, N]),
            beta: F64,
            C: parpy.types.buffer(F64, [M, N])):
        with parpy.gpu:
            cublas_gemm_f64(M, N, K, alpha, A, B, beta, C)

    M = 1024
    N = 512
    K = 256
    A = torch.randn(M, K, dtype=torch.float64, device='cuda')
    B = torch.randn(K, N, dtype=torch.float64, device='cuda')
    C = torch.empty(M, N, dtype=torch.float64, device='cuda')
    opts = parpy.par({})
    # Add the directory of this file to the include path, so the header file is found.
    opts.includes += [str(pathlib.Path(__file__).parent.resolve())]
    # Add a flag to link the cuBLAS library when compiling.
    opts.extra_flags += ["-lcublas"]
    try:
        cublas_gemm(1.0, A, B, 0.0, C, opts=opts)
        assert torch.allclose(C, A @ B, atol=1e-5)
    except RuntimeError as e:
        if "cublas_v2.h" in str(e):
            print("Skipping because cuBLAS header was not found")
        else:
            raise RuntimeError(e)

print("Test OK")
