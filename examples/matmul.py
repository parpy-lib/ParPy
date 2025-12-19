import numpy as np
import parpy

@parpy.jit
def matmul(A, B, C, M, N):
    parpy.label('M')
    for i in range(M):
        parpy.label('N')
        for j in range(N):
            parpy.label('K')
            C[i, j] = parpy.reduce.sum(A[i, :] * B[:, j])

M = 128
N = 32
K = 64
backend = parpy.default_backend()
A = np.random.randn(M, K).astype(np.float32)
A = parpy.buffer.from_array(A, backend)
B = np.random.randn(K, N).astype(np.float32)
B = parpy.buffer.from_array(B, backend)
C = parpy.buffer.zeros((M, N), B.dtype, backend)
p = {
    'M': parpy.threads(M),
    'N': parpy.threads(N),
    'K': parpy.threads(32),
}
opts = parpy.par(p)
matmul(A, B, C, M, N, opts=opts)
assert np.allclose(C.numpy(), A.numpy() @ B.numpy(), atol=1e-3)

T = parpy.types.type_var()
M = parpy.types.shape_var()
N = parpy.types.shape_var()
K = parpy.types.shape_var()

@parpy.jit
def matmul2(
        A: parpy.types.buffer(T, [M, K]),
        B: parpy.types.buffer(T, [K, N]),
        C: parpy.types.buffer(T, [M, N])):
    for i in range(M):
        for j in range(N):
            C[i, j] = parpy.reduce.sum(A[i, :K] * B[:K, j])

C = parpy.buffer.zeros_like(C)
matmul2(A, B, C, opts=opts)
assert np.allclose(C.numpy(), A.numpy() @ B.numpy(), atol=1e-3)
