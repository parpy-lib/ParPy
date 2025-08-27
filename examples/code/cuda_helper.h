
// Implementation of warp reduction that uses different versions of the code
// depending on the compute capability of the target GPU.
__device__ int32_t warp_sum(int32_t *values) {
  int32_t v = values[threadIdx.x];
#if (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(0xFFFFFFFF, v);
#else
  for (int i = 16; i > 0; i /= 2) {
    v = __shfl_xor_sync(0xFFFFFFFF, v, i);
  }
  return v;
#endif
}
