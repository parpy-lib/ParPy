#include "parpy_cuda.h"

extern "C" void parpy_init(int64_t _x) {
  return;
}

extern "C" int32_t parpy_sync() {
  return cudaDeviceSynchronize();
}

extern "C" void *parpy_alloc_buffer(int64_t nbytes) {
  void *ptr;
  if (cudaMallocAsync(&ptr, nbytes, 0) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

extern "C" int32_t parpy_memcpy(void *dst, void *src, int64_t nbytes, int64_t k) {
  cudaError_t err;
  if (k == 0) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToHost, 0);
  } else if (k == 1) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, 0);
  } else if (k == 2) {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, 0);
  } else {
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, 0);
  }
  return (int32_t)err;
}

extern "C" int32_t parpy_memset(void *ptr, int64_t nbytes, int8_t value) {
  return (int32_t)cudaMemset(ptr, value, nbytes);
}

extern "C" int32_t parpy_free_buffer(void *p) {
  return (int32_t)cudaFreeAsync(p, 0);
}
