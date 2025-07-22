#pragma once

#include <cstdio>

namespace prickle {
  void cuda_check_error(cudaError_t err) {
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }
}
