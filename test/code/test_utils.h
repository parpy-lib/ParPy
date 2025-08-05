#pragma once

__device__ float clamp(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// This function cannot be used as an external because it is not declared with
// the device attribute (i.e., it is not accessible from the GPU).
float clamp_non_device(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

__device__ double sum_row_ext(double *p, int64_t n) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += p[i];
  }
  return s;
}
