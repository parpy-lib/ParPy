#pragma once

#include "Metal/Metal.hpp"
#include <cstdint>
#include <string>
#include <vector>

#define prickle_check_error(e) \
  if (e != 0) return 1;

// Functions used by the Parir library when initializing, synchronizing with
// running GPU code, and operating on buffers.
extern "C" void prickle_init(int64_t);
extern "C" void prickle_sync();
extern "C" MTL::Buffer *prickle_alloc_buffer(int64_t);
extern "C" void *prickle_ptr_buffer(MTL::Buffer*);
extern "C" void prickle_memcpy(void*, void*, int64_t);
extern "C" void prickle_free_buffer(MTL::Buffer*);

// The below functions are to be used in the generated kernel code from C++. We
// wrap these in a namespace to avoid risk of name conflicts.
namespace prickle_metal {
  const char *error_message = nullptr;

  MTL::Library *load_library(const char*);
  MTL::Function *get_fun(MTL::Library*, const char*);
  int32_t alloc(MTL::Buffer**, int64_t);
  void free(MTL::Buffer*);
  void copy(void*, void*, int64_t, int64_t);
  int32_t launch_kernel(
      MTL::Function*, std::vector<MTL::Buffer*>, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t);
  void submit_work();
  void sync();
}

extern "C"
const char *prickle_get_error_message() {
  return prickle_metal::error_message;
}
