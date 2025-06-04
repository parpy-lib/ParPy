#pragma once

#include "Metal/Metal.hpp"
#include <cstdint>
#include <string>
#include <vector>

// Functions used by the Parir library when initializing, synchronizing with
// running GPU code, and operating on buffers.
extern "C" void parir_metal_init();
extern "C" void parir_metal_sync();
extern "C" MTL::Buffer *parir_metal_alloc_buffer(int64_t);
extern "C" void *parir_metal_ptr_buffer(MTL::Buffer*);
extern "C" void parir_metal_free_buffer(MTL::Buffer*);

// The below functions are to be used in the generated kernel code from C++. We
// wrap these in a namespace to avoid risk of name conflicts.
namespace parir_metal {
  MTL::Library *load_library(const char*);
  MTL::Function *get_fun(MTL::Library*, const char*);
  void launch_kernel(
      MTL::Function*, std::vector<MTL::Buffer*>, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t);
  void submit_work();
}
