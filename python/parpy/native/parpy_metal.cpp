#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "parpy_metal.h"

static MTL::Device *device;
static MTL::CommandQueue *cq;
static MTL::CommandBuffer *cb;
static MTL::ComputeCommandEncoder *ce;
static int64_t queue_cap = 0;
static int64_t queue_size = 0;

extern "C" void parpy_init(int64_t queue_capacity) {
  if (device == nullptr) {
    device = MTL::CreateSystemDefaultDevice();
    queue_cap = queue_capacity;
    cq = device->newCommandQueue(queue_cap);
    if (cq == nullptr) {
      fprintf(stderr, "Failed to set up command queue\n");
      exit(1);
    }
  }
}

extern "C" int32_t parpy_sync() {
  parpy_metal::sync();
  return 0;
}

extern "C" MTL::Buffer *parpy_alloc_buffer(int64_t nbytes) {
  MTL::Buffer *b = device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
  if (b == nullptr) {
    parpy_metal::error_message = "Failed to allocate buffer";
    return nullptr;
  }
  return b;
}

extern "C" void *parpy_ptr_buffer(metal_buffer *b) {
  return b->buf->contents();
}

extern "C" metal_buffer *parpy_buffer_wrap_with_offset(MTL::Buffer *b, int64_t offset) {
  metal_buffer *buf = (metal_buffer*)malloc(sizeof(metal_buffer));
  buf->buf = b;
  buf->offset = offset;
  return buf;
}

extern "C" int32_t parpy_buffer_wrap_free(metal_buffer *b) {
  free(b);
  return 0;
}

extern "C" int32_t parpy_memcpy(void *dst, void *src, int64_t nbytes, int64_t k) {
  // If an argument represents device memory, it is a Metal buffer pointer
  // from which we need to extract the actual data pointer. Otherwise, we use
  // the provided pointer immediately. We use 'k' to encode the memory types
  // of the arguments:
  //  0: both host
  //  1: source is in host memory, destination on device
  //  2: source is in device memory, destination on host
  //  3: both device
  if (k & 1) {
    metal_buffer *b = (metal_buffer*)dst;
    dst = (void*)((char*)b->buf->contents() + b->offset);
  }
  if (k & 2) {
    metal_buffer *b = (metal_buffer*)src;
    src = (void*)((char*)b->buf->contents() + b->offset);
  }
  memcpy(dst, src, nbytes);
  return 0;
}

extern "C" int32_t parpy_memcpy_buffer(void *dst, void *src, int64_t nbytes, int64_t k) {
  dst = k & 1 ? ((MTL::Buffer*)dst)->contents() : dst;
  src = k & 2 ? ((MTL::Buffer*)src)->contents() : src;
  memcpy(dst, src, nbytes);
  return 0;
}

extern "C" int32_t parpy_memset(void *ptr, int64_t nbytes, int8_t value) {
  memset(ptr, nbytes, value);
  return 0;
}

extern "C" int32_t parpy_free_buffer(MTL::Buffer *b) {
  b->release();
  return 0;
}

extern "C" const char *parpy_get_error_message() {
  return parpy_metal::error_message;
}

namespace parpy_metal {
  MTL::Library *load_library(const char *lib_str) {
    NS::String *code = NS::String::string(lib_str, NS::ASCIIStringEncoding);
    NS::Error *err;
    MTL::Library *lib = device->newLibrary(code, nullptr, &err);
    if (lib == nullptr) {
      fprintf(stderr, "Error compiling library: %s\n", err->description()->utf8String());
      exit(1);
    }
    return lib;
  }

  MTL::Function *get_fun(MTL::Library *lib, const char *fun_id) {
    NS::String *s = NS::String::string(fun_id, NS::ASCIIStringEncoding);
    MTL::Function *f = lib->newFunction(s);
    if (f == nullptr) {
      fprintf(stderr, "Could not find function %s in library\n", fun_id);
      exit(1);
    }
    return f;
  }

  int32_t alloc(metal_buffer **buf, int64_t nbytes) {
    MTL::Buffer *b = parpy_alloc_buffer(nbytes);
    if (b == nullptr) {
      parpy_metal::error_message = "Buffer allocation failed";
      return 1;
    }
    *buf = parpy_buffer_wrap_with_offset(b, 0);
    if (*buf == nullptr) {
      parpy_metal::error_message = "Buffer allocation failed";
      return 1;
    }
    return 0;
  }

  void free(metal_buffer *b) {
    parpy_free_buffer(b->buf);
    parpy_buffer_wrap_free(b);
  }

  void copy(void *dst, void *src, int64_t nbytes, int64_t k) {
    parpy_memcpy(dst, src, nbytes, k);
  }

  int32_t launch_kernel(
      MTL::Function *kernel,
      std::vector<metal_buffer*> args,
      int64_t block_x, int64_t block_y, int64_t block_z,
      int64_t thread_x, int64_t thread_y, int64_t thread_z,
      uint64_t smem_bytes) {
    if (cb == nullptr || cb->status() != MTL::CommandBufferStatusNotEnqueued) {
      if (cb != nullptr) cb->release();
      cb = cq->commandBuffer();
      if (cb == nullptr) {
        parpy_metal::error_message = "Failed to set up command buffer";
        return 1;
      }
    }

    if (ce == nullptr) {
      ce = cb->computeCommandEncoder();
      if (ce == nullptr) {
        parpy_metal::error_message = "Failed to set up compute command encoder";
        return 1;
      }
    }

    NS::Error *err;
    MTL::ComputePipelineState *state = device->newComputePipelineState(kernel, &err);
    if (state == nullptr) {
      parpy_metal::error_message = "Error setting up compute pipeline state";
      return 1;
    }

    ce->setComputePipelineState(state);

    uint64_t max_smem = device->maxThreadgroupMemoryLength();
    if (smem_bytes > max_smem) {
      snprintf(parpy_metal::err_buf, 1024,
          "Insufficient shared memory. Kernels of the function use up to %llu "
          "bytes of shared memory per block, while the current device only "
          "supports up to %llu bytes.", smem_bytes, max_smem);
      parpy_metal::error_message = err_buf;
      return 1;
    }
    ce->setThreadgroupMemoryLength(smem_bytes, 0);
    for (int i = 0; i < args.size(); i++) {
      ce->setBuffer(args[i]->buf, args[i]->offset, i);
    }

    int simd_width = state->threadExecutionWidth();
    if (simd_width != 32) {
      parpy_metal::error_message = "ParPy only supports target with a SIMD width of 32";
      return 1;
    }
    NS::UInteger maxthreads = state->maxTotalThreadsPerThreadgroup();
    assert(thread_x * thread_y * thread_z <= maxthreads);

    MTL::Size grid_size = MTL::Size::Make(block_x * thread_x, block_y * thread_y, block_z * thread_z);
    MTL::Size block_size = MTL::Size::Make(thread_x, thread_y, thread_z);
    ce->dispatchThreads(grid_size, block_size);
    if (++queue_size == queue_cap) {
      submit_work();
    }
    return 0;
  }

  void submit_work() {
    if (ce != nullptr) {
      ce->endEncoding();
      cb->commit();
      cb->waitUntilScheduled();
      ce->release();
      ce = nullptr;
      queue_size = 0;
    }
  }

  void sync() {
    submit_work();
    if (cb != nullptr) {
      cb->waitUntilCompleted();
      cb->release();
      cb = nullptr;
    }
  }
}
