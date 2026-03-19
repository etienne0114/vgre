#ifndef VGRE_COMPILER_CPU_CUDA_ENV_H
#define VGRE_COMPILER_CPU_CUDA_ENV_H

#include <cmath>
#include <cstdint>
#include "vgre/runtime/gpu_thread_context.h"

namespace vgre_cuda {

struct dim3 {
  uint32_t x, y, z;
  dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1)
      : x(_x), y(_y), z(_z) {}
};

// Global thread/block indices provided by the wrapper, using an array mapping to bypass JIT TLS corruption
extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();
  
  // Dynamic TLS-based built-ins for stable parallel JIT linkage
  vgre_cuda::dim3* vgre_jit_get_threadIdx();
  vgre_cuda::dim3* vgre_jit_get_blockIdx();
  vgre_cuda::dim3* vgre_jit_get_blockDim();
  vgre_cuda::dim3* vgre_jit_get_gridDim();
  void** vgre_jit_get_sharedMem();
}

#define threadIdx (*vgre_jit_get_threadIdx())
#define blockIdx  (*vgre_jit_get_blockIdx())
#define blockDim  (*vgre_jit_get_blockDim())
#define gridDim   (*vgre_jit_get_gridDim())
#define sharedMem (*vgre_jit_get_sharedMem())

// Atomic mappings for CPU
#define atomicAdd(addr, val) __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST)
#define atomicSub(addr, val) __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST)
#define atomicExch(addr, val) __atomic_exchange_n(addr, val, __ATOMIC_SEQ_CST)
#define atomicCAS(addr, comp, val) ({ \
    auto _comp = (comp); \
    __atomic_compare_exchange_n(addr, &_comp, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
    _comp; \
})
// Block barrier: no-op when not executing in a true multi-threaded block
#define __syncthreads() vgre::runtime::GPUThreadContext::blockBarrier()

// CUDA-like math functions for the CPU
inline float __fdividef(float a, float b) { return a / b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }

#define __global__
#define __device__
#define __host__

// For JIT translation, we map __shared__ to a pointer into our block buffer.
// This requires the parser to replace declarations, but for dynamic shared memory (extern),
// we can use this macro. 
#define __shared__ 

struct ptr_unwrapper {
  void* p;
  template<typename T>
  operator T*() const { return static_cast<T*>(p); }
};

} // namespace vgre_cuda

using namespace vgre_cuda;

#endif // VGRE_COMPILER_CPU_CUDA_ENV_H
