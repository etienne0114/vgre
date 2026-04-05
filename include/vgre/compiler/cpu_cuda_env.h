#ifndef VGRE_COMPILER_CPU_CUDA_ENV_H
#define VGRE_COMPILER_CPU_CUDA_ENV_H

#include <cmath>
#include <cstdint>

#include "vgre/common/types.h"

namespace vgre_cuda {
    using vgre::dim3;

// Global thread/block indices provided by the wrapper, using an array mapping to bypass JIT TLS corruption
extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();
  
  // Dynamic TLS-based built-ins for stable parallel JIT linkage
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockIdx();
  vgre::dim3* vgre_jit_get_blockDim();
  vgre::dim3* vgre_jit_get_gridDim();
  void** vgre_jit_get_sharedMem();
}

#define threadIdx (*vgre_jit_get_threadIdx())
#define blockIdx  (*vgre_jit_get_blockIdx())
#define blockDim  (*vgre_jit_get_blockDim())
#define gridDim   (*vgre_jit_get_gridDim())
#define sharedMem (*vgre_jit_get_sharedMem())

// Atomic mappings for CPU
template <typename T> inline T atomicAdd(T *addr, T val) { return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicSub(T *addr, T val) { return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicExch(T *addr, T val) { return __atomic_exchange_n(addr, val, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicCAS(T *addr, T comp, T val) { 
    T expected = comp; 
    __atomic_compare_exchange_n(addr, &expected, val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); 
    return expected; 
}

extern "C" void vgre_jit_block_barrier_sync();

// Block barrier: synchronizes threads in a block using the host environment
inline void __syncthreads() { vgre_jit_block_barrier_sync(); }

// CUDA-like math functions for the CPU
inline float __fdividef(float a, float b) { return a / b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }

#define __global__ __attribute__((section("vgre_global")))
#define __device__ __attribute__((section("vgre_device")))
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

// ── Texture/Surface Built-ins ──────────────────────────────────────────────
extern "C" {
  float vgre_tex1D_f32(uint64_t tex, float x);
  float vgre_tex2D_f32(uint64_t tex, float x, float y);
  float vgre_tex3D_f32(uint64_t tex, float x, float y, float z);
  float vgre_tex1Dfetch_f32(uint64_t tex, int x);
  void vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y);
  void vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y);
}

template<typename T>
inline T tex1D(uint64_t tex, float x) {
  return static_cast<T>(vgre_tex1D_f32(tex, x));
}

template<typename T>
inline T tex2D(uint64_t tex, float x, float y) {
  return static_cast<T>(vgre_tex2D_f32(tex, x, y));
}

template<typename T>
inline T tex3D(uint64_t tex, float x, float y, float z) {
  return static_cast<T>(vgre_tex3D_f32(tex, x, y, z));
}

template<typename T>
inline void surf2Dwrite(T val, uint64_t surf, int x, int y) {
  vgre_surf2Dwrite_f32(surf, static_cast<float>(val), x, y);
}

template<typename T>
inline void surf2Dread(T* val, uint64_t surf, int x, int y) {
  float fval;
  vgre_surf2Dread_f32(surf, &fval, x, y);
  *val = static_cast<T>(fval);
}

// ── BFloat16 Support ───────────────────────────────────────────────────────
struct __nv_bfloat16 {
    uint16_t __x;

    __nv_bfloat16() = default;
    __nv_bfloat16(float f) {
        uint32_t i = *reinterpret_cast<uint32_t*>(&f);
        uint32_t bias = 0x7FFF + ((i >> 16) & 1);
        __x = static_cast<uint16_t>((i + bias) >> 16);
    }
    operator float() const {
        uint32_t f = static_cast<uint32_t>(__x) << 16;
        return *reinterpret_cast<float*>(&f);
    }
};

inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
inline float __bfloat162float(__nv_bfloat16 h) { return static_cast<float>(h); }
inline __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16(float(a) * float(b)); }
inline __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16(float(a) + float(b)); }

} // namespace vgre_cuda

using namespace vgre_cuda;

#endif // VGRE_COMPILER_CPU_CUDA_ENV_H
