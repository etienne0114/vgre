#ifndef VGRE_COMPILER_CPU_CUDA_ENV_H
#define VGRE_COMPILER_CPU_CUDA_ENV_H

#include <cmath>
#include <cstdint>

#include "vgre/common/types.h"
#include "cpu_cuda_intrinsics.h"
#include "cpu_cuda_fp16.h"
#include "cpu_cuda_warp.h"
#include "wmma_emulation.h"

namespace vgre_cuda {
    using vgre::dim3;

// Global thread/block indices provided by the wrapper, using an array mapping to bypass JIT TLS corruption
extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();

  // Thread-block cluster (P3-6): cluster barrier + Distributed Shared Memory.
  void     vgre_jit_cluster_sync();
  void*    vgre_jit_mapa_shared_cluster(void* addr, unsigned dstRank);
  unsigned vgre_jit_cluster_ctarank();
  unsigned vgre_jit_cluster_nctarank();

  // Blackwell tcgen05 Tensor Memory (P3-7): alloc/dealloc + ld/st/cp + mma.
  unsigned vgre_jit_tmem_alloc(int nCols);
  void     vgre_jit_tmem_dealloc(unsigned addr, int nCols);
  void     vgre_jit_tmem_relinquish();
  // NB: these signatures must match include/vgre/runtime/gpu_thread_context.h
  // byte-for-byte — both declare the same extern "C" symbols, and a generated
  // JIT translation unit includes both. uint32_t/uint64_t (not unsigned / unsigned
  // long long: uint64_t != unsigned long long on LP64, which clang flags as a
  // conflicting type and fails every kernel compile).
  void     vgre_jit_tcgen05_ld(uint32_t* dst, uint32_t addr, int lanes, int wordsPerLane);
  void     vgre_jit_tcgen05_st(uint32_t addr, const uint32_t* src, int lanes, int wordsPerLane);
  void     vgre_jit_tcgen05_cp(uint32_t addr, const void* smem, int lanes, int wordsPerLane);
  void     vgre_jit_tcgen05_mma(uint32_t addr, uint64_t descA, uint64_t descB,
                                int M, int N, int K, int kind, int accumulate);

  // Dynamic TLS-based built-ins for stable parallel JIT linkage
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockIdx();
  vgre::dim3* vgre_jit_get_blockDim();
  vgre::dim3* vgre_jit_get_gridDim();
  void** vgre_jit_get_sharedMem();
  void** vgre_jit_get_warp_buffer();  // per-block warp exchange buffer

  // CUDA Dynamic Parallelism (CDP) — device-side child kernel launch
  void* vgre_cdp_get_param_buffer(size_t bytes);
  void* vgre_cdp_get_param_buffer_v2(size_t alignment, size_t bytes);
  void  vgre_cdp_launch_device(void* fn, void* paramBuf,
                                unsigned gx, unsigned gy, unsigned gz,
                                unsigned bx, unsigned by, unsigned bz,
                                size_t sharedMem, unsigned long long streamId);
  struct vgre_cdp_launch_config;
  void  vgre_cdp_launch_device_v2(void* fn, void* paramBuf,
                                   const vgre_cdp_launch_config* config);
  void  vgre_cdp_device_synchronize();
  void  vgre_cdp_drain();
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

// Bitwise atomics — integer types only (matches CUDA).
template <typename T> inline T atomicAnd(T *addr, T val) { return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicOr (T *addr, T val) { return __atomic_fetch_or (addr, val, __ATOMIC_SEQ_CST); }
template <typename T> inline T atomicXor(T *addr, T val) { return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST); }

// Min/Max atomics via compare-and-swap. Works for int/unsigned/long long and,
// like CUDA's float/double atomicMin/Max extensions, any comparable scalar.
template <typename T> inline T atomicMin(T *addr, T val) {
    T old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val < old && !__atomic_compare_exchange_n(addr, &old, val, true,
                            __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {}
    return old;
}
template <typename T> inline T atomicMax(T *addr, T val) {
    T old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val > old && !__atomic_compare_exchange_n(addr, &old, val, true,
                            __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {}
    return old;
}

// atomicInc/atomicDec — unsigned counters with CUDA wrap-around semantics.
//   atomicInc: old' = (old >= val) ? 0 : old + 1
//   atomicDec: old' = (old == 0 || old > val) ? val : old - 1
inline unsigned int atomicInc(unsigned int *addr, unsigned int val) {
    unsigned int old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    unsigned int next;
    do { next = (old >= val) ? 0u : old + 1u; }
    while (!__atomic_compare_exchange_n(addr, &old, next, true,
                __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return old;
}
inline unsigned int atomicDec(unsigned int *addr, unsigned int val) {
    unsigned int old = __atomic_load_n(addr, __ATOMIC_RELAXED);
    unsigned int next;
    do { next = (old == 0u || old > val) ? val : old - 1u; }
    while (!__atomic_compare_exchange_n(addr, &old, next, true,
                __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
    return old;
}

extern "C" void vgre_jit_block_barrier_sync();
extern "C" void vgre_jit_syncgrid();
// Block-wide barrier reduce (op: 0=popcount/count, 1=AND, 2=OR). Returns the
// reduced value to every thread in the block. Backs __syncthreads_count/and/or.
extern "C" int  vgre_jit_block_barrier_reduce(int predicate, int op);

// Block barrier: synchronizes threads in a block using the host environment
inline void __syncthreads() { vgre_jit_block_barrier_sync(); }

// Barrier + block-wide vote (CUDA __syncthreads_count / _and / _or).
inline int __syncthreads_count(int predicate) { return vgre_jit_block_barrier_reduce(predicate, 0); }
inline int __syncthreads_and  (int predicate) { return vgre_jit_block_barrier_reduce(predicate, 1); }
inline int __syncthreads_or   (int predicate) { return vgre_jit_block_barrier_reduce(predicate, 2); }

// ── Cooperative Groups & CUB Fallback ───────────────────────────────────────
// Full cooperative groups API (thread_block, thread_block_tile, coalesced_group,
// grid_group, multi_grid_group) plus CUB primitives (WarpReduce, BlockReduce,
// WarpScan, BlockScan) for kernel portability.
#include "cuda_device_libs/cooperative_groups.h"
#include "cuda_device_libs/cub_fallback.h"

// CUDA-like math functions for the CPU
inline float __fdividef(float a, float b) { return a / b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }

#if defined(_MSC_VER)
// MSVC does not support GCC/Clang __attribute__((section(...))).
// These decorators are used only for static-analysis tagging; the JIT
// parser identifies kernel functions by name convention, so the section
// attribute is not needed at runtime on Windows.
#  define __global__
#  define __device__
#  define __host__
#elif defined(__APPLE__)
// The section attribute is ONLY a discovery marker for the AST parser (which
// uses its own stub, not this header). The JIT path finds kernels by name, so
// the section is vestigial here — and a bare ELF name is rejected by Mach-O
// while a custom __TEXT subsection placed by arm64 JITLink risked mis-aligned
// instruction fetch (SIGBUS) at execution. Drop it on macOS, like Windows.
#  define __global__
#  define __device__
#  define __host__
#else
#  define __global__ __attribute__((section("vgre_global")))
#  define __device__ __attribute__((section("vgre_device")))
#  define __host__
#endif

// For JIT translation, we map __shared__ to a pointer into our block buffer.
// This requires the parser to replace declarations, but for dynamic shared memory (extern),
// we can use this macro. 
#define __shared__ 

struct ptr_unwrapper {
  void* p;
  template<typename T>
  operator T*() const { return static_cast<T*>(p); }
};

// ── VNNI / AMX accelerated matmul — callable from JIT-compiled CUDA kernels ──
// A CUDA kernel that needs an INT8 matrix multiply can call vgre_matmul_int8
// directly. The runtime dispatches to AVX-VNNI (Alder Lake+) or AMX-BF16
// (Sapphire Rapids+) as detected at startup.
extern "C" {
  void  vgre_matmul_int8(const signed char* A, const signed char* B,
                          int* C, int M, int N, int K);
  void  vgre_matmul_bf16(const unsigned short* A, const unsigned short* B,
                          float* C, int M, int N, int K);
  signed char vgre_quant_f32_to_int8(float val, float inv_scale);
  float       vgre_dequant_int8_to_f32(signed char val, float scale);
}

// ── Texture/Surface Built-ins ──────────────────────────────────────────────
extern "C" {
  float    vgre_tex1D_f32(uint64_t tex, float x);
  float    vgre_tex2D_f32(uint64_t tex, float x, float y);
  float    vgre_tex3D_f32(uint64_t tex, float x, float y, float z);
  float    vgre_tex1Dfetch_f32(uint64_t tex, int x);
  void     vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y);
  void     vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y);
  // Per-channel fetch for packed vector textures (float2/float3/float4)
  float    vgre_tex1D_chan_f32(uint64_t tex, float x, unsigned channel);
  float    vgre_tex2D_chan_f32(uint64_t tex, float x, float y, unsigned channel);
  float    vgre_tex3D_chan_f32(uint64_t tex, float x, float y, float z, unsigned channel);
  // Texture metadata queries for PTX txq instructions
  uint32_t vgre_txq_width(uint64_t tex);
  uint32_t vgre_txq_height(uint64_t tex);
  uint32_t vgre_txq_depth(uint64_t tex);
  uint32_t vgre_txq_channels(uint64_t tex);
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
inline __nv_bfloat16 __hmul_bf(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16(float(a)*float(b)); }
inline __nv_bfloat16 __hadd_bf(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16(float(a)+float(b)); }

// ── Dynamic Parallelism (device-side) ────────────────────────────────────────
// Kernels compiled with VGRE can call cudaLaunchDevice to spawn child kernels.
// The child is enqueued via CDPExecutor and runs after the current block.
inline void* cudaGetParameterBuffer(size_t /*alignment*/, size_t size) {
    return vgre_cdp_get_param_buffer(size);
}

inline void* cudaGetParameterBufferV2(size_t alignment, size_t size) {
    return vgre_cdp_get_param_buffer_v2(alignment, size);
}

template<typename Fn>
inline void cudaLaunchDevice(Fn* fn, void* paramBuf,
    vgre_cuda::dim3 gd, vgre_cuda::dim3 bd,
    unsigned sharedMem, unsigned long long stream)
{
    vgre_cdp_launch_device((void*)fn, paramBuf,
        gd.x, gd.y, gd.z,
        bd.x, bd.y, bd.z,
        sharedMem, stream);
}

// V2 launch config (mirrors cudaLaunchConfig)
struct vgre_cuda_launch_config {
    vgre_cuda::dim3 gridDim;
    vgre_cuda::dim3 blockDim;
    size_t dynamicSmemBytes;
    unsigned long long stream;
    int cooperative;
};

template<typename Fn>
inline void cudaLaunchDeviceV2(Fn* fn, void* paramBuf,
                               const vgre_cuda_launch_config* config)
{
    if (!config) return;
    vgre_cdp_launch_device_v2((void*)fn, paramBuf,
        reinterpret_cast<const vgre_cdp_launch_config*>(config));
}

inline void cudaDeviceSynchronize() {
    vgre_cdp_device_synchronize();
}

} // namespace vgre_cuda

using namespace vgre_cuda;

// ── Device-side cuRAND ───────────────────────────────────────────────────────
// Included outside namespace vgre_cuda so its <cmath>/<cstring> transitive
// pulls don't pollute that namespace.  curand_* symbols land in the global
// namespace, matching CUDA's own curand_kernel.h placement.
#include "cuda_device_libs/curand_kernel.h"

#endif // VGRE_COMPILER_CPU_CUDA_ENV_H
