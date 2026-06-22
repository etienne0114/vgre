#ifndef VGRE_RUNTIME_GPU_THREAD_CONTEXT_H
#define VGRE_RUNTIME_GPU_THREAD_CONTEXT_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include "vgre/common/platform.h"
#include "vgre/runtime/gpu_cache.h"

namespace vgre {
namespace runtime {

class BlockBarrier;

// ── GPU Thread Context ───────────────────────────────────────────────────────
// Provides thread-local storage for GPU intrinsic authoritative state (e.g. __activemask)
// Uses a thread-safe map instead of OS TLS to avoid libgomp destruction crashes.
class GPUThreadContext {
public:
  static void setWarpMask(uint32_t mask);
  static void clearWarpMask();
  static uint32_t getWarpMask();
  // Optional per-block barrier for __syncthreads
  static void setBlockBarrier(BlockBarrier *barrier);
  static BlockBarrier* getBlockBarrier();
  static void clearBlockBarrier();
  static void blockBarrier();
  static int  blockBarrierReduce(int predicate, int op);
};

// C-friendly wrappers exported for JIT symbol resolution.
extern "C" VGRE_PUBLIC_API void vgre_jit_set_block_barrier(void *barrier);
extern "C" VGRE_PUBLIC_API void vgre_jit_clear_block_barrier();
extern "C" VGRE_PUBLIC_API void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg);
// Barrier + block-wide vote (backs __syncthreads_count/_and/_or).
extern "C" VGRE_PUBLIC_API int  vgre_jit_block_barrier_reduce(int predicate, int op);

// ── Thread-block cluster context (P3-6) ──────────────────────────────────────
// The executor's clustered path installs the current CTA's Cluster + cluster-rank
// on each worker thread before running the kernel; the JIT helpers below read
// that context. With no cluster installed they degenerate to a size-1 self
// cluster (sync is a no-op, mapa returns the address unchanged) — the correct
// behavior for a non-clustered launch.
extern "C" VGRE_PUBLIC_API void     vgre_jit_set_cluster(void* cluster, unsigned rank);
extern "C" VGRE_PUBLIC_API void     vgre_jit_clear_cluster();
extern "C" VGRE_PUBLIC_API void     vgre_jit_cluster_sync();
extern "C" VGRE_PUBLIC_API void*    vgre_jit_mapa_shared_cluster(void* addr, unsigned dstRank);
extern "C" VGRE_PUBLIC_API unsigned vgre_jit_cluster_ctarank();
extern "C" VGRE_PUBLIC_API unsigned vgre_jit_cluster_nctarank();

// ── Blackwell tcgen05 Tensor Memory (TMEM) data path (P3-7) ──────────────────
// A per-CTA TMEM lives on each worker thread (lazily created, reset per launch).
// tcgen05.alloc/.dealloc manage columns; the MMA accumulates into a TMEM address;
// tcgen05.ld/.st/.cp move fragments between registers/SMEM and TMEM.
extern "C" VGRE_PUBLIC_API uint32_t vgre_jit_tmem_alloc(int nCols);
extern "C" VGRE_PUBLIC_API void     vgre_jit_tmem_dealloc(uint32_t addr, int nCols);
extern "C" VGRE_PUBLIC_API void     vgre_jit_tmem_relinquish();
extern "C" VGRE_PUBLIC_API void     vgre_jit_tcgen05_ld(uint32_t* dst, uint32_t addr, int lanes, int wordsPerLane);
extern "C" VGRE_PUBLIC_API void     vgre_jit_tcgen05_st(uint32_t addr, const uint32_t* src, int lanes, int wordsPerLane);
extern "C" VGRE_PUBLIC_API void     vgre_jit_tcgen05_cp(uint32_t addr, const void* smem, int lanes, int wordsPerLane);
// MMA into a TMEM accumulator: kind 0=bf16 1=f16 2=tf32 3=e4m3 4=e5m2 5=e4m3e5m2.
extern "C" VGRE_PUBLIC_API void     vgre_jit_tcgen05_mma(uint32_t addr, uint64_t descA, uint64_t descB,
                                                         int M, int N, int K, int kind, int accumulate);

// ── Shared Memory ──────────────────────────────────────────────────────────
// Per-block shared memory buffer. Allocated once per block execution and
// zeroed at the start of each block. Kernels can use typed accessors to
// read/write into the shared pool.
class SharedMemory {
public:
  explicit SharedMemory(size_t sizeBytes) : size_(sizeBytes), buffer_(nullptr) {
    if (sizeBytes > 0) {
      buffer_ = new uint8_t[sizeBytes];
      memset(buffer_, 0, sizeBytes);
    }
  }

  ~SharedMemory() { delete[] buffer_; }

  // Non-copyable, movable
  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;
  SharedMemory(SharedMemory &&other) noexcept
      : size_(other.size_), buffer_(other.buffer_),
        conflicts_(other.conflicts_.load()) {
    other.buffer_ = nullptr;
    other.size_ = 0;
  }

  // Typed accessor — equivalent to `extern __shared__ T sdata[];`
  template <typename T> T *as(size_t byteOffset = 0) {
    trackAccess(byteOffset, sizeof(T), /*isWrite=*/true);
    return reinterpret_cast<T *>(buffer_ + byteOffset);
  }

  template <typename T> const T *as(size_t byteOffset = 0) const {
    const_cast<SharedMemory*>(this)->trackAccess(byteOffset, sizeof(T), /*isWrite=*/false);
    return reinterpret_cast<const T *>(buffer_ + byteOffset);
  }

  void *raw() { return buffer_; }
  const void *raw() const { return buffer_; }
  size_t size() const { return size_; }

  void ensureCapacity(size_t sizeBytes) {
    if (sizeBytes > size_) {
      delete[] buffer_;
      buffer_ = new uint8_t[sizeBytes];
      size_ = sizeBytes;
    }
    reset();
  }

  void reset() {
    if (buffer_ && size_ > 0)
      memset(buffer_, 0, size_);
    conflicts_ = 0;
  }

  uint64_t getConflictCount() const { return conflicts_; }

private:
  void trackAccess(size_t offset, size_t size, bool isWrite = false) {
    // ── Bank Conflict Profiling (warp model) ───────────────────────────────
    // 4-byte banks, 32 banks per warp. Conflict = two accesses to the same
    // bank at different addresses in the same warp cycle.
    static thread_local uint32_t lastBank   = 0xFFFFFFFF;
    static thread_local uint32_t lastOffset = 0xFFFFFFFF;
    uint32_t bank = (static_cast<uint32_t>(offset) / 4) % 32;
    if (bank == lastBank && static_cast<uint32_t>(offset) != lastOffset)
        conflicts_++;
    lastBank   = bank;
    lastOffset = static_cast<uint32_t>(offset);

    // ── L1/L2 Cache Modeling ───────────────────────────────────────────────
    // Shared memory is architecturally L1-resident on GPU hardware — it never
    // goes to L2. Model it as an L1 hit for statistics purposes (hit rate = 1).
    // Global memory accesses go through L2; those are modeled in the executor.
    // Here we just record the L1 hit for the shared memory access.
    vgre::runtime::GPUCacheL2::instance().access(
        reinterpret_cast<uint64_t>(buffer_) + offset, size, isWrite);
  }

  size_t size_;
  uint8_t *buffer_;
  std::atomic<uint64_t> conflicts_{0};
};

// ── Block Barrier ──────────────────────────────────────────────────────────
// Emulates __syncthreads() for multi-threaded block execution.
// Uses a reusable counting barrier (C++17 compatible).
// For single-threaded per-block execution (e.g., sequential thread loops
// within a pattern matcher), syncthreads() is a no-op since threads are
// processed one at a time.
class BlockBarrier {
public:
  explicit BlockBarrier(int threadCount)
      : count_(threadCount), waiting_(0), generation_(0) {}

  // Wait for all threads in the block to reach this point.
  // Safe for reuse across multiple syncthreads() calls.
  void arrive_and_wait() {
    // Correct generation-based barrier using mutex + condition variable.
    //
    // The previous implementation used a spin + atomic-load on a non-atomic
    // `waiting_` field, which can allow threads to "run ahead" before the last
    // arrival resets `waiting_`, causing `waiting_` to overshoot `count_`.
    // That leads to deadlock (all threads wait for a generation change that
    // never happens).
    std::unique_lock<std::mutex> lock(mutex_);
    const int gen = generation_;
    if (++waiting_ == count_) {
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return generation_ != gen; });
    }
  }

  // Barrier + block-wide reduction of `predicate` across all threads.
  // op: 0 = count of nonzero predicates (CUDA __syncthreads_count)
  //     1 = AND  (nonzero iff every predicate is nonzero — __syncthreads_and)
  //     2 = OR   (nonzero iff any predicate is nonzero  — __syncthreads_or)
  // Every thread receives the same reduced result. Reuses the generation barrier
  // so it is safe to interleave with arrive_and_wait() across launches.
  int arrive_and_reduce(int predicate, int op) {
    std::unique_lock<std::mutex> lock(mutex_);
    const int gen = generation_;
    if (waiting_ == 0) reduceAccum_ = (op == 1) ? 1 : 0;   // AND identity is 1
    const int p = (predicate != 0) ? 1 : 0;
    switch (op) {
      case 1:  reduceAccum_ = reduceAccum_ && p; break;     // AND
      case 2:  reduceAccum_ = reduceAccum_ || p; break;     // OR
      default: reduceAccum_ += p;                break;     // count
    }
    if (++waiting_ == count_) {
      reduceResult_ = reduceAccum_;
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
      return reduceResult_;
    }
    cv_.wait(lock, [this, gen] { return generation_ != gen; });
    return reduceResult_;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
  int waiting_;
  int generation_;
  int reduceAccum_ = 0;
  int reduceResult_ = 0;
};

// ── Atomic Operations ──────────────────────────────────────────────────────
// CPU-backed atomic operations that match CUDA atomic semantics.
// These use GCC/Clang __atomic builtins for hardware-level atomicity.
namespace AtomicOps {

#if defined(_MSC_VER)

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicAdd(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->fetch_add(
      val, std::memory_order_seq_cst);
}

inline float atomicAdd(float *addr, float val) {
  static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
  auto *bits = reinterpret_cast<std::atomic<uint32_t> *>(addr);
  uint32_t oldBits = bits->load(std::memory_order_relaxed);
  while (true) {
    float oldVal;
    memcpy(&oldVal, &oldBits, sizeof(float));
    float newVal = oldVal + val;
    uint32_t newBits;
    memcpy(&newBits, &newVal, sizeof(uint32_t));
    if (bits->compare_exchange_weak(oldBits, newBits, std::memory_order_seq_cst,
                                    std::memory_order_relaxed)) {
      return oldVal;
    }
  }
}

inline double atomicAdd(double *addr, double val) {
  static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
  auto *bits = reinterpret_cast<std::atomic<uint64_t> *>(addr);
  uint64_t oldBits = bits->load(std::memory_order_relaxed);
  while (true) {
    double oldVal;
    memcpy(&oldVal, &oldBits, sizeof(double));
    double newVal = oldVal + val;
    uint64_t newBits;
    memcpy(&newBits, &newVal, sizeof(uint64_t));
    if (bits->compare_exchange_weak(oldBits, newBits, std::memory_order_seq_cst,
                                    std::memory_order_relaxed)) {
      return oldVal;
    }
  }
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicSub(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->fetch_sub(
      val, std::memory_order_seq_cst);
}

template <typename T> inline T atomicMin(T *addr, T val) {
  auto *a = reinterpret_cast<std::atomic<T> *>(addr);
  T oldVal = a->load(std::memory_order_relaxed);
  while (val < oldVal) {
    if (a->compare_exchange_weak(oldVal, val, std::memory_order_seq_cst,
                                 std::memory_order_relaxed)) {
      return oldVal;
    }
  }
  return oldVal;
}

template <typename T> inline T atomicMax(T *addr, T val) {
  auto *a = reinterpret_cast<std::atomic<T> *>(addr);
  T oldVal = a->load(std::memory_order_relaxed);
  while (val > oldVal) {
    if (a->compare_exchange_weak(oldVal, val, std::memory_order_seq_cst,
                                 std::memory_order_relaxed)) {
      return oldVal;
    }
  }
  return oldVal;
}

template <typename T> inline T atomicCAS(T *addr, T compare, T val) {
  auto *a = reinterpret_cast<std::atomic<T> *>(addr);
  a->compare_exchange_strong(compare, val, std::memory_order_seq_cst,
                             std::memory_order_seq_cst);
  return compare;
}

template <typename T> inline T atomicExch(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->exchange(
      val, std::memory_order_seq_cst);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicOr(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->fetch_or(
      val, std::memory_order_seq_cst);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicAnd(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->fetch_and(
      val, std::memory_order_seq_cst);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicXor(T *addr, T val) {
  return reinterpret_cast<std::atomic<T> *>(addr)->fetch_xor(
      val, std::memory_order_seq_cst);
}

#else

// atomicAdd: returns old value at *addr, stores (old + val)
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicAdd(T *addr, T val) {
  return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

// Float specialization via CAS loop
inline float atomicAdd(float *addr, float val) {
  static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
  uint32_t *addr_as_uint = reinterpret_cast<uint32_t *>(addr);
  uint32_t old_val = __atomic_load_n(addr_as_uint, __ATOMIC_RELAXED);
  uint32_t assumed;
  do {
    assumed = old_val;
    float newVal;
    memcpy(&newVal, &assumed, sizeof(float));
    newVal += val;
    uint32_t newBits;
    memcpy(&newBits, &newVal, sizeof(uint32_t));
    old_val = __sync_val_compare_and_swap(addr_as_uint, assumed, newBits);
  } while (assumed != old_val);
  float result;
  memcpy(&result, &old_val, sizeof(float));
  return result;
}

// Double specialization via CAS loop
inline double atomicAdd(double *addr, double val) {
  static_assert(sizeof(double) == sizeof(uint64_t), "double must be 64-bit");
  uint64_t *addr_as_uint = reinterpret_cast<uint64_t *>(addr);
  uint64_t old_val = __atomic_load_n(addr_as_uint, __ATOMIC_RELAXED);
  uint64_t assumed;
  do {
    assumed = old_val;
    double newVal;
    memcpy(&newVal, &assumed, sizeof(double));
    newVal += val;
    uint64_t newBits;
    memcpy(&newBits, &newVal, sizeof(uint64_t));
    old_val = __sync_val_compare_and_swap(addr_as_uint, assumed, newBits);
  } while (assumed != old_val);
  double result;
  memcpy(&result, &old_val, sizeof(double));
  return result;
}

// atomicSub: returns old value at *addr, stores (old - val)
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicSub(T *addr, T val) {
  return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

// atomicMin: returns old value, stores min(old, val)
template <typename T> inline T atomicMin(T *addr, T val) {
  T old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);
  while (val < old_val) {
    if (__atomic_compare_exchange_n(addr, &old_val, val, true, __ATOMIC_SEQ_CST,
                                    __ATOMIC_RELAXED))
      return old_val;
  }
  return old_val;
}

// atomicMax: returns old value, stores max(old, val)
template <typename T> inline T atomicMax(T *addr, T val) {
  T old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);
  while (val > old_val) {
    if (__atomic_compare_exchange_n(addr, &old_val, val, true, __ATOMIC_SEQ_CST,
                                    __ATOMIC_RELAXED))
      return old_val;
  }
  return old_val;
}

// atomicCAS: if (*addr == compare) { *addr = val; } returns old value
template <typename T> inline T atomicCAS(T *addr, T compare, T val) {
  __atomic_compare_exchange_n(addr, &compare, val, false, __ATOMIC_SEQ_CST,
                              __ATOMIC_SEQ_CST);
  return compare; // compare holds the original value after the CAS
}

// atomicExch: stores val at *addr, returns old value
template <typename T> inline T atomicExch(T *addr, T val) {
  return __atomic_exchange_n(addr, val, __ATOMIC_SEQ_CST);
}

// atomicOr: bitwise OR
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicOr(T *addr, T val) {
  return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

// atomicAnd: bitwise AND
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicAnd(T *addr, T val) {
  return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

// atomicXor: bitwise XOR
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
atomicXor(T *addr, T val) {
  return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

#endif

} // namespace AtomicOps

// ── Convenience syncthreads ────────────────────────────────────────────────
// For pattern matchers that process threads sequentially within a block,
// this is a no-op. For parallel thread execution, pass a BlockBarrier.
inline void syncthreads() {
  // No-op in sequential per-block execution model.
  // When threads within a block are executed sequentially (one at a time),
  // implicit ordering guarantees __syncthreads() semantics.
}

inline void syncthreads(BlockBarrier &barrier) { barrier.arrive_and_wait(); }

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_GPU_THREAD_CONTEXT_H
