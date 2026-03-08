#ifndef VGRE_RUNTIME_GPU_THREAD_CONTEXT_H
#define VGRE_RUNTIME_GPU_THREAD_CONTEXT_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace vgre {
namespace runtime {

// ── GPU Thread Context ───────────────────────────────────────────────────────
// Provides thread-local storage for GPU intrinsic simulations (e.g. __activemask)
// Uses a thread-safe map instead of OS TLS to avoid libgomp destruction crashes.
class GPUThreadContext {
public:
  static void setWarpMask(uint32_t mask);
  static void clearWarpMask();
  static uint32_t getWarpMask();
};

// ── Shared Memory ──────────────────────────────────────────────────────────
// Per-block shared memory buffer. Allocated once per block execution and
// zeroed at the start of each block. Kernels can use typed accessors to
// read/write into the shared pool.
class SharedMemory {
public:
  explicit SharedMemory(size_t sizeBytes) : size_(sizeBytes), buffer_(nullptr) {
    if (sizeBytes > 0) {
      buffer_ = new uint8_t[sizeBytes];
      std::memset(buffer_, 0, sizeBytes);
    }
  }

  ~SharedMemory() { delete[] buffer_; }

  // Non-copyable, movable
  SharedMemory(const SharedMemory &) = delete;
  SharedMemory &operator=(const SharedMemory &) = delete;
  SharedMemory(SharedMemory &&other) noexcept
      : size_(other.size_), buffer_(other.buffer_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
  }

  // Typed accessor — equivalent to `extern __shared__ T sdata[];`
  template <typename T> T *as(size_t byteOffset = 0) {
    return reinterpret_cast<T *>(buffer_ + byteOffset);
  }

  template <typename T> const T *as(size_t byteOffset = 0) const {
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
      std::memset(buffer_, 0, size_);
  }

private:
  size_t size_;
  uint8_t *buffer_;
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
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;
    if (++waiting_ == count_) {
      // Last thread to arrive — release everyone
      waiting_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      // Wait until the generation advances
      cv_.wait(lock, [this, gen] { return generation_ != gen; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_;
  int waiting_;
  int generation_;
};

// ── Atomic Operations ──────────────────────────────────────────────────────
// CPU-backed atomic operations that match CUDA atomic semantics.
// These use GCC/Clang __atomic builtins for hardware-level atomicity.
namespace AtomicOps {

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
    std::memcpy(&newVal, &assumed, sizeof(float));
    newVal += val;
    uint32_t newBits;
    std::memcpy(&newBits, &newVal, sizeof(uint32_t));
    old_val = __sync_val_compare_and_swap(addr_as_uint, assumed, newBits);
  } while (assumed != old_val);
  float result;
  std::memcpy(&result, &old_val, sizeof(float));
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
    std::memcpy(&newVal, &assumed, sizeof(double));
    newVal += val;
    uint64_t newBits;
    std::memcpy(&newBits, &newVal, sizeof(uint64_t));
    old_val = __sync_val_compare_and_swap(addr_as_uint, assumed, newBits);
  } while (assumed != old_val);
  double result;
  std::memcpy(&result, &old_val, sizeof(double));
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
