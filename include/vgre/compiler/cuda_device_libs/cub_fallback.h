#ifndef VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H
#define VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <type_traits>
#include <vector>
#include <limits>
// Deliberately not <unordered_map>: its bucket implementation drags in an
// internal <list> whose template metaprogramming (allocator_traits-derived
// aliases) some MSVC STL releases express in syntax this pinned clang
// version can't parse at all — a real parse failure, not just a version
// mismatch _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH can paper over. This
// header is force-included into every JIT-compiled kernel (see
// cpu_cuda_env.h), so pulling in <unordered_map> here breaks kernel
// compilation on any such machine even when the kernel never touches CUB.
// The free-list below only ever holds a handful of distinct allocation
// sizes, so a linear scan over a small std::vector is fine.
#if defined(_WIN32)
#  include <malloc.h>   // _aligned_malloc / _aligned_free
#endif

// Warp shuffle builtins (__shfl_sync, __shfl_up_sync, etc.) are defined in
// cpu_cuda_warp.h which is included first via cpu_cuda_env.h.  Do NOT re-declare
// them here — duplicate default arguments are a compile error in C++11 and later.

namespace cub {

// ── WarpReduce ───────────────────────────────────────────────────────────────
// Reduction within a warp using butterfly shuffle.
template<typename T>
class WarpReduce {
    static constexpr unsigned int WARP_SIZE = 32;
public:
    // Inclusive sum of all threads in the warp.
    static inline T Sum(T val) {
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            T other = __shfl_xor_sync(0xFFFFFFFFu, val, static_cast<int>(offset), WARP_SIZE);
            val = val + other;
        }
        return val;
    }

    // Inclusive min of all threads in the warp.
    static inline T Min(T val) {
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            T other = __shfl_xor_sync(0xFFFFFFFFu, val, static_cast<int>(offset), WARP_SIZE);
            val = (other < val) ? other : val;
        }
        return val;
    }

    // Inclusive max of all threads in the warp.
    static inline T Max(T val) {
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            T other = __shfl_xor_sync(0xFFFFFFFFu, val, static_cast<int>(offset), WARP_SIZE);
            val = (other > val) ? other : val;
        }
        return val;
    }

    // Reduce with a custom binary op (in-place, returns result for lane 0).
    template<typename Op>
    static inline T Reduce(T val, Op op) {
        for (unsigned int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            T other = __shfl_xor_sync(0xFFFFFFFFu, val, static_cast<int>(offset), WARP_SIZE);
            val = op(val, other);
        }
        return val;
    }
};

// ── BlockReduce ──────────────────────────────────────────────────────────────
// Two-level reduce: warp-level then single-warp reduction of partials.
template<typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockReduce {
    static constexpr unsigned int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    struct _Storage {
        T warp_results[WARPS];
    };

public:
    using TempStorage = _Storage;

    // Inclusive sum across the entire block.  Uses shared temp storage.
    static inline T Sum(T val, TempStorage& temp) {
        int lane = vgre_jit_get_threadIdx()->x & 31;
        int warp_id = vgre_jit_get_threadIdx()->x >> 5;

        // Step 1: warp-level sum
        T warp_sum = WarpReduce<T>::Sum(val);

        // Step 2: lane 0 of each warp writes partial to shared storage
        if (lane == 0) temp.warp_results[warp_id] = warp_sum;
        vgre_jit_block_barrier_sync();

        // Step 3: warp 0 reduces the partials
        T result = T{};
        if (warp_id == 0) {
            T partial = (lane < static_cast<int>(WARPS)) ? temp.warp_results[lane] : T{};
            result = WarpReduce<T>::Sum(partial);
        }
        vgre_jit_block_barrier_sync();
        return result;
    }

    // Sum without temp storage (synchronous, works for small blocks).
    static inline T Sum(T val) {
        TempStorage temp{};
        return Sum(val, temp);
    }
};

// ── WarpScan ─────────────────────────────────────────────────────────────────
// Prefix scan within a warp using shuffle-up.
template<typename T>
class WarpScan {
    static constexpr unsigned int WARP_SIZE = 32;
public:
    // Inclusive prefix sum.
    static inline T InclusiveSum(T val) {
        for (unsigned int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            T other = __shfl_up_sync(0xFFFFFFFFu, val, offset, WARP_SIZE);
            int lane = vgre_jit_get_threadIdx()->x & 31;
            if (lane >= offset) val = val + other;
        }
        return val;
    }

    // Exclusive prefix sum.
    static inline T ExclusiveSum(T val, T& total) {
        T inclusive = InclusiveSum(val);
        total = __shfl_sync(0xFFFFFFFFu, inclusive, WARP_SIZE - 1, WARP_SIZE);
        T exclusive;
        int lane = vgre_jit_get_threadIdx()->x & 31;
        if (lane == 0) exclusive = T{};
        else exclusive = __shfl_up_sync(0xFFFFFFFFu, inclusive, 1, WARP_SIZE);
        return exclusive;
    }

    // Exclusive sum without total output.
    static inline T ExclusiveSum(T val) {
        T total;
        return ExclusiveSum(val, total);
    }
};

// ── BlockScan ────────────────────────────────────────────────────────────────
// Two-level scan: warp-level inclusive, then warp-level exclusive on warp totals.
template<typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockScan {
    static constexpr unsigned int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr unsigned int WARP_SIZE = 32;
    static constexpr unsigned int WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    struct _Storage {
        T warp_inclusive[WARPS];
        T warp_exclusive[WARPS];
    };

public:
    using TempStorage = _Storage;

    // Inclusive sum across the entire block.
    static inline T InclusiveSum(T val, TempStorage& temp) {
        int lane = vgre_jit_get_threadIdx()->x & 31;
        int warp_id = vgre_jit_get_threadIdx()->x >> 5;

        // Step 1: warp-level inclusive scan
        T warp_incl = WarpScan<T>::InclusiveSum(val);

        // Step 2: lane 0 writes warp total
        if (lane == 0) temp.warp_inclusive[warp_id] = warp_incl;
        vgre_jit_block_barrier_sync();

        // Step 3: warp 0 computes exclusive prefix over warp totals
        if (warp_id == 0) {
            T w_val = (lane < static_cast<int>(WARPS)) ? temp.warp_inclusive[lane] : T{};
            T w_total;
            T w_excl = WarpScan<T>::ExclusiveSum(w_val, w_total);
            if (lane < static_cast<int>(WARPS)) temp.warp_exclusive[lane] = w_excl;
        }
        vgre_jit_block_barrier_sync();

        // Step 4: add warp prefix to each thread's inclusive sum
        return warp_incl + temp.warp_exclusive[warp_id];
    }

    static inline T InclusiveSum(T val) {
        TempStorage temp{};
        return InclusiveSum(val, temp);
    }

    // Exclusive sum across the entire block.
    static inline T ExclusiveSum(T val, TempStorage& temp, T& total) {
        int lane = vgre_jit_get_threadIdx()->x & 31;
        int warp_id = vgre_jit_get_threadIdx()->x >> 5;

        T warp_incl = WarpScan<T>::InclusiveSum(val);
        if (lane == 0) temp.warp_inclusive[warp_id] = warp_incl;
        vgre_jit_block_barrier_sync();

        if (warp_id == 0) {
            T w_val = (lane < static_cast<int>(WARPS)) ? temp.warp_inclusive[lane] : T{};
            T w_total;
            T w_excl = WarpScan<T>::ExclusiveSum(w_val, w_total);
            if (lane < static_cast<int>(WARPS)) temp.warp_exclusive[lane] = w_excl;
            if (lane == 0) total = w_total;
        }
        vgre_jit_block_barrier_sync();

        // exclusive = (warp_incl shifted up by 1) + warp_exclusive
        int tid = vgre_jit_get_threadIdx()->x;
        T result = temp.warp_exclusive[warp_id];
        if (tid > 0) {
            T prev = __shfl_up_sync(0xFFFFFFFFu, warp_incl, 1, WARP_SIZE);
            if (lane != 0) result = result + prev;
        }
        return result;
    }

    static inline T ExclusiveSum(T val) {
        TempStorage temp{};
        T total;
        return ExclusiveSum(val, temp, total);
    }
};

// ── CachingDeviceAllocator — thread-safe size-class free-list pool ─────────────
// Caches freed allocations per byte-size in a per-allocator free list so that
// repeated alloc/free cycles of the same size (common in CUB reduction kernels)
// avoid repeated malloc/free calls.
//
// Alignment: all blocks are aligned to 64 bytes (cache-line) to keep SIMD loads
// valid regardless of whether AVX-512 (64-byte) or AVX2 (32-byte) code is emitted.
//
// Thread safety: a single mutex guards the free lists; CUB typically instantiates
// one allocator per device, so contention is bounded.
template<typename T>
class CachingDeviceAllocator {
public:
    CachingDeviceAllocator() = default;
    ~CachingDeviceAllocator() {
        // Drain all cached blocks on destruction to avoid leaks.
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& entry : pool_) {
            for (void* p : entry.blocks) aligned_free_impl(p);
        }
    }
    CachingDeviceAllocator(const CachingDeviceAllocator&) = delete;
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator&) = delete;

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        const size_t bytes = n * sizeof(T);
        void* raw = nullptr;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& entry : pool_) {
                if (entry.bytes == bytes && !entry.blocks.empty()) {
                    raw = entry.blocks.back();
                    entry.blocks.pop_back();
                    break;
                }
            }
        }
        if (!raw) {
            // Align to 64 bytes (cache line) for optimal SIMD performance.
            // Use platform-specific aligned allocation for Windows/POSIX portability.
#if defined(_WIN32)
            raw = ::_aligned_malloc(bytes, 64);
#elif defined(__APPLE__) || defined(__linux__)
            if (::posix_memalign(&raw, 64, bytes) != 0) raw = nullptr;
#else
            // C++17 std::aligned_alloc requires size to be a multiple of alignment.
            const size_t aligned_bytes = (bytes + 63u) & ~size_t{63u};
            raw = std::aligned_alloc(64, aligned_bytes);
#endif
            if (!raw) throw std::bad_alloc{};
        }
        return static_cast<T*>(raw);
    }

    void deallocate(T* p, size_t n) noexcept {
        if (!p || n == 0) return;
        const size_t bytes = n * sizeof(T);
        // Cap the per-size cache at 16 entries to bound memory usage.
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& entry : pool_) {
            if (entry.bytes == bytes) {
                if (entry.blocks.size() < 16) entry.blocks.push_back(static_cast<void*>(p));
                else aligned_free_impl(p);
                return;
            }
        }
        pool_.push_back(SizeClass{bytes, {static_cast<void*>(p)}});
    }

private:
    static void aligned_free_impl(void* p) noexcept {
#if defined(_WIN32)
        ::_aligned_free(p);
#else
        ::free(p);
#endif
    }

    // Free-list: byte-size -> cached raw blocks of that size. A handful of
    // distinct sizes at most in practice, so linear scan over a vector
    // (instead of std::unordered_map — see the include comment above) costs
    // nothing measurable and avoids a real clang/MSVC-STL parse conflict.
    struct SizeClass {
        size_t bytes;
        std::vector<void*> blocks;
    };
    std::mutex mu_;
    std::vector<SizeClass> pool_;
};

} // namespace cub

#endif // VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H
