#ifndef VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H
#define VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H

#include <cstdint>
#include <type_traits>
#include <limits>

// Re-use the warp shuffle builtins declared in cpu_cuda_warp.h
template<typename T> inline T __shfl_sync(unsigned int mask, T val, int srcLane, int width = 32);
template<typename T> inline T __shfl_up_sync(unsigned int mask, T val, unsigned int delta, int width = 32);

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

// ── CachingDeviceAllocator stub ────────────────────────────────────────────────
// Many CUB-using kernels instantiate this; provide a minimal stub.
template<typename T>
class CachingDeviceAllocator {
public:
    T* allocate(size_t n) { return new T[n]; }
    void deallocate(T* p, size_t) { delete[] p; }
};

} // namespace cub

#endif // VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H
