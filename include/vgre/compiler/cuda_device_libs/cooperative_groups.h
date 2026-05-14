#ifndef VGRE_COMPILER_CUDA_DEVICE_LIBS_COOPERATIVE_GROUPS_H
#define VGRE_COMPILER_CUDA_DEVICE_LIBS_COOPERATIVE_GROUPS_H

#include <cstdint>
#include <type_traits>

// Forward-declare JIT runtime functions needed by cooperative groups.
extern "C" {
  int   vgre_jit_get_thread_id();
  void  vgre_jit_syncgrid();
  void  vgre_jit_block_barrier_sync();
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockIdx();
  vgre::dim3* vgre_jit_get_blockDim();
  vgre::dim3* vgre_jit_get_gridDim();
}

namespace cooperative_groups {

// ── Base class ───────────────────────────────────────────────────────────────
class thread_group {
public:
    virtual ~thread_group() = default;
    virtual void sync() const = 0;
    virtual unsigned int size() const = 0;
    virtual unsigned int thread_rank() const = 0;
    bool is_valid() const { return true; }
};

// ── thread_block (maps to a CUDA block) ────────────────────────────────────
class thread_block : public thread_group {
public:
    inline void sync() const override { vgre_jit_block_barrier_sync(); }

    inline unsigned int size() const override {
        auto* bd = vgre_jit_get_blockDim();
        return bd->x * bd->y * bd->z;
    }

    inline unsigned int thread_rank() const override {
        auto* ti = vgre_jit_get_threadIdx();
        auto* bd = vgre_jit_get_blockDim();
        return (ti->z * bd->y + ti->y) * bd->x + ti->x;
    }

    inline dim3 group_index() const {
        auto* bi = vgre_jit_get_blockIdx();
        return dim3(bi->x, bi->y, bi->z);
    }

    inline dim3 thread_index() const {
        auto* ti = vgre_jit_get_threadIdx();
        return dim3(ti->x, ti->y, ti->z);
    }

    inline unsigned int num_threads() const { return size(); }
    inline unsigned int rank() const { return thread_rank(); }
};

inline thread_block this_thread_block() { return thread_block{}; }

// ── coalesced_group (contiguous threads within a warp) ──────────────────────
// In VGRE's serial CPU model, all threads are effectively coalesced.
class coalesced_group : public thread_group {
    unsigned int m_size;
    unsigned int m_rank;
public:
    explicit coalesced_group(unsigned int sz = 32, unsigned int rk = 0)
        : m_size(sz), m_rank(rk) {}

    inline void sync() const override { vgre_jit_block_barrier_sync(); }
    inline unsigned int size() const override { return m_size; }
    inline unsigned int thread_rank() const override { return m_rank; }

    template<typename T>
    inline T shfl(T val, int srcLane) const {
        return __shfl_sync(0xFFFFFFFFu, val, srcLane, static_cast<int>(m_size));
    }

    template<typename T>
    inline T shfl_down(T val, unsigned int delta) const {
        return __shfl_down_sync(0xFFFFFFFFu, val, delta, static_cast<int>(m_size));
    }

    template<typename T>
    inline T shfl_up(T val, unsigned int delta) const {
        return __shfl_up_sync(0xFFFFFFFFu, val, delta, static_cast<int>(m_size));
    }

    template<typename T>
    inline T shfl_xor(T val, int laneMask) const {
        return __shfl_xor_sync(0xFFFFFFFFu, val, laneMask, static_cast<int>(m_size));
    }
};

inline coalesced_group coalesced_threads() {
    return coalesced_group{32u, static_cast<unsigned int>(vgre_jit_get_threadIdx()->x & 31)};
}

// ── thread_block_tile<N> (tiled sub-group inside a block) ──────────────────
template<unsigned int TileSize>
class thread_block_tile : public thread_group {
    static_assert(TileSize == 4 || TileSize == 8 || TileSize == 16 || TileSize == 32,
                  "TileSize must be 4, 8, 16, or 32");
public:
    inline void sync() const override { vgre_jit_block_barrier_sync(); }

    inline unsigned int size() const override { return TileSize; }

    inline unsigned int thread_rank() const override {
        int tid = static_cast<int>(vgre_jit_get_threadIdx()->x);
        return static_cast<unsigned int>(tid % static_cast<int>(TileSize));
    }

    template<typename T>
    inline T shfl(T val, int srcLane) const {
        return __shfl_sync(0xFFFFFFFFu, val, srcLane, static_cast<int>(TileSize));
    }

    template<typename T>
    inline T shfl_down(T val, unsigned int delta) const {
        return __shfl_down_sync(0xFFFFFFFFu, val, delta, static_cast<int>(TileSize));
    }

    template<typename T>
    inline T shfl_up(T val, unsigned int delta) const {
        return __shfl_up_sync(0xFFFFFFFFu, val, delta, static_cast<int>(TileSize));
    }

    template<typename T>
    inline T shfl_xor(T val, int laneMask) const {
        return __shfl_xor_sync(0xFFFFFFFFu, val, laneMask, static_cast<int>(TileSize));
    }
};

// tiled_partition free function
template<unsigned int TileSize>
inline thread_block_tile<TileSize> tiled_partition(const thread_block&) {
    return thread_block_tile<TileSize>{};
}

template<unsigned int TileSize>
inline thread_block_tile<TileSize> tiled_partition(const coalesced_group&) {
    return thread_block_tile<TileSize>{};
}

// ── grid_group (already declared in cpu_cuda_env.h; keep compatible) ─────────
class grid_group : public thread_group {
public:
    inline void sync() const override { vgre_jit_syncgrid(); }

    inline unsigned int size() const override {
        auto* gd = vgre_jit_get_gridDim();
        auto* bd = vgre_jit_get_blockDim();
        return gd->x * gd->y * gd->z * bd->x * bd->y * bd->z;
    }

    inline unsigned int thread_rank() const override {
        auto* ti = vgre_jit_get_threadIdx();
        auto* bi = vgre_jit_get_blockIdx();
        auto* bd = vgre_jit_get_blockDim();
        auto* gd = vgre_jit_get_gridDim();
        unsigned int threads_per_block = bd->x * bd->y * bd->z;
        unsigned int block_rank = (bi->z * gd->y + bi->y) * gd->x + bi->x;
        unsigned int thread_in_block = (ti->z * bd->y + ti->y) * bd->x + ti->x;
        return block_rank * threads_per_block + thread_in_block;
    }

    inline bool is_valid() const { return true; }
};

inline grid_group this_grid() { return grid_group{}; }

// ── multi_grid_group (cross-device cooperative groups) ──────────────────────
// In VGRE this is a single-host abstraction; behaves like grid_group.
class multi_grid_group : public thread_group {
public:
    inline void sync() const override { vgre_jit_syncgrid(); }

    inline unsigned int size() const override {
        auto* gd = vgre_jit_get_gridDim();
        auto* bd = vgre_jit_get_blockDim();
        return gd->x * gd->y * gd->z * bd->x * bd->y * bd->z;
    }

    inline unsigned int thread_rank() const override {
        auto* ti = vgre_jit_get_threadIdx();
        auto* bi = vgre_jit_get_blockIdx();
        auto* bd = vgre_jit_get_blockDim();
        auto* gd = vgre_jit_get_gridDim();
        unsigned int threads_per_block = bd->x * bd->y * bd->z;
        unsigned int block_rank = (bi->z * gd->y + bi->y) * gd->x + bi->x;
        unsigned int thread_in_block = (ti->z * bd->y + ti->y) * bd->x + ti->x;
        return block_rank * threads_per_block + thread_in_block;
    }

    inline unsigned int num_grids() const { return 1; }
    inline unsigned int grid_rank() const { return 0; }
};

inline multi_grid_group this_multi_grid() { return multi_grid_group{}; }

// ── reduce() ─────────────────────────────────────────────────────────────────
// Binary reduction across a thread_group using shfl_xor.
template<typename Group, typename T, typename BinaryOp>
inline T reduce(const Group& g, T val, BinaryOp op) {
    const unsigned int sz = g.size();
    if (sz == 1) return val;
    // Only valid for power-of-two sizes (warp / tile sizes are always Po2)
    for (unsigned int offset = sz >> 1; offset > 0; offset >>= 1) {
        T other = g.shfl_xor(val, static_cast<int>(offset));
        val = op(val, other);
    }
    return val;
}

// Convenience overloads for built-in ops
namespace detail {
    struct sum_op { template<typename T> inline T operator()(T a, T b) const { return a + b; } };
    struct min_op { template<typename T> inline T operator()(T a, T b) const { return a < b ? a : b; } };
    struct max_op { template<typename T> inline T operator()(T a, T b) const { return a > b ? a : b; } };
}

template<typename Group, typename T>
inline T reduce(const Group& g, T val) { return reduce(g, val, detail::sum_op{}); }

template<typename Group, typename T>
inline T reduce_sum(const Group& g, T val) { return reduce(g, val, detail::sum_op{}); }

template<typename Group, typename T>
inline T reduce_min(const Group& g, T val) { return reduce(g, val, detail::min_op{}); }

template<typename Group, typename T>
inline T reduce_max(const Group& g, T val) { return reduce(g, val, detail::max_op{}); }

// ── partition() ──────────────────────────────────────────────────────────────
// Splits a group into sub-groups of size `sub_group_size`.
// Returns a thread_block_tile for the caller's sub-group.
template<unsigned int SubGroupSize>
inline thread_block_tile<SubGroupSize> partition(const thread_block&) {
    return thread_block_tile<SubGroupSize>{};
}

// ── match_any / match_all (warp-level match primitives) ────────────────────
// Returns a bitmask of threads in the group with the same value.
// Serial CPU model: all threads match → return full mask.
template<typename Group, typename T>
inline unsigned int match_any(const Group&, T) {
    return 0xFFFFFFFFu;
}

template<typename Group, typename T>
inline unsigned int match_all(const Group&, T, int* pred) {
    if (pred) *pred = 1;
    return 0xFFFFFFFFu;
}

} // namespace cooperative_groups

#endif // VGRE_COMPILER_CUDA_DEVICE_LIBS_COOPERATIVE_GROUPS_H
