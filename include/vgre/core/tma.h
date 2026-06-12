#ifndef VGRE_CORE_TMA_H
#define VGRE_CORE_TMA_H

// Phase 3, Track P3-5 — Tensor Memory Accelerator (TMA).
//
// Hopper/Blackwell move tiles between global and shared memory with the TMA: an
// asynchronous *bulk tensor* copy driven by a `CUtensorMap` descriptor (global
// shape + strides + box/tile shape + element type) and completed via an
// mbarrier transaction-byte count. `cp.async.bulk.tensor` copies an N-D box at
// given coordinates from global → shared (box-contiguous), **zero-filling
// out-of-bounds elements** (the real TMA boundary behavior), and signals the
// mbarrier with the bytes delivered. This module implements that data path
// exactly (strided gather, OOB padding, mbarrier completion); the async aspect
// is synchronous on CPU but the completion semantics are modeled faithfully.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vgre {
namespace tma {

constexpr int kMaxRank = 5;

enum class Swizzle { NONE, B32, B64, B128 };

// CUtensorMap equivalent.
struct TensorMap {
    const uint8_t* globalBase = nullptr;
    int      rank = 0;
    uint64_t globalDim[kMaxRank]    = {0};   // global shape, elements per dim
    uint64_t globalStride[kMaxRank] = {0};   // bytes between consecutive elems per dim
    uint32_t boxDim[kMaxRank]       = {0};   // box (tile) shape, elements per dim
    uint32_t elemBytes              = 0;
    Swizzle  swizzle = Swizzle::NONE;
};

// cuTensorMapEncodeTiled equivalent. `globalStride` is bytes per dim (the caller
// supplies the layout; e.g. row-major [D0,D1]: stride = {D1*elem, elem}).
inline TensorMap tensor_map_encode_tiled(const void* base, int rank,
        const uint64_t* globalDim, const uint64_t* globalStride,
        const uint32_t* boxDim, uint32_t elemBytes, Swizzle sw = Swizzle::NONE) {
    TensorMap tm;
    tm.globalBase = static_cast<const uint8_t*>(base);
    tm.rank = rank;
    tm.elemBytes = elemBytes;
    tm.swizzle = sw;
    for (int d = 0; d < rank; ++d) {
        tm.globalDim[d]    = globalDim[d];
        tm.globalStride[d] = globalStride[d];
        tm.boxDim[d]       = boxDim[d];
    }
    return tm;
}

// ── mbarrier (transaction-byte completion) ────────────────────────────────────
struct Mbarrier { uint64_t expectBytes = 0; uint64_t arrivedBytes = 0; };
inline void mbarrier_init(Mbarrier& m) { m.expectBytes = 0; m.arrivedBytes = 0; }
inline void mbarrier_expect_tx(Mbarrier& m, uint64_t bytes)   { m.expectBytes  += bytes; }
inline void mbarrier_complete_tx(Mbarrier& m, uint64_t bytes) { m.arrivedBytes += bytes; }
inline bool mbarrier_try_wait(const Mbarrier& m) {
    return m.expectBytes > 0 && m.arrivedBytes >= m.expectBytes;
}

// Bytes a full box transfers (the value passed to expect_tx).
inline uint64_t box_bytes(const TensorMap& tm) {
    uint64_t n = 1;
    for (int d = 0; d < tm.rank; ++d) n *= tm.boxDim[d];
    return n * tm.elemBytes;
}

// cp.async.bulk.tensor.Nd: copy the box at `coord` (per-dim start index) from the
// global tensor to `dstSmem` (box-contiguous, innermost dim fastest), zero-
// filling any out-of-bounds element, then complete the mbarrier transaction.
inline void cp_async_bulk_tensor(void* dstSmem, const TensorMap& tm,
                                 const int32_t* coord, Mbarrier* mbar) {
    uint8_t* dst = static_cast<uint8_t*>(dstSmem);
    size_t boxElems = 1;
    for (int d = 0; d < tm.rank; ++d) boxElems *= tm.boxDim[d];
    int idx[kMaxRank] = {0};
    uint64_t bytes = 0;
    for (size_t e = 0; e < boxElems; ++e) {
        bool oob = false;
        uint64_t off = 0;
        for (int d = 0; d < tm.rank; ++d) {
            const int64_t g = static_cast<int64_t>(coord[d]) + idx[d];
            if (g < 0 || static_cast<uint64_t>(g) >= tm.globalDim[d]) { oob = true; break; }
            off += static_cast<uint64_t>(g) * tm.globalStride[d];
        }
        if (oob) std::memset(dst, 0, tm.elemBytes);                 // TMA OOB → zero
        else     std::memcpy(dst, tm.globalBase + off, tm.elemBytes);
        dst   += tm.elemBytes;
        bytes += tm.elemBytes;
        for (int d = tm.rank - 1; d >= 0; --d) {                    // innermost fastest
            if (++idx[d] < static_cast<int>(tm.boxDim[d])) break;
            idx[d] = 0;
        }
    }
    if (mbar) mbarrier_complete_tx(*mbar, bytes);
}

// cp.async.bulk: plain contiguous bulk copy global → shared, then complete tx.
inline void cp_async_bulk(void* dst, const void* src, size_t bytes, Mbarrier* mbar) {
    std::memcpy(dst, src, bytes);
    if (mbar) mbarrier_complete_tx(*mbar, bytes);
}

} // namespace tma
} // namespace vgre

#endif // VGRE_CORE_TMA_H
