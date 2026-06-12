#ifndef VGRE_CORE_TMEM_H
#define VGRE_CORE_TMEM_H

// Phase 3, Track P3-7 — Blackwell tcgen05 Tensor Memory (TMEM).
//
// Blackwell's 5th-gen tensor cores (tcgen05) keep the MMA accumulator in a
// dedicated on-chip *Tensor Memory* — distinct from registers and shared memory.
// A CTA `tcgen05.alloc`s columns of TMEM, the `tcgen05.mma` accumulates the
// D = A·B (+D) result *into* a TMEM address, and `tcgen05.ld` / `tcgen05.st`
// move fragments between TMEM and registers (`tcgen05.cp` stages SMEM→TMEM).
//
// Hardware TMEM is 128 lanes (rows) × 512 columns × 32-bit, allocated in column
// units; an M×N FP32 accumulator occupies M lanes × N columns. This models that
// geometry exactly: a per-CTA lane×column word arena, a column bump-allocator
// (the `tcgen05.alloc` permit), tile scatter/gather in the real (lane,column)
// layout, and accumulate-in-place for the K-loop. The async aspect is synchronous
// on CPU, but the data path — alloc → mma-into-TMEM → ld-out — is faithful, so a
// tcgen05 GEMM driven through TMEM matches a reference GEMM.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgre {
namespace tmem {

constexpr int kLanes = 128;   // TMEM rows (one per lane)
constexpr int kCols  = 512;   // 32-bit columns per lane (Blackwell)

// A TMEM address packs the lane base in the high 16 bits and the column base in
// the low 16 bits (tcgen05.alloc returns one with lane base 0).
inline uint32_t make_addr(uint32_t lane, uint32_t col) { return (lane << 16) | (col & 0xFFFFu); }
inline uint32_t addr_lane(uint32_t a) { return a >> 16; }
inline uint32_t addr_col(uint32_t a)  { return a & 0xFFFFu; }

class TensorMemory {
public:
    TensorMemory() : arena_(static_cast<size_t>(kLanes) * kCols, 0u) {}

    // tcgen05.alloc: reserve `nCols` columns (across all lanes), returning the
    // TMEM address of the allocation (lane base 0). Bump allocator = the single
    // outstanding allocation permit a CTA holds.
    uint32_t alloc(int nCols) {
        uint32_t col = static_cast<uint32_t>(colNext_);
        colNext_ += nCols;
        return make_addr(0, col);
    }
    // tcgen05.dealloc: release the most recent allocation (LIFO).
    void dealloc(uint32_t /*addr*/, int nCols) {
        colNext_ -= nCols;
        if (colNext_ < 0) colNext_ = 0;
    }
    void reset() { colNext_ = 0; std::fill(arena_.begin(), arena_.end(), 0u); }
    int  columns_used() const { return colNext_; }

    // Word accessor at (lane, column) — column = addr_col(base) + colOffset.
    uint32_t*       word(uint32_t lane, uint32_t col)       { return &arena_[(size_t)lane * kCols + col]; }
    const uint32_t* word(uint32_t lane, uint32_t col) const { return &arena_[(size_t)lane * kCols + col]; }

    // ── tile scatter / gather (the (lane,column) accumulator layout) ──────────
    // Store a contiguous row-major [M,N] FP32 tile so element (m,n) lands at
    // lane (laneBase+m), column (colBase+n).
    void store_tile(uint32_t addr, const float* src, int M, int N) {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        for (int m = 0; m < M; ++m)
            std::memcpy(word(l0 + m, c0), src + (size_t)m * N, (size_t)N * 4);
    }
    void load_tile(float* dst, uint32_t addr, int M, int N) const {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        for (int m = 0; m < M; ++m)
            std::memcpy(dst + (size_t)m * N, word(l0 + m, c0), (size_t)N * 4);
    }
    // D += tile (the tcgen05.mma K-loop accumulate, enable_input_d = true).
    void accumulate_tile(uint32_t addr, const float* src, int M, int N) {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        for (int m = 0; m < M; ++m) {
            float* d = reinterpret_cast<float*>(word(l0 + m, c0));
            const float* s = src + (size_t)m * N;
            for (int n = 0; n < N; ++n) d[n] += s[n];
        }
    }

    // ── generic register/SMEM ↔ TMEM word moves (tcgen05.ld / .st / .cp) ──────
    // `addr` is a TMEM address; the lane dimension carries `lanes` rows of
    // `nWordsPerLane` 32-bit words each (the fragment shape, e.g. 32x32b.xN).
    void ld(uint32_t* dstRegs, uint32_t addr, int lanes, int nWordsPerLane) const {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        for (int l = 0; l < lanes; ++l)
            std::memcpy(dstRegs + (size_t)l * nWordsPerLane, word(l0 + l, c0),
                        (size_t)nWordsPerLane * 4);
    }
    void st(uint32_t addr, const uint32_t* srcRegs, int lanes, int nWordsPerLane) {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        for (int l = 0; l < lanes; ++l)
            std::memcpy(word(l0 + l, c0), srcRegs + (size_t)l * nWordsPerLane,
                        (size_t)nWordsPerLane * 4);
    }
    // tcgen05.cp: stage a [lanes × nWordsPerLane] block from SMEM into TMEM.
    void cp_from_smem(uint32_t addr, const void* smem, int lanes, int nWordsPerLane) {
        const uint32_t l0 = addr_lane(addr), c0 = addr_col(addr);
        const uint8_t* s = static_cast<const uint8_t*>(smem);
        for (int l = 0; l < lanes; ++l)
            std::memcpy(word(l0 + l, c0), s + (size_t)l * nWordsPerLane * 4,
                        (size_t)nWordsPerLane * 4);
    }

private:
    std::vector<uint32_t> arena_;
    int colNext_ = 0;
};

}  // namespace tmem
}  // namespace vgre

#endif  // VGRE_CORE_TMEM_H
