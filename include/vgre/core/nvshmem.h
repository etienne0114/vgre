#ifndef VGRE_CORE_NVSHMEM_H
#define VGRE_CORE_NVSHMEM_H

// Phase 3, Track P3-16 — NVSHMEM-style symmetric memory + one-sided collectives.
//
// NVSHMEM is a PGAS: every PE (GPU) owns a *symmetric heap* — an allocation lives
// at the same offset on every PE — and communicates with one-sided put/get/atomic
// (the initiator drives the transfer; the target is passive). This is the
// substrate for fine-grained overlap and one-sided collectives. This module
// models P PEs in-process (each a heap buffer) with put/get/atomic, and builds a
// one-sided ring AllReduce (reduce-scatter + all-gather) on top — the result is
// bit-identical to a direct elementwise sum.

#include <cstdint>
#include <cstring>
#include <vector>

namespace vgre {
namespace dist {

class SymmetricHeap {
public:
    SymmetricHeap(int numPEs, size_t bytesPerPE)
        : pes_(numPEs), heap_(numPEs, std::vector<uint8_t>(bytesPerPE, 0)) {}

    int    n_pes() const { return pes_; }
    size_t malloc(size_t bytes) { const size_t off = bump_; bump_ += bytes; return off; }

    template <typename T> T* ptr(int pe, size_t off) {
        return reinterpret_cast<T*>(heap_[pe].data() + off);
    }

    // One-sided put: initiator writes `bytes` from `src` into PE `destPE`'s heap.
    void put(int destPE, size_t destOff, const void* src, size_t bytes) {
        std::memcpy(heap_[destPE].data() + destOff, src, bytes);
    }
    // One-sided get: initiator reads `bytes` from PE `srcPE`'s heap into `dest`.
    void get(void* dest, int srcPE, size_t srcOff, size_t bytes) {
        std::memcpy(dest, heap_[srcPE].data() + srcOff, bytes);
    }
    // One-sided float atomic add into a remote PE's heap.
    void atomic_add_f(int destPE, size_t destOff, float v) {
        *reinterpret_cast<float*>(heap_[destPE].data() + destOff) += v;
    }

private:
    int    pes_;
    size_t bump_ = 0;
    std::vector<std::vector<uint8_t>> heap_;
};

// One-sided ring AllReduce (sum) of `count` floats (count % P == 0). Each PE's
// contribution is at symmetric offset `dataOff`; `tmpOff` is a count/P-float
// scratch. After the call every PE's [dataOff, dataOff+count) holds the sum.
// Canonical ring: P−1 reduce-scatter steps (put neighbor's chunk, add) then P−1
// all-gather steps (propagate the reduced chunks).
inline void nvshmem_allreduce_sum(SymmetricHeap& sh, size_t dataOff, size_t tmpOff,
                                  int count) {
    const int P = sh.n_pes();
    if (P <= 1) return;
    const int cs = count / P;                  // chunk size (floats)
    const size_t cbytes = static_cast<size_t>(cs) * sizeof(float);

    auto chunk = [&](int pe, int c) { return sh.ptr<float>(pe, dataOff) + c * cs; };
    auto tmp   = [&](int pe)        { return sh.ptr<float>(pe, tmpOff); };

    // Reduce-scatter: after P-1 steps, PE r owns the reduced chunk (r+1)%P.
    for (int i = 0; i < P - 1; ++i) {
        for (int r = 0; r < P; ++r) {           // put: r → (r+1)%P, chunk (r-i)
            const int sc = ((r - i) % P + P) % P;
            sh.put((r + 1) % P, tmpOff, chunk(r, sc), cbytes);
        }
        for (int r = 0; r < P; ++r) {           // add received chunk into own buffer
            const int rc = ((r - 1 - i) % P + P) % P;
            float* dst = chunk(r, rc); const float* add = tmp(r);
            for (int e = 0; e < cs; ++e) dst[e] += add[e];
        }
    }
    // All-gather: propagate each reduced chunk around the ring (overwrite).
    for (int i = 0; i < P - 1; ++i)
        for (int r = 0; r < P; ++r) {
            const int sc = ((r + 1 - i) % P + P) % P;
            sh.put((r + 1) % P, dataOff + static_cast<size_t>(sc) * cs * sizeof(float),
                   chunk(r, sc), cbytes);
        }
}

} // namespace dist
} // namespace vgre

#endif // VGRE_CORE_NVSHMEM_H
