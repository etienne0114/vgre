#ifndef VGRE_CORE_DETERMINISTIC_H
#define VGRE_CORE_DETERMINISTIC_H

// Phase 3, Track P3-18 — Bit-deterministic mode.
//
// Floating-point reductions are non-associative, so a parallel sum whose partial
// results combine in thread-completion order is NOT byte-reproducible — a real
// blocker for debugging and regression gating. Determinism needs two primitives:
//   1. A reduction whose result depends ONLY on the data and a fixed chunk
//      shape, never on how threads are scheduled — fixed in-chunk order, fixed
//      partial-combine order.
//   2. A counter-based RNG: the value for a given (seed, counter) is the same
//      regardless of when/where it is drawn — no shared sequential state.
// `VGRE_DETERMINISTIC=1` selects these paths at runtime.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vgre {
namespace det {

// Whether deterministic mode is requested (VGRE_DETERMINISTIC=1).
inline bool enabled() {
    const char* e = std::getenv("VGRE_DETERMINISTIC");
    return e && e[0] == '1';
}

// Deterministic reduction: split [0,n) into fixed `chunk`-sized chunks, sum each
// chunk left-to-right, then sum the partials left-to-right. The result is a pure
// function of (x, n, chunk) — identical no matter which thread computes which
// chunk — so two runs are byte-identical.
inline float deterministic_reduce(const float* x, size_t n, size_t chunk = 256) {
    if (n == 0) return 0.0f;
    const size_t nchunks = (n + chunk - 1) / chunk;
    std::vector<float> partial(nchunks);
    for (size_t c = 0; c < nchunks; ++c) {
        const size_t lo = c * chunk, hi = (lo + chunk < n) ? lo + chunk : n;
        float s = 0.0f;
        for (size_t i = lo; i < hi; ++i) s += x[i];     // fixed in-chunk order
        partial[c] = s;
    }
    float total = 0.0f;
    for (size_t c = 0; c < nchunks; ++c) total += partial[c];  // fixed combine order
    return total;
}

// ── Counter-based RNG (splitmix64 mixing) ─────────────────────────────────────
// Stateless: u01(seed, counter) is a pure function, so it is order- and
// thread-independent — the basis of reproducible RNG (cuRAND-Philox-style).
inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
inline uint32_t counter_u32(uint64_t seed, uint64_t counter) {
    return static_cast<uint32_t>(splitmix64(seed ^ splitmix64(counter)) >> 32);
}
// Uniform [0,1) with 24-bit resolution (matches float mantissa).
inline float counter_u01(uint64_t seed, uint64_t counter) {
    return (counter_u32(seed, counter) >> 8) * (1.0f / 16777216.0f);
}

} // namespace det
} // namespace vgre

#endif // VGRE_CORE_DETERMINISTIC_H
