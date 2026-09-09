#pragma once

// Lock-free atomic read-modify-write primitives on raw memory, shared by every
// tier that implements CUDA `atomicAdd` (the Tier-1 compiled backend and the
// Tier-0 PTX interpreter both run a grid's CTAs on parallel OS threads, so an
// atomic accumulate is required or concurrent CTAs racing the same address lose
// updates). Each returns the OLD value, matching CUDA atomicAdd semantics.
//
// Integers use the compiler's fetch-add intrinsic; floats/doubles use a bit
// compare-exchange loop (hardware has no native FP fetch-add). Where the GCC/
// Clang __atomic builtins are unavailable (e.g. MSVC) a single process-wide lock
// gives the same correctness — atomics are a small fraction of real kernel work.

#include <cstdint>
#include <cstring>

#if !(defined(__GNUC__) || defined(__clang__))
#include <mutex>
#endif

namespace vgre {
namespace common {

#if defined(__GNUC__) || defined(__clang__)

inline uint32_t atomicAddU32(void* p, uint32_t v) {
    return __atomic_fetch_add(static_cast<uint32_t*>(p), v, __ATOMIC_RELAXED);
}
inline uint64_t atomicAddU64(void* p, uint64_t v) {
    return __atomic_fetch_add(static_cast<uint64_t*>(p), v, __ATOMIC_RELAXED);
}
inline float atomicAddF32(void* p, float add) {
    auto* q = static_cast<uint32_t*>(p);
    uint32_t ob = __atomic_load_n(q, __ATOMIC_RELAXED), nb;
    float old, nw;
    do {
        std::memcpy(&old, &ob, 4);
        nw = old + add;
        std::memcpy(&nb, &nw, 4);
    } while (!__atomic_compare_exchange_n(q, &ob, nb, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    return old;
}
inline double atomicAddF64(void* p, double add) {
    auto* q = static_cast<uint64_t*>(p);
    uint64_t ob = __atomic_load_n(q, __ATOMIC_RELAXED), nb;
    double old, nw;
    do {
        std::memcpy(&old, &ob, 8);
        nw = old + add;
        std::memcpy(&nb, &nw, 8);
    } while (!__atomic_compare_exchange_n(q, &ob, nb, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    return old;
}

#else  // portable locked fallback — one shared lock (inline-function local static).

inline std::mutex& atomicRmwMutex() {
    static std::mutex m;
    return m;
}
inline uint32_t atomicAddU32(void* p, uint32_t v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    uint32_t o; std::memcpy(&o, p, 4); uint32_t n = o + v; std::memcpy(p, &n, 4); return o;
}
inline uint64_t atomicAddU64(void* p, uint64_t v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    uint64_t o; std::memcpy(&o, p, 8); uint64_t n = o + v; std::memcpy(p, &n, 8); return o;
}
inline float atomicAddF32(void* p, float add) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    float o; std::memcpy(&o, p, 4); float n = o + add; std::memcpy(p, &n, 4); return o;
}
inline double atomicAddF64(void* p, double add) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    double o; std::memcpy(&o, p, 8); double n = o + add; std::memcpy(p, &n, 8); return o;
}

#endif

}  // namespace common
}  // namespace vgre
