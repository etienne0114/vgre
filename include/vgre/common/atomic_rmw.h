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

// exchange / bitwise / compare-and-swap map straight to the builtins; each returns
// the OLD value (atomicCAS returns old whether or not the swap happened).
template <typename T> inline T atomicExchT(void* p, T v) {
    return __atomic_exchange_n(static_cast<T*>(p), v, __ATOMIC_RELAXED);
}
template <typename T> inline T atomicAndT(void* p, T v) {
    return __atomic_fetch_and(static_cast<T*>(p), v, __ATOMIC_RELAXED);
}
template <typename T> inline T atomicOrT(void* p, T v) {
    return __atomic_fetch_or(static_cast<T*>(p), v, __ATOMIC_RELAXED);
}
template <typename T> inline T atomicXorT(void* p, T v) {
    return __atomic_fetch_xor(static_cast<T*>(p), v, __ATOMIC_RELAXED);
}
template <typename T> inline T atomicCasT(void* p, T cmp, T v) {
    T expected = cmp;   // on return holds the actual old value either way
    __atomic_compare_exchange_n(static_cast<T*>(p), &expected, v, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}
// min/max: CAS loop; signedness comes from T's comparison. Returns OLD value.
template <typename T> inline T atomicMaxT(void* p, T v) {
    auto* q = static_cast<T*>(p);
    T o = __atomic_load_n(q, __ATOMIC_RELAXED);
    while (v > o && !__atomic_compare_exchange_n(q, &o, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return o;
}
template <typename T> inline T atomicMinT(void* p, T v) {
    auto* q = static_cast<T*>(p);
    T o = __atomic_load_n(q, __ATOMIC_RELAXED);
    while (v < o && !__atomic_compare_exchange_n(q, &o, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return o;
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

template <typename T> inline T atomicExchT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); std::memcpy(p, &v, sizeof(T)); return o;
}
template <typename T> inline T atomicAndT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); T n = o & v; std::memcpy(p, &n, sizeof(T)); return o;
}
template <typename T> inline T atomicOrT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); T n = o | v; std::memcpy(p, &n, sizeof(T)); return o;
}
template <typename T> inline T atomicXorT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); T n = o ^ v; std::memcpy(p, &n, sizeof(T)); return o;
}
template <typename T> inline T atomicCasT(void* p, T cmp, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); if (o == cmp) std::memcpy(p, &v, sizeof(T)); return o;
}
template <typename T> inline T atomicMaxT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); if (v > o) std::memcpy(p, &v, sizeof(T)); return o;
}
template <typename T> inline T atomicMinT(void* p, T v) {
    std::lock_guard<std::mutex> lk(atomicRmwMutex());
    T o; std::memcpy(&o, p, sizeof(T)); if (v < o) std::memcpy(p, &v, sizeof(T)); return o;
}

#endif

// Concrete width/sign-typed entry points shared by both implementations. min/max
// carry signedness in the type; exch/and/or/xor/cas are bit operations (unsigned).
inline uint32_t atomicExchU32(void* p, uint32_t v) { return atomicExchT<uint32_t>(p, v); }
inline uint64_t atomicExchU64(void* p, uint64_t v) { return atomicExchT<uint64_t>(p, v); }
inline uint32_t atomicAndU32(void* p, uint32_t v)  { return atomicAndT<uint32_t>(p, v); }
inline uint64_t atomicAndU64(void* p, uint64_t v)  { return atomicAndT<uint64_t>(p, v); }
inline uint32_t atomicOrU32(void* p, uint32_t v)   { return atomicOrT<uint32_t>(p, v); }
inline uint64_t atomicOrU64(void* p, uint64_t v)   { return atomicOrT<uint64_t>(p, v); }
inline uint32_t atomicXorU32(void* p, uint32_t v)  { return atomicXorT<uint32_t>(p, v); }
inline uint64_t atomicXorU64(void* p, uint64_t v)  { return atomicXorT<uint64_t>(p, v); }
inline uint32_t atomicCasU32(void* p, uint32_t c, uint32_t v) { return atomicCasT<uint32_t>(p, c, v); }
inline uint64_t atomicCasU64(void* p, uint64_t c, uint64_t v) { return atomicCasT<uint64_t>(p, c, v); }
inline uint32_t atomicMaxU32(void* p, uint32_t v) { return atomicMaxT<uint32_t>(p, v); }
inline uint64_t atomicMaxU64(void* p, uint64_t v) { return atomicMaxT<uint64_t>(p, v); }
inline int32_t  atomicMaxS32(void* p, int32_t v)  { return atomicMaxT<int32_t>(p, v); }
inline int64_t  atomicMaxS64(void* p, int64_t v)  { return atomicMaxT<int64_t>(p, v); }
inline uint32_t atomicMinU32(void* p, uint32_t v) { return atomicMinT<uint32_t>(p, v); }
inline uint64_t atomicMinU64(void* p, uint64_t v) { return atomicMinT<uint64_t>(p, v); }
inline int32_t  atomicMinS32(void* p, int32_t v)  { return atomicMinT<int32_t>(p, v); }
inline int64_t  atomicMinS64(void* p, int64_t v)  { return atomicMinT<int64_t>(p, v); }

}  // namespace common
}  // namespace vgre
