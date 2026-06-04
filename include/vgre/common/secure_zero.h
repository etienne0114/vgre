#pragma once
// QUEUE-39: Portable cryptographic zeroization — prevents dead-store elimination.
//
// Use vgre_secure_zero() wherever sensitive buffers (keys, passwords, tokens,
// DH scalars) must be cleared before deallocation or reuse.  Standard memset()
// may be eliminated by optimizing compilers when the compiler can prove the
// buffer is not accessed again; these platform wrappers guarantee the write is
// not removed.
//
// Platform dispatch:
//   Windows       — SecureZeroMemory  (always available via kernel32)
//   Linux/glibc≥2.25 — explicit_bzero (avoids optimizer removal by design)
//   macOS/C11     — memset_s          (C11 §K.3.7.4.1: zeroing always happens)
//   Fallback       — volatile byte loop + atomic_thread_fence(seq_cst)
//
// All callers must #include this header; do NOT call memset() for security use.

#include <cstddef>
#include <cstdint>
#include <atomic>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
#  include <string.h>   // explicit_bzero
#elif defined(__APPLE__)
#  include <string.h>   // memset_s
#endif

namespace vgre {
namespace common {

// vgre_secure_zero: zero n bytes at p, guaranteed not optimised away.
// Precondition: p != nullptr || n == 0.
inline void vgre_secure_zero(void* p, std::size_t n) noexcept {
#if defined(_WIN32)
    SecureZeroMemory(p, n);
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
    explicit_bzero(p, n);
#elif defined(__APPLE__)
    if (n) memset_s(p, n, 0, n);
#else
    volatile std::uint8_t* vp = static_cast<volatile std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i)
        vp[i] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// Convenience overload for fixed-size arrays.
template<std::size_t N>
inline void vgre_secure_zero(std::uint8_t (&arr)[N]) noexcept {
    vgre_secure_zero(arr, N);
}

} // namespace common
} // namespace vgre
