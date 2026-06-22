/**
 * @file cpu_barriers.h
 * @brief Cross-platform CPU barriers and synchronization instructions.
 *
 * Provides CPU-specific spin hints (pause/yield) and compiler/hardware
 * memory barriers portable across GCC, Clang, and MSVC for x86 and ARM64.
 */

#pragma once

#include <thread>

#if defined(_MSC_VER)
#  include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#endif

// ── VGRE_CPU_PAUSE: Spin-wait loop hint ──────────────────────────────────────
// Tells the processor that the thread is in a spin-wait loop.
// Reduces power consumption and pipeline pressure.
#if defined(_MSC_VER)
#  if defined(_M_ARM64)
#    define VGRE_CPU_PAUSE() __yield()
#  else
#    define VGRE_CPU_PAUSE() _mm_pause()
#  endif
#elif defined(__x86_64__) || defined(__i386__)
#  define VGRE_CPU_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#  define VGRE_CPU_PAUSE() __asm__ volatile("yield" ::: "memory")
#else
#  define VGRE_CPU_PAUSE() std::this_thread::yield()
#endif

// ── VGRE_COMPILER_BARRIER: Prevent compiler reordering ────────────────────────
// Forces the compiler to commit all register/memory updates to RAM and
// prevents instruction scheduling across the barrier.
#if defined(_MSC_VER)
#  define VGRE_COMPILER_BARRIER() _ReadWriteBarrier()
#else
#  define VGRE_COMPILER_BARRIER() __asm__ volatile("" ::: "memory")
#endif

// ── VGRE_FULL_BARRIER: Hardware memory fence ───────────────────────────────
// Emits processor-level memory fence instructions (e.g. mfence/dmb) to
// guarantee global memory ordering across physical cores.
#if defined(_MSC_VER)
#  define VGRE_FULL_BARRIER() MemoryBarrier()
#else
#  define VGRE_FULL_BARRIER() __sync_synchronize()
#endif

// ── Atomic load/store (acquire/release) on int* ──────────────────────────────
// GCC/Clang expose __atomic_load_n/__atomic_store_n builtins with explicit
// memory-order arguments.  MSVC has no equivalent builtins; on x86/x64 (TSO)
// volatile reads already carry acquire semantics and volatile writes carry
// release semantics.  On ARM64/MSVC the compiler emits ldaex/stlex.
// _ReadWriteBarrier() suppresses compiler reordering across the access.
#if defined(_MSC_VER)
inline int  vgre_atomic_load_acquire (const int* p) noexcept {
    int v = *reinterpret_cast<const volatile int*>(p);
    _ReadWriteBarrier();
    return v;
}
inline void vgre_atomic_store_release(int* p, int v) noexcept {
    _ReadWriteBarrier();
    *reinterpret_cast<volatile int*>(p) = v;
}
#  define VGRE_ATOMIC_LOAD_ACQUIRE(ptr)        vgre_atomic_load_acquire(ptr)
#  define VGRE_ATOMIC_STORE_RELEASE(ptr, val)  vgre_atomic_store_release((ptr), (val))
#else
#  define VGRE_ATOMIC_LOAD_ACQUIRE(ptr)        __atomic_load_n((ptr), __ATOMIC_ACQUIRE)
#  define VGRE_ATOMIC_STORE_RELEASE(ptr, val)  __atomic_store_n((ptr), (val), __ATOMIC_RELEASE)
#endif
