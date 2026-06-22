#ifndef VGRE_COMPILER_CPU_CUDA_INTRINSICS_H
#define VGRE_COMPILER_CPU_CUDA_INTRINSICS_H

// ─────────────────────────────────────────────────────────────────────────────
// CUDA device intrinsics + built-in vector types for the CPU JIT environment.
//
// When VGRE JIT-compiles a CUDA C kernel it compiles the source as ordinary host
// C++ (clang++), so none of the implicit device intrinsics the NVCC front-end
// would normally inject are available.  This header provides faithful CPU
// implementations of the device-side intrinsics and the `floatN`/`intN`/… vector
// types that real CUDA and ML/DL kernels rely on, so that those kernels compile
// *and run* unchanged through the JIT path.
//
// Everything here is exact (not an approximation) unless noted: the "rounded"
// arithmetic intrinsics (`__fadd_rn`, `__dmul_rn`, …) compute the IEEE-754
// round-to-nearest result, which is what the host FPU already produces, and the
// "fast-math" intrinsics (`__expf`, `__sinf`, …) map to the full-precision libm
// function — strictly more accurate than the hardware approximation, never less.
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdint>
#include <cstring>
#include <atomic>

// ── Built-in vector types ───────────────────────────────────────────────────
// Defined at global scope (matching CUDA) and guarded so the device cuRAND
// header — which also needs `float2` — reuses these definitions instead of
// redefining them.
#ifndef VGRE_CUDA_VECTOR_TYPES_DEFINED
#define VGRE_CUDA_VECTOR_TYPES_DEFINED 1

#define VGRE_VEC1(NAME, T)                                                     \
    struct NAME { T x; };                                                      \
    static inline NAME make_##NAME(T x) { NAME r; r.x = x; return r; }
#define VGRE_VEC2(NAME, T, ALIGN)                                              \
    struct alignas(ALIGN) NAME { T x, y; };                                    \
    static inline NAME make_##NAME(T x, T y) { NAME r; r.x = x; r.y = y; return r; }
#define VGRE_VEC3(NAME, T)                                                     \
    struct NAME { T x, y, z; };                                                \
    static inline NAME make_##NAME(T x, T y, T z) { NAME r; r.x = x; r.y = y; r.z = z; return r; }
#define VGRE_VEC4(NAME, T, ALIGN)                                              \
    struct alignas(ALIGN) NAME { T x, y, z, w; };                              \
    static inline NAME make_##NAME(T x, T y, T z, T w) { NAME r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }

VGRE_VEC1(char1,  signed char)   VGRE_VEC2(char2,  signed char, 2)   VGRE_VEC3(char3,  signed char)   VGRE_VEC4(char4,  signed char, 4)
VGRE_VEC1(uchar1, unsigned char) VGRE_VEC2(uchar2, unsigned char, 2) VGRE_VEC3(uchar3, unsigned char) VGRE_VEC4(uchar4, unsigned char, 4)
VGRE_VEC1(short1, short)         VGRE_VEC2(short2, short, 4)         VGRE_VEC3(short3, short)         VGRE_VEC4(short4, short, 8)
VGRE_VEC1(ushort1,unsigned short)VGRE_VEC2(ushort2,unsigned short, 4)VGRE_VEC3(ushort3,unsigned short)VGRE_VEC4(ushort4,unsigned short, 8)
VGRE_VEC1(int1,   int)           VGRE_VEC2(int2,   int, 8)           VGRE_VEC3(int3,   int)           VGRE_VEC4(int4,   int, 16)
VGRE_VEC1(uint1,  unsigned)      VGRE_VEC2(uint2,  unsigned, 8)      VGRE_VEC3(uint3,  unsigned)      VGRE_VEC4(uint4,  unsigned, 16)
VGRE_VEC1(long1,  long long)     VGRE_VEC2(long2,  long long, 16)    VGRE_VEC3(long3,  long long)     VGRE_VEC4(long4,  long long, 16)
VGRE_VEC1(ulong1, unsigned long long) VGRE_VEC2(ulong2, unsigned long long, 16) VGRE_VEC3(ulong3, unsigned long long) VGRE_VEC4(ulong4, unsigned long long, 16)
VGRE_VEC1(float1, float)         VGRE_VEC2(float2, float, 8)         VGRE_VEC3(float3, float)         VGRE_VEC4(float4, float, 16)
VGRE_VEC1(double1,double)        VGRE_VEC2(double2,double, 16)       VGRE_VEC3(double3,double)        VGRE_VEC4(double4,double, 16)

// longlong/ulonglong aliases (same layout as long/ulong here on LP64).
using longlong2  = long2;   using longlong3  = long3;   using longlong4  = long4;
using ulonglong2 = ulong2;  using ulonglong3 = ulong3;  using ulonglong4 = ulong4;
static inline longlong2  make_longlong2 (long long x, long long y) { return make_long2(x, y); }
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) { return make_ulong2(x, y); }

#undef VGRE_VEC1
#undef VGRE_VEC2
#undef VGRE_VEC3
#undef VGRE_VEC4
#endif // VGRE_CUDA_VECTOR_TYPES_DEFINED

namespace vgre_cuda {

// ── Bit-reinterpret intrinsics ──────────────────────────────────────────────
static inline float  __int_as_float(int x)            { float f;  std::memcpy(&f, &x, 4); return f; }
static inline int    __float_as_int(float x)          { int i;    std::memcpy(&i, &x, 4); return i; }
static inline float  __uint_as_float(unsigned x)      { float f;  std::memcpy(&f, &x, 4); return f; }
static inline unsigned __float_as_uint(float x)       { unsigned u; std::memcpy(&u, &x, 4); return u; }
static inline long long __double_as_longlong(double x){ long long i; std::memcpy(&i, &x, 8); return i; }
static inline double __longlong_as_double(long long x){ double d;  std::memcpy(&d, &x, 8); return d; }
static inline int    __double2hiint(double x)         { long long i = __double_as_longlong(x); return (int)(i >> 32); }
static inline int    __double2loint(double x)         { long long i = __double_as_longlong(x); return (int)(i & 0xFFFFFFFFll); }
static inline double __hiloint2double(int hi, int lo) {
    long long i = ((long long)(unsigned)hi << 32) | (unsigned)lo;
    return __longlong_as_double(i);
}

// ── Integer intrinsics ──────────────────────────────────────────────────────
static inline int __mul24(int a, int b) {
    auto s24 = [](int v) { v &= 0x00FFFFFF; if (v & 0x00800000) v |= 0xFF000000; return v; };
    return s24(a) * s24(b);
}
static inline unsigned __umul24(unsigned a, unsigned b) { return (a & 0x00FFFFFFu) * (b & 0x00FFFFFFu); }
static inline int      __mulhi (int a, int b)           { return (int)(((long long)a * (long long)b) >> 32); }
static inline unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a * b) >> 32); }
// High 64 bits of a 64×64 product. Uses __int128 where the compiler has it
// (clang/gcc on all 64-bit targets, including Windows); otherwise a portable
// 32×32 decomposition so the JIT header also compiles under MSVC / on targets
// without __int128.
#if defined(__SIZEOF_INT128__)
static inline long long __mul64hi(long long a, long long b)            { return (long long)(((__int128)a * b) >> 64); }
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) { return (unsigned long long)(((unsigned __int128)a * b) >> 64); }
#else
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    const unsigned long long al = a & 0xFFFFFFFFull, ah = a >> 32;
    const unsigned long long bl = b & 0xFFFFFFFFull, bh = b >> 32;
    const unsigned long long ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    const unsigned long long cross = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
    return hh + (lh >> 32) + (hl >> 32) + (cross >> 32);
}
static inline long long __mul64hi(long long a, long long b) {
    unsigned long long hi = __umul64hi((unsigned long long)a, (unsigned long long)b);
    if (a < 0) hi -= (unsigned long long)b;   // signed correction (two's complement)
    if (b < 0) hi -= (unsigned long long)a;
    return (long long)hi;
}
#endif
static inline int      __sad (int a, int b, int c)                { return (a > b ? a - b : b - a) + c; }
static inline unsigned __usad(unsigned a, unsigned b, unsigned c) { return (a > b ? a - b : b - a) + c; }
static inline unsigned __byte_perm(unsigned lo, unsigned hi, unsigned sel) {
    unsigned long long both = ((unsigned long long)hi << 32) | lo;
    unsigned res = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned s = (sel >> (i * 4)) & 0xF;
        unsigned byte = (unsigned)((both >> ((s & 0x7) * 8)) & 0xFF);
        if (s & 0x8) byte = (byte & 0x80) ? 0xFF : 0x00; // sign-replicate
        res |= byte << (i * 8);
    }
    return res;
}

// ── Read-only data cache load ───────────────────────────────────────────────
template <typename T> static inline T __ldg(const T* ptr) { return *ptr; }

// ── Memory fences ───────────────────────────────────────────────────────────
static inline void __threadfence()        { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void __threadfence_block()  { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }

// ── Misc ────────────────────────────────────────────────────────────────────
static inline float __saturatef(float x) { return x != x ? 0.0f : (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x)); }
static inline void  __nanosleep(unsigned) { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void  __trap()  { std::abort(); }

// ── Single-precision rounded arithmetic ─────────────────────────────────────
// The host FPU already rounds to nearest-even; rz/ru/rd map to the same result
// (the JIT does not model alternate IEEE rounding modes for these helpers).
static inline float __fadd_rz(float a, float b) { return a + b; }
static inline float __fadd_ru(float a, float b) { return a + b; }
static inline float __fadd_rd(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rz(float a, float b) { return a * b; }
static inline float __fmul_ru(float a, float b) { return a * b; }
static inline float __fmul_rd(float a, float b) { return a * b; }
static inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
static inline float __fmaf_rz(float a, float b, float c) { return std::fma(a, b, c); }
static inline float __fmaf_ru(float a, float b, float c) { return std::fma(a, b, c); }
static inline float __fmaf_rd(float a, float b, float c) { return std::fma(a, b, c); }
static inline float __frcp_rn(float x) { return 1.0f / x; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float x) { return std::sqrt(x); }
static inline float __frsqrt_rn(float x) { return 1.0f / std::sqrt(x); }

// ── Double-precision rounded arithmetic ─────────────────────────────────────
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __drcp_rn(double x) { return 1.0 / x; }
static inline double __dsqrt_rn(double x) { return std::sqrt(x); }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }

} // namespace vgre_cuda

// ── Fast-math single precision (map to full-precision libm) ─────────────────
// These CUDA intrinsic names collide with glibc's *declared* internal libm
// symbols (`::__expf`, `::__sinf`, …) which are not actually linkable. Aliasing
// them to the public libm function via a function-like macro both avoids the
// declaration clash and resolves at link time, while being strictly more
// accurate than the GPU's fast-math approximation. Defined after <cmath> so the
// libc declarations are already processed; function-like macros only expand on
// a following '(', leaving bare identifiers untouched.
#ifndef __expf
#define __expf(x)        expf(x)
#endif
#ifndef __exp2f
#define __exp2f(x)       exp2f(x)
#endif
#ifndef __exp10f
#define __exp10f(x)      powf(10.0f, (x))
#endif
#ifndef __logf
#define __logf(x)        logf(x)
#endif
#ifndef __log2f
#define __log2f(x)       log2f(x)
#endif
#ifndef __log10f
#define __log10f(x)      log10f(x)
#endif
#ifndef __sinf
#define __sinf(x)        sinf(x)
#endif
#ifndef __cosf
#define __cosf(x)        cosf(x)
#endif
#ifndef __tanf
#define __tanf(x)        tanf(x)
#endif
#ifndef __powf
#define __powf(a, b)     powf((a), (b))
#endif
#ifndef __sincosf
#define __sincosf(x, s, c) sincosf((x), (s), (c))
#endif

namespace vgre_cuda {

// ── CUDA-named math (not in libm) ───────────────────────────────────────────
static inline float  rsqrtf(float x)  { return 1.0f / std::sqrt(x); }
static inline double rsqrt(double x)  { return 1.0 / std::sqrt(x); }
static inline float  rcbrtf(float x)  { return 1.0f / std::cbrt(x); }
static inline float  norm3df(float a, float b, float c)  { return std::sqrt(a * a + b * b + c * c); }
static inline float  rnorm3df(float a, float b, float c) { return 1.0f / std::sqrt(a * a + b * b + c * c); }
static inline float  norm4df(float a, float b, float c, float d)  { return std::sqrt(a * a + b * b + c * c + d * d); }
static inline float  rnorm4df(float a, float b, float c, float d) { return 1.0f / std::sqrt(a * a + b * b + c * c + d * d); }
static inline float  normf(int dim, const float* a) { float s = 0.0f; for (int i = 0; i < dim; ++i) s += a[i] * a[i]; return std::sqrt(s); }
static inline float  rnormf(int dim, const float* a) { return 1.0f / normf(dim, a); }
static inline float  rhypotf(float a, float b) { return 1.0f / std::hypot(a, b); }
static inline void   sincospif(float x, float* s, float* c) { *s = std::sin(x * 3.14159265358979323846f); *c = std::cos(x * 3.14159265358979323846f); }
static inline float  sinpif(float x) { return std::sin(x * 3.14159265358979323846f); }
static inline float  cospif(float x) { return std::cos(x * 3.14159265358979323846f); }

} // namespace vgre_cuda

#endif // VGRE_COMPILER_CPU_CUDA_INTRINSICS_H
