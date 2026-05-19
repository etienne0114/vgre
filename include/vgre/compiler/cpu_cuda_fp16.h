#ifndef VGRE_COMPILER_CPU_CUDA_FP16_H
#define VGRE_COMPILER_CPU_CUDA_FP16_H

// IEEE-754 FP16: sign(1) | exponent(5, bias=15) | mantissa(10)
// Conversion from/to float32 (sign(1) | exponent(8, bias=127) | mantissa(23))

#include <cstdint>
#include <cmath>

namespace vgre_cuda {

struct __half {
    uint16_t __x;

    __half() = default;

    __half(float f) {
        uint32_t fi;
        std::memcpy(&fi, &f, 4);
        uint16_t sign  = (fi >> 31) & 0x1u;
        int32_t  exp32 = ((fi >> 23) & 0xFF) - 127;  // unbiased exponent
        uint32_t mant  = fi & 0x7FFFFFu;              // 23-bit mantissa

        uint16_t exp16, mant16;
        if (exp32 >= 16) {
            // Overflow → infinity
            exp16  = 0x1F;
            mant16 = 0;
        } else if (exp32 >= -14) {
            // Normalized range
            exp16  = static_cast<uint16_t>(exp32 + 15);
            // Round-to-nearest-even: shift 23-bit mantissa to 10-bit
            uint32_t shifted = mant + 0x1000u;  // round bit + guard bits
            if (shifted & 0x800000u) { ++exp16; shifted = 0; }
            mant16 = static_cast<uint16_t>(shifted >> 13);
        } else if (exp32 >= -24) {
            // Denormalized: place mantissa at correct position
            int shift = -exp32 - 14 + 13;
            uint32_t m = (mant | 0x800000u);  // add implicit leading 1
            mant16 = static_cast<uint16_t>(m >> shift);
            exp16  = 0;
        } else {
            // Underflow → zero (signed)
            exp16  = 0;
            mant16 = 0;
        }
        __x = static_cast<uint16_t>((sign << 15) | (exp16 << 10) | mant16);
    }

    operator float() const {
        uint32_t sign  = (__x >> 15) & 0x1u;
        uint32_t exp16 = (__x >> 10) & 0x1Fu;
        uint32_t mant  = __x & 0x3FFu;

        uint32_t fi;
        if (exp16 == 0x1F) {
            // Infinity or NaN
            fi = (sign << 31) | (0xFF << 23) | (mant << 13);
        } else if (exp16 == 0) {
            // Denormalized → normalized float
            if (mant == 0) {
                fi = sign << 31;  // +/- zero
            } else {
                int shift = 0;
                uint32_t m = mant;
                while (!(m & 0x400u)) { m <<= 1; ++shift; }
                m &= 0x3FFu;
                uint32_t e = 127 - 14 - shift;
                fi = (sign << 31) | (e << 23) | (m << 13);
            }
        } else {
            uint32_t e = exp16 - 15 + 127;
            fi = (sign << 31) | (e << 23) | (mant << 13);
        }
        float result;
        std::memcpy(&result, &fi, 4);
        return result;
    }
};

// ── Conversion helpers ────────────────────────────────────────────────────────
inline __half  __float2half(float f)       { return __half(f); }
inline float   __half2float(__half h)      { return static_cast<float>(h); }
inline __half  __float2half_rn(float f)    { return __half(f); }  // round-to-nearest
inline float   __half2float_rn(__half h)   { return static_cast<float>(h); }

// Integer conversions
inline __half  __int2half_rn(int i)        { return __half(static_cast<float>(i)); }
inline int     __half2int_rn(__half h)     { return static_cast<int>(static_cast<float>(h)); }

// ── Arithmetic (round-trip through float32) ──────────────────────────────────
inline __half __hadd (__half a, __half b)  { return __half(float(a) + float(b)); }
inline __half __hsub (__half a, __half b)  { return __half(float(a) - float(b)); }
inline __half __hmul (__half a, __half b)  { return __half(float(a) * float(b)); }
inline __half __hdiv (__half a, __half b)  { return __half(float(a) / float(b)); }
inline __half __hneg (__half a)            { return __half(-float(a)); }
inline __half __habs (__half a)            { return __half(fabsf(float(a))); }
inline __half __hfma (__half a, __half b, __half c) { return __half(fmaf(float(a), float(b), float(c))); }

// ── Comparison (return bool) ─────────────────────────────────────────────────
inline bool __heq (__half a, __half b)     { return float(a) == float(b); }
inline bool __hne (__half a, __half b)     { return float(a) != float(b); }
inline bool __hlt (__half a, __half b)     { return float(a) <  float(b); }
inline bool __hle (__half a, __half b)     { return float(a) <= float(b); }
inline bool __hgt (__half a, __half b)     { return float(a) >  float(b); }
inline bool __hge (__half a, __half b)     { return float(a) >= float(b); }

// ── Math functions ───────────────────────────────────────────────────────────
inline __half __hexp  (__half a)           { return __half(expf(float(a))); }
inline __half __hlog  (__half a)           { return __half(logf(float(a))); }
inline __half __hsqrt (__half a)           { return __half(sqrtf(float(a))); }
inline __half __hrsqrt(__half a)           { return __half(1.0f / sqrtf(float(a))); }
inline __half __hsin  (__half a)           { return __half(sinf(float(a))); }
inline __half __hcos  (__half a)           { return __half(cosf(float(a))); }
inline __half hmax    (__half a, __half b) { return float(a) >= float(b) ? a : b; }
inline __half hmin    (__half a, __half b) { return float(a) <= float(b) ? a : b; }

// ── Vector types ─────────────────────────────────────────────────────────────
struct __half2 { __half x, y; };

inline __half2 __floats2half2_rn(float a, float b) { return {__half(a), __half(b)}; }
inline void    __half22float2(float& a, float& b, __half2 v) { a = float(v.x); b = float(v.y); }
inline __half2 __hadd2  (__half2 a, __half2 b) { return {__hadd(a.x,b.x), __hadd(a.y,b.y)}; }
inline __half2 __hmul2  (__half2 a, __half2 b) { return {__hmul(a.x,b.x), __hmul(a.y,b.y)}; }
inline __half2 __hfma2  (__half2 a, __half2 b, __half2 c) {
    return {__hfma(a.x,b.x,c.x), __hfma(a.y,b.y,c.y)};
}

// ── Operator overloads ────────────────────────────────────────────────────────
inline __half operator+(__half a, __half b) { return __hadd(a, b); }
inline __half operator-(__half a, __half b) { return __hsub(a, b); }
inline __half operator*(__half a, __half b) { return __hmul(a, b); }
inline __half operator/(__half a, __half b) { return __hdiv(a, b); }
inline __half& operator+=(__half& a, __half b) { a = __hadd(a, b); return a; }
inline __half& operator-=(__half& a, __half b) { a = __hsub(a, b); return a; }
inline __half& operator*=(__half& a, __half b) { a = __hmul(a, b); return a; }
inline __half& operator/=(__half& a, __half b) { a = __hdiv(a, b); return a; }
inline bool operator==(__half a, __half b) { return __heq(a, b); }
inline bool operator!=(__half a, __half b) { return __hne(a, b); }
inline bool operator< (__half a, __half b) { return __hlt(a, b); }
inline bool operator<=(__half a, __half b) { return __hle(a, b); }
inline bool operator> (__half a, __half b) { return __hgt(a, b); }
inline bool operator>=(__half a, __half b) { return __hge(a, b); }

// Make __half usable with printf %g etc. via implicit float conversion:
// float(h) already works since operator float() is defined above.

} // namespace vgre_cuda

using vgre_cuda::__half;
using vgre_cuda::__half2;

#endif // VGRE_COMPILER_CPU_CUDA_FP16_H
