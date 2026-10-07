#ifndef VGRE_COMPILER_CPU_CUDA_FP16_H
#define VGRE_COMPILER_CPU_CUDA_FP16_H

// IEEE-754 FP16: sign(1) | exponent(5, bias=15) | mantissa(10)
// Conversion from/to float32 (sign(1) | exponent(8, bias=127) | mantissa(23))

#include <cstdint>
#include <cmath>
#include <cstring>

namespace vgre_cuda {

enum class FloatRounding { NearestEven, NearestAway, TowardZero, Downward, Upward };

inline uint32_t round_half_significand(double scaled, bool negative,
                                       FloatRounding rounding) {
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    bool increment = false;
    switch (rounding) {
        case FloatRounding::NearestEven:
            increment = fraction > 0.5 || (fraction == 0.5 &&
                         (static_cast<uint32_t>(lower) & 1u) != 0);
            break;
        case FloatRounding::NearestAway:
            increment = fraction >= 0.5;
            break;
        case FloatRounding::TowardZero:
            break;
        case FloatRounding::Downward:
            increment = negative && fraction != 0.0;
            break;
        case FloatRounding::Upward:
            increment = !negative && fraction != 0.0;
            break;
    }
    return static_cast<uint32_t>(lower) + static_cast<uint32_t>(increment);
}

// IEEE binary32 -> binary16 conversion with an explicit rounding direction.
// Scaling by powers of two keeps every binary32 input exactly representable in
// the wider double intermediate, including subnormal and midpoint cases.
inline uint16_t f32_to_f16_bits(float value, FloatRounding rounding) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    const uint32_t fraction = bits & 0x7FFFFFu;

    if (exponent == 0xFFu) {
        if (fraction == 0) return static_cast<uint16_t>(sign | 0x7C00u);
        uint16_t payload = static_cast<uint16_t>(fraction >> 13);
        payload = static_cast<uint16_t>(payload | 0x0200u); // quiet NaN
        return static_cast<uint16_t>(sign | 0x7C00u | payload);
    }

    const double magnitude = std::fabs(static_cast<double>(value));
    if (magnitude == 0.0) return sign;

    constexpr double kMaxFinite = 65504.0;
    constexpr double kOverflowMidpoint = 65520.0;
    if (magnitude > kMaxFinite) {
        bool toInfinity = false;
        switch (rounding) {
            case FloatRounding::NearestEven:
            case FloatRounding::NearestAway:
                toInfinity = magnitude >= kOverflowMidpoint;
                break;
            case FloatRounding::TowardZero:
                break;
            case FloatRounding::Downward:
                toInfinity = sign != 0;
                break;
            case FloatRounding::Upward:
                toInfinity = sign == 0;
                break;
        }
        if (toInfinity) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | 0x7BFFu);
    }

    if (magnitude < 0x1p-14) {
        const uint32_t subnormal = round_half_significand(
            std::ldexp(magnitude, 24), sign != 0, rounding);
        return static_cast<uint16_t>(sign | subnormal);
    }

    const int exponent2 = std::ilogb(magnitude);
    uint32_t significand = round_half_significand(
        std::ldexp(magnitude, 10 - exponent2), sign != 0, rounding);
    int outputExponent = exponent2;
    if (significand == 0x800u) {
        significand = 0x400u;
        ++outputExponent;
    }
    if (outputExponent > 15) return static_cast<uint16_t>(sign | 0x7C00u);
    const uint16_t exponent16 = static_cast<uint16_t>(outputExponent + 15);
    const uint16_t mantissa16 = static_cast<uint16_t>(significand - 0x400u);
    return static_cast<uint16_t>(sign | (exponent16 << 10) | mantissa16);
}

struct __half {
    uint16_t __x;

    __half() = default;

    __half(float f) : __x(f32_to_f16_bits(f, FloatRounding::NearestEven)) {}

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
        memcpy(&result, &fi, 4);
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
