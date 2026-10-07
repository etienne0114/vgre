#ifndef VGRE_COMPILER_PTX_CONVERSION_H
#define VGRE_COMPILER_PTX_CONVERSION_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cpu_cuda_fp16.h"

namespace vgre_ptx_conversion {

enum class IntegerRounding { NearestEven, TowardZero, Downward, Upward };

inline double round_float_to_integer(double value, IntegerRounding rounding) {
    switch (rounding) {
        case IntegerRounding::TowardZero: return std::trunc(value);
        case IntegerRounding::Downward:   return std::floor(value);
        case IntegerRounding::Upward:     return std::ceil(value);
        case IntegerRounding::NearestEven: {
            const double lower = std::floor(value);
            const double fraction = value - lower;
            if (fraction < 0.5) return lower;
            if (fraction > 0.5) return lower + 1.0;
            return std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
        }
    }
    return std::trunc(value);
}

// PTX float-to-integer conversions round first, then clamp to the destination
// range. This avoids undefined out-of-range C++ casts and host rounding state.
template <typename Destination, typename Source>
inline Destination cvt_float_to_integer(Source value, IntegerRounding rounding) {
    static_assert(std::is_integral<Destination>::value,
                  "PTX float-to-integer destination must be integral");
    static_assert(std::is_floating_point<Source>::value,
                  "PTX float-to-integer source must be floating point");
    constexpr int digits = std::numeric_limits<Destination>::digits;

    if (std::isnan(value)) {
        // PTX specifies zero for ordinary f32 -> <=32-bit conversions. For an
        // f64 source or 64-bit destination it specifies the high-bit pattern.
        if (!std::is_same<Source, double>::value && sizeof(Destination) < sizeof(uint64_t))
            return Destination(0);
        if (std::numeric_limits<Destination>::is_signed)
            return std::numeric_limits<Destination>::lowest();
        return static_cast<Destination>(std::uint64_t(1) << (digits - 1));
    }

    const double rounded = round_float_to_integer(static_cast<double>(value), rounding);
    const double limit = std::ldexp(1.0, digits);
    if (std::numeric_limits<Destination>::is_signed) {
        if (rounded <= -limit) return std::numeric_limits<Destination>::lowest();
        if (rounded >= limit) return std::numeric_limits<Destination>::max();
    } else {
        if (rounded <= 0.0) return Destination(0);
        if (rounded >= limit) return std::numeric_limits<Destination>::max();
    }
    return static_cast<Destination>(rounded);
}

inline vgre_cuda::__half cvt_f32_to_f16(float value, vgre_cuda::FloatRounding rounding) {
    vgre_cuda::__half result;
    result.__x = vgre_cuda::f32_to_f16_bits(value, rounding);
    return result;
}

inline uint32_t round_f32_significand(double scaled, bool negative,
                                     vgre_cuda::FloatRounding rounding) {
    const double lower = std::floor(scaled);
    const double fraction = scaled - lower;
    bool increment = false;
    switch (rounding) {
        case vgre_cuda::FloatRounding::NearestEven:
            increment = fraction > 0.5 || (fraction == 0.5 &&
                         (static_cast<uint32_t>(lower) & 1u) != 0);
            break;
        case vgre_cuda::FloatRounding::NearestAway:
            increment = fraction >= 0.5;
            break;
        case vgre_cuda::FloatRounding::TowardZero:
            break;
        case vgre_cuda::FloatRounding::Downward:
            increment = negative && fraction != 0.0;
            break;
        case vgre_cuda::FloatRounding::Upward:
            increment = !negative && fraction != 0.0;
            break;
    }
    return static_cast<uint32_t>(lower) + static_cast<uint32_t>(increment);
}

// Binary64 -> binary32 conversion with explicit IEEE rounding. The scaled
// significand remains exact in binary64, so directed rounding is independent
// of the process-wide floating-point environment on every supported OS.
inline float cvt_f64_to_f32(double value, vgre_cuda::FloatRounding rounding) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = static_cast<uint32_t>((bits >> 32) & 0x80000000u);
    const uint64_t exponent = (bits >> 52) & 0x7FFu;
    const uint64_t fraction = bits & 0x000FFFFFFFFFFFFFull;

    uint32_t outputBits;
    if (exponent == 0x7FFu) {
        if (fraction == 0) {
            outputBits = sign | 0x7F800000u;
        } else {
            uint32_t payload = static_cast<uint32_t>(fraction >> 29) | 0x00400000u;
            outputBits = sign | 0x7F800000u | payload;
        }
        float result;
        std::memcpy(&result, &outputBits, sizeof(result));
        return result;
    }

    const double magnitude = std::fabs(value);
    if (magnitude == 0.0) {
        float result;
        std::memcpy(&result, &sign, sizeof(result));
        return result;
    }

    constexpr double kMaxFinite = static_cast<double>(std::numeric_limits<float>::max());
    const double kOverflowMidpoint = kMaxFinite + std::ldexp(1.0, 103);
    if (magnitude > kMaxFinite) {
        bool toInfinity = false;
        switch (rounding) {
            case vgre_cuda::FloatRounding::NearestEven:
            case vgre_cuda::FloatRounding::NearestAway:
                toInfinity = magnitude >= kOverflowMidpoint;
                break;
            case vgre_cuda::FloatRounding::TowardZero:
                break;
            case vgre_cuda::FloatRounding::Downward:
                toInfinity = sign != 0;
                break;
            case vgre_cuda::FloatRounding::Upward:
                toInfinity = sign == 0;
                break;
        }
        outputBits = sign | (toInfinity ? 0x7F800000u : 0x7F7FFFFFu);
        float result;
        std::memcpy(&result, &outputBits, sizeof(result));
        return result;
    }

    if (magnitude < 0x1p-126) {
        outputBits = sign | round_f32_significand(
            std::ldexp(magnitude, 149), sign != 0, rounding);
    } else {
        const int exponent2 = std::ilogb(magnitude);
        uint32_t significand = round_f32_significand(
            std::ldexp(magnitude, 23 - exponent2), sign != 0, rounding);
        int outputExponent = exponent2;
        if (significand == 0x1000000u) {
            significand = 0x800000u;
            ++outputExponent;
        }
        if (outputExponent > 127) {
            outputBits = sign | 0x7F800000u;
        } else {
            outputBits = sign |
                (static_cast<uint32_t>(outputExponent + 127) << 23) |
                (significand - 0x800000u);
        }
    }
    float result;
    std::memcpy(&result, &outputBits, sizeof(result));
    return result;
}

inline float cvt_sat_f32(float value) {
    if (std::isnan(value) || value < 0.0f) return 0.0f;
    return value > 1.0f ? 1.0f : value;
}

inline vgre_cuda::__half cvt_sat_f32_to_f16(float value) {
    return cvt_f32_to_f16(cvt_sat_f32(value), vgre_cuda::FloatRounding::NearestEven);
}

} // namespace vgre_ptx_conversion

#endif // VGRE_COMPILER_PTX_CONVERSION_H
