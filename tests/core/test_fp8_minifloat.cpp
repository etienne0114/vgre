// FP8 minifloat codec (vgre::math::FP8) — verifies the IEEE-style E4M3/E5M2
// conversions are correct, in particular the subnormal band the old code
// flushed to zero. Two strong properties:
//   1. Exact round-trip: decode every one of the 256 bit patterns to fp32 and
//      re-encode — must return the identical code (NaN stays NaN).
//   2. Subnormals decode to their true nonzero values (not 0).

#include "mixed_precision.h"   // src/core/math (added to include path by CMake)
#include "vgre/compiler/wmma_emulation.h" // tensor-core E4M3FN/E5M2 codecs

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

using vgre::math::FP8;
using vgre::math::FP8Format;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static bool isNanCode(uint8_t c, FP8Format fmt) {
    const int M = (fmt == FP8Format::E4M3) ? 3 : 2;
    const int E = (fmt == FP8Format::E4M3) ? 4 : 5;
    const uint32_t be_max = (1u << E) - 1u;
    return (((c >> M) & ((1u << E) - 1u)) == be_max) && ((c & ((1u << M) - 1u)) != 0u);
}

static void roundTrip(FP8Format fmt, const char* name) {
    int mismatches = 0, subnormalsSeen = 0;
    for (int i = 0; i < 256; ++i) {
        const uint8_t code = (uint8_t)i;
        FP8 a((uint8_t)code, fmt);
        float fv = a.to_float();
        FP8 b = FP8::from_float(fv, fmt);
        if (isNanCode(code, fmt)) {
            // NaN must decode to a NaN float and re-encode to *some* NaN code.
            if (!std::isnan(fv) || !isNanCode(b.bits, fmt)) ++mismatches;
            continue;
        }
        if (b.bits != code) {
            std::printf("  %s: code 0x%02X -> %.9g -> 0x%02X\n", name, code, fv, b.bits);
            ++mismatches;
        }
        // count subnormals (biased exp 0, nonzero frac) and confirm nonzero value
        const int M = (fmt == FP8Format::E4M3) ? 3 : 2;
        const int E = (fmt == FP8Format::E4M3) ? 4 : 5;
        bool sub = (((code >> M) & ((1u << E) - 1u)) == 0u) && ((code & ((1u << M) - 1u)) != 0u);
        if (sub) { ++subnormalsSeen; CHECK(fv != 0.0f, "subnormal decodes to nonzero"); }
    }
    CHECK(mismatches == 0, name);
    CHECK(subnormalsSeen > 0, "format has a represented subnormal band");
}

static void tensorCoreRoundTrip(bool e4m3) {
    int mismatches = 0;
    for (unsigned i = 0; i < 256; ++i) {
        const uint8_t code = static_cast<uint8_t>(i);
        const float decoded = e4m3 ? detail::fp8e4m3_to_f32(code)
                                   : detail::fp8e5m2_to_f32(code);
        const unsigned sign = code >> 7;
        const unsigned exponent = e4m3 ? ((code >> 3) & 0x0Fu) : ((code >> 2) & 0x1Fu);
        const unsigned mantissa = e4m3 ? (code & 0x07u) : (code & 0x03u);
        const bool special = e4m3 ? (exponent == 0x0Fu && mantissa == 7u)
                                  : (exponent == 0x1Fu);
        float reference;
        if (special && (e4m3 || mantissa != 0)) {
            reference = std::numeric_limits<float>::quiet_NaN();
        } else if (special) {
            reference = std::numeric_limits<float>::infinity();
        } else if (exponent == 0) {
            reference = std::ldexp(static_cast<float>(mantissa), e4m3 ? -9 : -16);
        } else {
            const float significand = e4m3 ? (1.0f + static_cast<float>(mantissa) / 8.0f)
                                          : (1.0f + static_cast<float>(mantissa) / 4.0f);
            reference = std::ldexp(significand,
                static_cast<int>(exponent) - (e4m3 ? 7 : 15));
        }
        if (sign) reference = -reference;
        if ((std::isnan(decoded) && !std::isnan(reference)) ||
            (!std::isnan(decoded) && decoded != reference)) {
            std::printf("  tensor-core %s decode: 0x%02X -> %.9g, expected %.9g\n",
                        e4m3 ? "E4M3" : "E5M2", code, decoded, reference);
            ++mismatches;
        }
        const uint8_t encoded = e4m3 ? detail::f32_to_fp8e4m3(decoded)
                                     : detail::f32_to_fp8e5m2(decoded);
        const bool nan = std::isnan(decoded);
        if ((nan && !std::isnan(e4m3 ? detail::fp8e4m3_to_f32(encoded)
                                    : detail::fp8e5m2_to_f32(encoded))) ||
            (!nan && encoded != code)) {
            std::printf("  tensor-core %s: 0x%02X -> %.9g -> 0x%02X\n",
                        e4m3 ? "E4M3" : "E5M2", code, decoded, encoded);
            ++mismatches;
        }
    }
    CHECK(mismatches == 0, e4m3 ? "tensor-core E4M3 byte round-trip"
                                : "tensor-core E5M2 byte round-trip");
}

static void tensorCoreMidpointRounding(bool e4m3) {
    const unsigned lastFinite = e4m3 ? 0x7Eu : 0x7Bu;
    int rneFailures = 0;
    int rtzFailures = 0;
    for (unsigned lower = 0; lower < lastFinite; ++lower) {
        const float a = e4m3 ? detail::fp8e4m3_to_f32(static_cast<uint8_t>(lower))
                             : detail::fp8e5m2_to_f32(static_cast<uint8_t>(lower));
        const float b = e4m3 ? detail::fp8e4m3_to_f32(static_cast<uint8_t>(lower + 1))
                             : detail::fp8e5m2_to_f32(static_cast<uint8_t>(lower + 1));
        const float midpoint = (a + b) * 0.5f;
        const uint8_t expected = static_cast<uint8_t>((lower & 1u) ? lower + 1 : lower);
        const uint8_t positive = e4m3 ? detail::f32_to_fp8e4m3(midpoint)
                                      : detail::f32_to_fp8e5m2(midpoint);
        const uint8_t negative = e4m3 ? detail::f32_to_fp8e4m3(-midpoint)
                                      : detail::f32_to_fp8e5m2(-midpoint);
        const uint8_t towardZeroPositive = e4m3
            ? detail::f32_to_fp8e4m3_satfinite(midpoint, true)
            : detail::f32_to_fp8e5m2_satfinite(midpoint, true);
        const uint8_t towardZeroNegative = e4m3
            ? detail::f32_to_fp8e4m3_satfinite(-midpoint, true)
            : detail::f32_to_fp8e5m2_satfinite(-midpoint, true);
        if (positive != expected || negative != static_cast<uint8_t>(expected | 0x80u))
            ++rneFailures;
        if (towardZeroPositive != lower ||
            towardZeroNegative != static_cast<uint8_t>(lower | 0x80u))
            ++rtzFailures;
    }
    CHECK(rneFailures == 0, e4m3 ? "all E4M3 adjacent-value midpoints use ties-to-even"
                                 : "all E5M2 adjacent-value midpoints use ties-to-even");
    CHECK(rtzFailures == 0, e4m3 ? "all E4M3 midpoints truncate toward zero"
                                 : "all E5M2 midpoints truncate toward zero");
}

static void tensorCoreRoundingAndSaturation() {
    using detail::f32_to_fp8e4m3;
    using detail::f32_to_fp8e5m2;
    using detail::f32_to_fp8e4m3_satfinite;
    using detail::f32_to_fp8e5m2_satfinite;

    CHECK(f32_to_fp8e4m3(1.0625f) == 0x38, "tensor-core E4M3 ties-to-even lower");
    CHECK(f32_to_fp8e4m3(1.1875f) == 0x3A, "tensor-core E4M3 ties-to-even upper");
    CHECK(f32_to_fp8e5m2(1.125f) == 0x3C, "tensor-core E5M2 ties-to-even lower");
    CHECK(f32_to_fp8e5m2(1.375f) == 0x3E, "tensor-core E5M2 ties-to-even upper");
    CHECK(f32_to_fp8e4m3_satfinite(1.1875f, true) == 0x39,
          "tensor-core E4M3 round-toward-zero conversion");
    CHECK(f32_to_fp8e5m2_satfinite(1.375f, true) == 0x3D,
          "tensor-core E5M2 round-toward-zero conversion");
    CHECK(f32_to_fp8e4m3(std::ldexp(1.0f, -10)) == 0x00,
          "tensor-core E4M3 zero/min-subnormal midpoint ties to zero");
    CHECK(f32_to_fp8e4m3(3.0f * std::ldexp(1.0f, -10)) == 0x02,
          "tensor-core E4M3 subnormal midpoint ties to even");
    CHECK(f32_to_fp8e5m2(std::ldexp(1.0f, -17)) == 0x00,
          "tensor-core E5M2 zero/min-subnormal midpoint ties to zero");
    CHECK(f32_to_fp8e5m2(3.0f * std::ldexp(1.0f, -17)) == 0x02,
          "tensor-core E5M2 subnormal midpoint ties to even");

    CHECK(detail::fp8e4m3_to_f32(0x7E) == 448.0f, "E4M3 maximum finite decode");
    CHECK(detail::fp8e5m2_to_f32(0x7B) == 57344.0f, "E5M2 maximum finite decode");
    CHECK(f32_to_fp8e4m3(1000.0f) == 0x7E, "E4M3 clamps overflow to maximum finite");
    CHECK(f32_to_fp8e4m3(-1000.0f) == 0xFE, "E4M3 preserves sign when clamping");
    CHECK(f32_to_fp8e5m2(61440.0f) == 0x7C, "E5M2 RNE overflow rounds to infinity");
    CHECK(f32_to_fp8e5m2_satfinite(100000.0f) == 0x7B,
          "E5M2 satfinite clamps overflow to maximum finite");
    CHECK(f32_to_fp8e5m2_satfinite(-std::numeric_limits<float>::infinity()) == 0xFB,
          "E5M2 satfinite clamps negative infinity and preserves sign");
    CHECK(f32_to_fp8e4m3_satfinite(-1.0f, false, true) == 0x00,
          "E4M3 ReLU clamps negative values to positive zero");
    CHECK(f32_to_fp8e5m2_satfinite(-std::numeric_limits<float>::infinity(), false, true) == 0x00,
          "E5M2 ReLU clamps negative infinity to positive zero");
    CHECK(std::isnan(detail::fp8e4m3_to_f32(f32_to_fp8e4m3_satfinite(
              std::numeric_limits<float>::quiet_NaN(), false, true))),
          "E4M3 ReLU preserves NaN");
    CHECK(f32_to_fp8e4m3_satfinite(std::numeric_limits<float>::infinity()) == 0x7E,
          "E4M3 satfinite maps infinity to maximum finite");
    CHECK(f32_to_fp8e5m2(std::numeric_limits<float>::infinity()) == 0x7C,
          "E5M2 preserves infinity without satfinite");
    CHECK(std::isnan(detail::fp8e4m3_to_f32(f32_to_fp8e4m3(
              std::numeric_limits<float>::quiet_NaN()))), "E4M3 NaN round-trip");
    CHECK(std::isnan(detail::fp8e5m2_to_f32(f32_to_fp8e5m2_satfinite(
              std::numeric_limits<float>::quiet_NaN()))), "E5M2 satfinite preserves NaN");
}

int main() {
    roundTrip(FP8Format::E4M3, "E4M3 round-trip");
    roundTrip(FP8Format::E5M2, "E5M2 round-trip");
    tensorCoreRoundTrip(true);
    tensorCoreRoundTrip(false);
    tensorCoreMidpointRounding(true);
    tensorCoreMidpointRounding(false);
    tensorCoreRoundingAndSaturation();

    // Exact smallest positive subnormals (the case the old code returned 0 for).
    // E4M3: 2^(1-bias-M) = 2^(1-7-3) = 2^-9.  E5M2: 2^(1-15-2) = 2^-16.
    {
        float e4 = FP8((uint8_t)0x01, FP8Format::E4M3).to_float();
        float e5 = FP8((uint8_t)0x01, FP8Format::E5M2).to_float();
        CHECK(std::fabs(e4 - std::ldexp(1.0f, -9)) < 1e-12f, "E4M3 min subnormal == 2^-9");
        CHECK(std::fabs(e5 - std::ldexp(1.0f, -16)) < 1e-15f, "E5M2 min subnormal == 2^-16");
        CHECK(FP8::from_float(std::ldexp(1.0f, -9), FP8Format::E4M3).bits == 0x01,
              "E4M3 encodes 2^-9 to the min subnormal code");
        CHECK(FP8::from_float(std::ldexp(1.0f, -16), FP8Format::E5M2).bits == 0x01,
              "E5M2 encodes 2^-16 to the min subnormal code");
    }

    // Round-to-nearest-even on narrowing: a value exactly between two E4M3
    // codes rounds to the one with an even last bit.
    {
        // 1.0 and the next E4M3 up is 1.125 (1 + 1/8); midpoint 1.0625 → ties to
        // even → 1.0 (code 0x38, frac 000, even).
        FP8 r = FP8::from_float(1.0625f, FP8Format::E4M3);
        CHECK(r.bits == 0x38, "E4M3 RNE: 1.0625 ties down to 1.0 (even)");
    }

    if (g_fail == 0) { std::printf("test_fp8_minifloat: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_fp8_minifloat: %d FAILURE(S)\n", g_fail);
    return 1;
}
