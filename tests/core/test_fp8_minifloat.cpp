// FP8 minifloat codec (vgre::math::FP8) — verifies the IEEE-style E4M3/E5M2
// conversions are correct, in particular the subnormal band the old code
// flushed to zero. Two strong properties:
//   1. Exact round-trip: decode every one of the 256 bit patterns to fp32 and
//      re-encode — must return the identical code (NaN stays NaN).
//   2. Subnormals decode to their true nonzero values (not 0).

#include "mixed_precision.h"   // src/core/math (added to include path by CMake)

#include <cmath>
#include <cstdint>
#include <cstdio>

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

int main() {
    roundTrip(FP8Format::E4M3, "E4M3 round-trip");
    roundTrip(FP8Format::E5M2, "E5M2 round-trip");

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
