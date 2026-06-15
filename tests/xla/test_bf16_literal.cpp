// bf16/f16 native Literal storage (the rest of milestone L1): weights can be
// held at native 16-bit width — half the RAM of fp32 — and the engine
// decompresses them to f32 transiently as each op consumes them. Verifies the
// codec, the storage saving, and that a real matmul fed bf16 parameters matches
// the fp32 reference within bf16 tolerance.

#include "vgre/xla/hlo.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::xla;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static double relErr(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs((double)a[i] - b[i]) / (std::fabs((double)b[i]) + 1e-6));
    return m;
}

int main() {
    // ── Codec + storage ─────────────────────────────────────────────────────
    {
        Literal f = Literal::make({{4}}, {1.0f, -2.0f, 0.5f, 3.14159f});
        CHECK(f.storageBytes() == 16, "f32 storage = 4*4 bytes");

        Literal b = f.toBF16();
        CHECK(b.dtype == DType::BF16, "toBF16 sets dtype");
        CHECK(b.storageBytes() == 8, "bf16 storage halved (4*2 bytes)");
        CHECK(b.shape.dims == f.shape.dims, "shape preserved");
        // exact-in-bf16 values round-trip exactly; pi is approximate.
        CHECK(b.at(0) == 1.0f && b.at(1) == -2.0f && b.at(2) == 0.5f, "bf16 exact values");
        CHECK(std::fabs(b.at(3) - 3.14159f) < 0.02f, "bf16 pi within ~1%");

        Literal h = f.toF16();
        CHECK(h.dtype == DType::F16 && h.storageBytes() == 8, "f16 storage halved");
        CHECK(h.at(0) == 1.0f && h.at(2) == 0.5f, "f16 exact values");
        CHECK(std::fabs(h.at(3) - 3.14159f) < 0.001f, "f16 pi within ~0.05%");

        // toF32 of a bf16 literal reproduces the dequantized data.
        Literal back = b.toF32();
        CHECK(back.dtype == DType::F32 && back.data.size() == 4, "toF32 from bf16");
        CHECK(back.data[0] == 1.0f && back.data[2] == 0.5f, "toF32 values");
    }

    // ── Engine: matmul with bf16 parameters matches fp32 reference ──────────
    {
        const int M = 4, K = 8, N = 4;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> A(M * K), B(K * N);
        for (auto& v : A) v = d(rng);
        for (auto& v : B) v = d(rng);

        HloModule m;
        int a = m.parameter(0, Shape{{M, K}});
        int b = m.parameter(1, Shape{{K, N}});
        m.dot(a, b);

        Literal Af = Literal::make({{M, K}}, A);
        Literal Bf = Literal::make({{K, N}}, B);
        Literal ref = m.evaluate({Af, Bf});                      // fp32 reference

        // Same graph, parameters supplied as bf16 (half the bytes).
        Literal Ab = Af.toBF16();
        Literal Bb = Bf.toBF16();
        CHECK(Ab.storageBytes() * 2 == Af.storageBytes(), "bf16 param A is half the bytes");
        Literal got = m.evaluate({Ab, Bb});

        CHECK(got.data.size() == (size_t)(M * N), "bf16-param matmul produced full output");
        double e = relErr(got.data, ref.data);
        std::printf("bf16-parameter matmul vs fp32: max rel err = %.4f\n", e);
        CHECK(e < 0.02, "bf16-parameter matmul within 2% of fp32");

        // Mixed: one bf16 param, one fp32 param (decompression is per-parameter).
        Literal mixed = m.evaluate({Ab, Bf});
        CHECK(relErr(mixed.data, ref.data) < 0.02, "mixed bf16/fp32 params within 2%");
    }

    if (g_fail == 0) { std::printf("test_bf16_literal: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_bf16_literal: %d FAILURE(S)\n", g_fail);
    return 1;
}
