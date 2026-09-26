// Mixed-precision module (vgre::math) + its C API — proves every method is
// wired and functioning:
//   * FP16 / BF16 batch converters (runtime-dispatched AVX-512 vs scalar match)
//   * mixedPrecisionMatmul for FP16 / BF16 / FP8 vs an independent fp32 reference
//   * affine quantize / dequantize round-trip
//   * INT4 pack / unpack
//   * the public C API (vgre_convert_precision / _quantize_affine /
//     _dequantize_affine / _mixed_precision_gemm) that host, device and cluster
//     paths call.

#include "mixed_precision.h"          // src/core/math (added to include path by CMake)
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/simd_dispatch.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vgre::math;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// Independent, obviously-correct fp32 GEMM reference: C[m×k] = A[m×n]·B[n×k].
static void refGemm(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, size_t m, size_t n, size_t k) {
    C.assign(m * k, 0.f);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t p = 0; p < k; ++p)
                C[i * k + p] += A[i * n + j] * B[j * k + p];
}

static bool close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

int main() {
    // ── 1. FP16 batch converters round-trip + AVX-512-path == scalar-path ────
    {
        const size_t N = 37;  // deliberately not a multiple of 16 (exercise tail)
        std::vector<float> in(N), back(N);
        std::vector<FP16>  h(N);
        for (size_t i = 0; i < N; ++i) in[i] = std::ldexp(1.0f, -3) * (float)((int)i - 18);
        float_to_fp16(in.data(), h.data(), N);
        fp16_to_float(h.data(), back.data(), N);
        for (size_t i = 0; i < N; ++i)
            CHECK(close(in[i], back[i], 1e-2f), "fp16 batch round-trip");

        // When the CPU has AVX-512, the vectorized kernel must match the scalar codec.
        if (vgre::simd::have_avx512()) {
            std::vector<float> vref(N);
            for (size_t i = 0; i < N; ++i) vref[i] = h[i].to_float();
            std::vector<float> vavx(N);
            fp16_to_float_avx512(h.data(), vavx.data(), N);
            for (size_t i = 0; i < N; ++i)
                CHECK(vref[i] == vavx[i], "fp16_to_float_avx512 == scalar codec");
        }
    }

    // ── 2. BF16 batch converters round-trip ─────────────────────────────────
    {
        const size_t N = 33;
        std::vector<float> in(N), back(N);
        std::vector<BF16>  b(N);
        for (size_t i = 0; i < N; ++i) in[i] = 0.25f * (float)((int)i - 16);
        float_to_bf16(in.data(), b.data(), N);
        bf16_to_float(b.data(), back.data(), N);
        for (size_t i = 0; i < N; ++i)
            CHECK(close(in[i], back[i], 2e-2f), "bf16 batch round-trip");
    }

    // ── 3. mixedPrecisionMatmul (FP16 / BF16 / FP8) vs fp32 reference ────────
    {
        const size_t m = 3, n = 4, k = 5;
        std::vector<float> Af(m * n), Bf(n * k), Cref;
        for (size_t i = 0; i < Af.size(); ++i) Af[i] = 0.5f * (float)((int)(i % 7) - 3);
        for (size_t i = 0; i < Bf.size(); ++i) Bf[i] = 0.25f * (float)((int)(i % 5) - 2);
        refGemm(Af, Bf, Cref, m, n, k);

        // FP16
        {
            std::vector<FP16> A(m * n), B(n * k);
            for (size_t i = 0; i < A.size(); ++i) A[i] = FP16::from_float(Af[i]);
            for (size_t i = 0; i < B.size(); ++i) B[i] = FP16::from_float(Bf[i]);
            std::vector<float> C(m * k, 0.f);
            mixedPrecisionMatmul<FP16, float, float>(A.data(), B.data(), C.data(), m, n, k, n, k, k);
            for (size_t i = 0; i < C.size(); ++i)
                CHECK(close(C[i], Cref[i], 0.1f), "mixedPrecisionMatmul<FP16>");
        }
        // BF16 (coarser tolerance — 8-bit mantissa)
        {
            std::vector<BF16> A(m * n), B(n * k);
            for (size_t i = 0; i < A.size(); ++i) A[i] = BF16::from_float(Af[i]);
            for (size_t i = 0; i < B.size(); ++i) B[i] = BF16::from_float(Bf[i]);
            std::vector<float> C(m * k, 0.f);
            mixedPrecisionMatmul<BF16, float, float>(A.data(), B.data(), C.data(), m, n, k, n, k, k);
            for (size_t i = 0; i < C.size(); ++i)
                CHECK(close(C[i], Cref[i], 0.3f), "mixedPrecisionMatmul<BF16>");
        }
        // FP8 E4M3
        {
            std::vector<FP8> A(m * n), B(n * k);
            for (size_t i = 0; i < A.size(); ++i) A[i] = FP8::from_float(Af[i], FP8Format::E4M3);
            for (size_t i = 0; i < B.size(); ++i) B[i] = FP8::from_float(Bf[i], FP8Format::E4M3);
            std::vector<float> C(m * k, 0.f);
            mixedPrecisionMatmul<FP8, float, float>(A.data(), B.data(), C.data(), m, n, k, n, k, k);
            for (size_t i = 0; i < C.size(); ++i)
                CHECK(close(C[i], Cref[i], 0.6f), "mixedPrecisionMatmul<FP8>");
        }
    }

    // ── 4. Affine quantize / dequantize round-trip ──────────────────────────
    {
        const size_t N = 64;
        const float scale = 0.05f, zp = 0.f;
        std::vector<float> in(N), out(N);
        std::vector<int8_t> q(N);
        for (size_t i = 0; i < N; ++i) in[i] = scale * (float)((int)i - 32);  // exactly representable
        QuantizationParams<float> qp(scale, zp);
        quantize<float, int8_t>(in.data(), q.data(), N, qp);
        for (size_t i = 0; i < N; ++i) out[i] = (static_cast<float>(q[i]) - zp) * scale;
        for (size_t i = 0; i < N; ++i)
            CHECK(close(in[i], out[i], scale), "affine quantize/dequantize round-trip");
    }

    // ── 5. INT4 pack / unpack (signed nibbles) ──────────────────────────────
    {
        for (int lo = -8; lo < 8; ++lo)
            for (int hi = -8; hi < 8; ++hi) {
                uint8_t packed = INT4::pack((int8_t)lo, (int8_t)hi);
                CHECK(INT4::unpack_low(packed) == (int8_t)lo, "INT4 unpack_low");
                CHECK(INT4::unpack_high(packed) == (int8_t)hi, "INT4 unpack_high");
            }
    }

    // ── 6. C API surface (host / device / cluster entry point) ──────────────
    {
        const size_t N = 20;
        std::vector<float> in(N), back(N);
        std::vector<uint16_t> h(N);
        for (size_t i = 0; i < N; ++i) in[i] = 0.5f * (float)((int)i - 10);
        CHECK(vgre_convert_precision(in.data(), h.data(), N, VGRE_PREC_F32, VGRE_PREC_F16) == 0,
              "vgre_convert_precision f32->f16");
        CHECK(vgre_convert_precision(h.data(), back.data(), N, VGRE_PREC_F16, VGRE_PREC_F32) == 0,
              "vgre_convert_precision f16->f32");
        for (size_t i = 0; i < N; ++i)
            CHECK(close(in[i], back[i], 1e-2f), "C API fp16 round-trip");

        // quantize / dequantize via the C API
        std::vector<int8_t> q(N);
        std::vector<float>  dq(N);
        CHECK(vgre_quantize_affine(in.data(), q.data(), N, 0.5f, 0.f, VGRE_PREC_INT8) == 0,
              "vgre_quantize_affine");
        CHECK(vgre_dequantize_affine(q.data(), dq.data(), N, 0.5f, 0.f, VGRE_PREC_INT8) == 0,
              "vgre_dequantize_affine");
        for (size_t i = 0; i < N; ++i)
            CHECK(close(in[i], dq[i], 0.5f), "C API quantize/dequantize round-trip");

        // reduced-precision GEMM via the C API vs fp32 reference
        const size_t m = 2, nn = 3, k = 2;
        std::vector<float> Af(m * nn), Bf(nn * k), Cref;
        for (size_t i = 0; i < Af.size(); ++i) Af[i] = 0.5f * (float)((int)i - 2);
        for (size_t i = 0; i < Bf.size(); ++i) Bf[i] = 0.5f * (float)((int)i - 1);
        refGemm(Af, Bf, Cref, m, nn, k);
        std::vector<uint16_t> Ah(m * nn), Bh(nn * k);
        vgre_convert_precision(Af.data(), Ah.data(), Af.size(), VGRE_PREC_F32, VGRE_PREC_BF16);
        vgre_convert_precision(Bf.data(), Bh.data(), Bf.size(), VGRE_PREC_F32, VGRE_PREC_BF16);
        std::vector<float> C(m * k, 0.f);
        CHECK(vgre_mixed_precision_gemm(Ah.data(), Bh.data(), C.data(), m, nn, k, VGRE_PREC_BF16) == 0,
              "vgre_mixed_precision_gemm bf16");
        for (size_t i = 0; i < C.size(); ++i)
            CHECK(close(C[i], Cref[i], 0.3f), "C API bf16 GEMM vs fp32 ref");

        // an unsupported pair must be rejected, not silently wrong
        CHECK(vgre_convert_precision(in.data(), h.data(), N, VGRE_PREC_F16, VGRE_PREC_BF16) != 0,
              "C API rejects unsupported conversion pair");
    }

    if (g_fail == 0) std::printf("test_mixed_precision: ALL PASS\n");
    else             std::printf("test_mixed_precision: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
