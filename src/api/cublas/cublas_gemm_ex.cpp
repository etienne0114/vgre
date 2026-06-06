// cuBLAS GemmEx: mixed-precision GEMM (HGEMM, FP16, BF16, INT8, complex)
// Split from cublas_level3.cpp to keep that file under 800 lines.

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"
#include "vgre/runtime/vector_engine.h"
#include <cmath>

extern "C" {

// Forward declarations for SGEMM/DGEMM defined in cublas_level3.cpp
cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);

// ── HGEMM (FP16) — route through FP32 with cast ──────────────────────────────
cublasStatus_t cublasHgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,  // __half*
    const void* A, int lda,
    const void* B, int ldb,
    const void* beta,   // __half*
    void*       C, int ldc)
{
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;
    auto half_to_float = [](uint16_t h) -> float {
        uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                        (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                         (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                        (static_cast<uint32_t>(h & 0x3ff) << 13);
        float f; memcpy(&f, &bits, 4); return f;
    };
    auto float_to_half = [](float f) -> uint16_t {
        uint32_t bits; memcpy(&bits, &f, 4);
        uint16_t sign = (bits >> 16) & 0x8000;
        int exp [[maybe_unused]] = ((bits >> 23) & 0xff) - 127 + 15;
        uint16_t mant = (bits >> 13) & 0x3ff;
        if (exp <= 0) return sign;
        if (exp >= 31) return sign | 0x7c00;
        return sign | (static_cast<uint16_t>(exp) << 10) | mant;
    };

    uint16_t ah, bh;
    memcpy(&ah, alpha, 2); memcpy(&bh, beta, 2);
    float fa = half_to_float(ah), fb = half_to_float(bh);

    const size_t szA = (transa == CUBLAS_OP_N ? m * lda : k * lda);
    const size_t szB = (transb == CUBLAS_OP_N ? k * ldb : n * ldb);
    const size_t szC = static_cast<size_t>(m) * ldc;

    std::vector<float> fA(szA), fB(szB), fC(szC);
    const uint16_t* pA = static_cast<const uint16_t*>(A);
    const uint16_t* pB = static_cast<const uint16_t*>(B);
    uint16_t*       pC = static_cast<uint16_t*>(C);

    for (size_t i = 0; i < szA; ++i) fA[i] = half_to_float(pA[i]);
    for (size_t i = 0; i < szB; ++i) fB[i] = half_to_float(pB[i]);
    for (size_t i = 0; i < szC; ++i) fC[i] = half_to_float(pC[i]);

    bool tA = (transa != CUBLAS_OP_N), tB = (transb != CUBLAS_OP_N);
    refSgemm(tB, tA, n, m, k, fa, fB.data(), ldb, fA.data(), lda, fb, fC.data(), ldc);
    for (size_t i = 0; i < szC; ++i) pC[i] = float_to_half(fC[i]);
    return CUBLAS_STATUS_SUCCESS;
}

// ── cublasGemmEx (mixed-precision GEMM including INT8) ────────────────────────
enum vgre_cudaDataType_ex {
  GEMEX_R_16F = 2, GEMEX_C_16F = 6, GEMEX_R_16BF = 14, GEMEX_C_16BF = 15,
  GEMEX_R_32F = 0, GEMEX_C_32F = 4, GEMEX_R_64F  = 1,  GEMEX_C_64F  = 5,
  GEMEX_R_8I  = 3, GEMEX_C_8I  = 7, GEMEX_R_8U   = 8,  GEMEX_C_8U   = 9,
  GEMEX_R_32I = 10, GEMEX_C_32I = 11,
};

static inline float _i8_to_f32_ex(const void* ptr, int idx) {
    return static_cast<float>(static_cast<const int8_t*>(ptr)[idx]);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo)
{
    (void)algo; (void)computeType;
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;

    if (Atype == (int)GEMEX_R_8I || Btype == (int)GEMEX_R_8I) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        const bool isInt8Both = (Atype == (int)GEMEX_R_8I && Btype == (int)GEMEX_R_8I);
        const bool noTranspose = (transa == CUBLAS_OP_N && transb == CUBLAS_OP_N);

        // Fast path: AVX-VNNI / AMX INT8 direct GEMM for NN non-transposed INT8×INT8.
        // Avoids FP32 heap allocation (~2× m*k + 2× k*n floats) and per-element widening.
        if (isInt8Both && noTranspose && Ctype == (int)GEMEX_R_32I) {
            int32_t* Ci = static_cast<int32_t*>(C);
            std::vector<int32_t> tmp(static_cast<size_t>(m) * n);
            vgre::runtime::VectorEngine::instance().matMulInt8(
                static_cast<const int8_t*>(A),
                static_cast<const int8_t*>(B),
                tmp.data(), m, n, k);
            for (int i = 0; i < m * n; ++i)
                Ci[i] = static_cast<int32_t>(std::round(alphaF * static_cast<float>(tmp[i])
                                                        + betaF  * static_cast<float>(Ci[i])));
            return CUBLAS_STATUS_SUCCESS;
        }

        // General INT8 path (transposed, mixed-type, FP32 output): widen to FP32.
        std::vector<float> Af(m * k), Bf(k * n);
        for (int i = 0; i < m * k; ++i) Af[i] = _i8_to_f32_ex(A, i);
        for (int i = 0; i < k * n; ++i) Bf[i] = _i8_to_f32_ex(B, i);
        if (Ctype == (int)GEMEX_R_32I) {
            std::vector<float> Cf(m * n, 0.f);
            refSgemm(transa, transb, m, n, k, 1.f, Af.data(), lda, Bf.data(), ldb, 0.f, Cf.data(), ldc);
            int32_t* Ci = static_cast<int32_t*>(C);
            for (int i = 0; i < m * n; ++i)
                Ci[i] = static_cast<int32_t>(std::round(alphaF * Cf[i] + betaF * Ci[i]));
        } else {
            refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF,
                     static_cast<float*>(C), ldc);
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    if (Atype == (int)GEMEX_R_32F && Btype == (int)GEMEX_R_32F && Ctype == (int)GEMEX_R_32F)
        return cublasSgemm_v2(handle, transa, transb, m, n, k,
            (const float*)alpha, (const float*)A, lda,
            (const float*)B, ldb, (const float*)beta, (float*)C, ldc);
    if (Atype == (int)GEMEX_R_64F && Btype == (int)GEMEX_R_64F && Ctype == (int)GEMEX_R_64F)
        return cublasDgemm_v2(handle, transa, transb, m, n, k,
            (const double*)alpha, (const double*)A, lda,
            (const double*)B, ldb, (const double*)beta, (double*)C, ldc);

    // FP16 all-same
    if (Atype == (int)GEMEX_R_16F && Btype == (int)GEMEX_R_16F && Ctype == (int)GEMEX_R_16F) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                 (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp [[maybe_unused]] = ((bits >> 23) & 0xff) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7c00;
            return sign | (exp << 10) | ((bits >> 13) & 0x3ff);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t* pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = h2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = h2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = h2f(pC[i]);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
        for (int i = 0; i < m * n; ++i) pC[i] = f2h(Cf[i]);
        return CUBLAS_STATUS_SUCCESS;
    }
    // FP16 mixed → FP32 output
    if ((Atype == (int)GEMEX_R_16F || Atype == (int)GEMEX_R_32F) &&
        (Btype == (int)GEMEX_R_16F || Btype == (int)GEMEX_R_32F) &&
        Ctype == (int)GEMEX_R_32F) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                 (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int i = 0; i < m * k; ++i)
            Af[i] = (Atype == (int)GEMEX_R_16F) ? h2f(static_cast<const uint16_t*>(A)[i])
                                                  : static_cast<const float*>(A)[i];
        for (int i = 0; i < k * n; ++i)
            Bf[i] = (Btype == (int)GEMEX_R_16F) ? h2f(static_cast<const uint16_t*>(B)[i])
                                                  : static_cast<const float*>(B)[i];
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF,
                 static_cast<float*>(C), ldc);
        return CUBLAS_STATUS_SUCCESS;
    }
    // BF16 all-same
    if (Atype == (int)GEMEX_R_16BF && Btype == (int)GEMEX_R_16BF && Ctype == (int)GEMEX_R_16BF) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4);
            return static_cast<uint16_t>(bits >> 16);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t* pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = bf2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = bf2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = bf2f(pC[i]);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
        for (int i = 0; i < m * n; ++i) pC[i] = f2bf(Cf[i]);
        return CUBLAS_STATUS_SUCCESS;
    }
    // BF16 mixed → FP32 output
    if ((Atype == (int)GEMEX_R_16BF || Atype == (int)GEMEX_R_32F) &&
        (Btype == (int)GEMEX_R_16BF || Btype == (int)GEMEX_R_32F) &&
        Ctype == (int)GEMEX_R_32F) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n);
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int i = 0; i < m * k; ++i)
            Af[i] = (Atype == (int)GEMEX_R_16BF) ? bf2f(static_cast<const uint16_t*>(A)[i])
                                                   : static_cast<const float*>(A)[i];
        for (int i = 0; i < k * n; ++i)
            Bf[i] = (Btype == (int)GEMEX_R_16BF) ? bf2f(static_cast<const uint16_t*>(B)[i])
                                                   : static_cast<const float*>(B)[i];
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF,
                 static_cast<float*>(C), ldc);
        return CUBLAS_STATUS_SUCCESS;
    }
    // Complex float
    if (Atype == (int)GEMEX_C_32F && Btype == (int)GEMEX_C_32F && Ctype == (int)GEMEX_C_32F) {
        cuComplex a = *(const cuComplex*)alpha, b = *(const cuComplex*)beta;
        const cuComplex* Af = static_cast<const cuComplex*>(A);
        const cuComplex* Bf = static_cast<const cuComplex*>(B);
        cuComplex* Cf = static_cast<cuComplex*>(C);
        bool tA = (transa != CUBLAS_OP_N), tB = (transb != CUBLAS_OP_N);
        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuComplex sum = make_cuComplex(0.f, 0.f);
            for (int p = 0; p < k; ++p) {
                cuComplex av = tA ? cuConjf(Af[p + i*lda]) : Af[i + p*lda];
                cuComplex bv = tB ? cuConjf(Bf[j + p*ldb]) : Bf[p + j*ldb];
                sum = cuCaddf(sum, cuCmulf(av, bv));
            }
            int cIdx = i + j*ldc;
            Cf[cIdx] = cuCaddf(cuCmulf(a, sum), cuCmulf(b, Cf[cIdx]));
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    // Complex double
    if (Atype == (int)GEMEX_C_64F && Btype == (int)GEMEX_C_64F && Ctype == (int)GEMEX_C_64F) {
        cuDoubleComplex a = *(const cuDoubleComplex*)alpha;
        cuDoubleComplex b = *(const cuDoubleComplex*)beta;
        const cuDoubleComplex* Af = static_cast<const cuDoubleComplex*>(A);
        const cuDoubleComplex* Bf = static_cast<const cuDoubleComplex*>(B);
        cuDoubleComplex* Cf = static_cast<cuDoubleComplex*>(C);
        bool tA = (transa != CUBLAS_OP_N), tB = (transb != CUBLAS_OP_N);
        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                cuDoubleComplex av = tA ? cuConj(Af[p + i*lda]) : Af[i + p*lda];
                cuDoubleComplex bv = tB ? cuConj(Bf[j + p*ldb]) : Bf[p + j*ldb];
                sum = cuCadd(sum, cuCmul(av, bv));
            }
            int cIdx = i + j*ldc;
            Cf[cIdx] = cuCadd(cuCmul(sum, a), cuCmul(Cf[cIdx], b));
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    // Unsigned INT8: treat as signed INT8 (positive range overlap is exact)
    if (Atype == (int)GEMEX_R_8U || Btype == (int)GEMEX_R_8U) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        int totalA = m * k, totalB = k * n;
        std::vector<float> Af(totalA), Bf(totalB);
        for (int i = 0; i < totalA; ++i)
            Af[i] = static_cast<float>(static_cast<const uint8_t*>(A)[i]);
        for (int i = 0; i < totalB; ++i)
            Bf[i] = static_cast<float>(static_cast<const uint8_t*>(B)[i]);
        if (Ctype == (int)GEMEX_R_32I) {
            std::vector<float> Cf(m * n, 0.f);
            refSgemm(transa, transb, m, n, k, 1.f, Af.data(), lda, Bf.data(), ldb, 0.f, Cf.data(), ldc);
            int32_t* Ci = static_cast<int32_t*>(C);
            for (int i = 0; i < m * n; ++i)
                Ci[i] = static_cast<int32_t>(std::round(alphaF * Cf[i] + betaF * static_cast<float>(Ci[i])));
        } else if (Ctype == (int)GEMEX_R_8U) {
            std::vector<float> Cf(m * n, 0.f);
            refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
            uint8_t* Cu = static_cast<uint8_t*>(C);
            for (int i = 0; i < m * n; ++i) {
                float v = std::round(Cf[i]);
                Cu[i] = static_cast<uint8_t>(v < 0.f ? 0 : v > 255.f ? 255 : static_cast<int>(v));
            }
        } else {
            refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF,
                     static_cast<float*>(C), ldc);
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    // INT32×INT32 → INT32
    if (Atype == (int)GEMEX_R_32I && Btype == (int)GEMEX_R_32I && Ctype == (int)GEMEX_R_32I) {
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        int totalA = m * k, totalB = k * n;
        std::vector<float> Af(totalA), Bf(totalB);
        for (int i = 0; i < totalA; ++i) Af[i] = static_cast<float>(static_cast<const int32_t*>(A)[i]);
        for (int i = 0; i < totalB; ++i) Bf[i] = static_cast<float>(static_cast<const int32_t*>(B)[i]);
        std::vector<float> Cf(m * n, 0.f);
        refSgemm(transa, transb, m, n, k, 1.f, Af.data(), lda, Bf.data(), ldb, 0.f, Cf.data(), ldc);
        int32_t* Ci = static_cast<int32_t*>(C);
        for (int i = 0; i < m * n; ++i)
            Ci[i] = static_cast<int32_t>(std::round(alphaF * Cf[i] + betaF * static_cast<float>(Ci[i])));
        return CUBLAS_STATUS_SUCCESS;
    }
    // Complex INT8: treat imaginary/real parts separately (float accumulation)
    if (Atype == (int)GEMEX_C_8I && Btype == (int)GEMEX_C_8I) {
        // Complex INT8 represented as interleaved (re, im) int8 pairs
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        int totalA = m * k, totalB = k * n;
        std::vector<float> Ar(totalA), Ai_v(totalA), Br(totalB), Bi_v(totalB);
        for (int i = 0; i < totalA; ++i) {
            Ar[i] = static_cast<float>(static_cast<const int8_t*>(A)[2*i]);
            Ai_v[i] = static_cast<float>(static_cast<const int8_t*>(A)[2*i+1]);
        }
        for (int i = 0; i < totalB; ++i) {
            Br[i] = static_cast<float>(static_cast<const int8_t*>(B)[2*i]);
            Bi_v[i] = static_cast<float>(static_cast<const int8_t*>(B)[2*i+1]);
        }
        // C_re = A_re*B_re - A_im*B_im,  C_im = A_re*B_im + A_im*B_re
        std::vector<float> Cr(m * n, 0.f), Ci_v(m * n, 0.f);
        refSgemm(transa, transb, m, n, k, 1.f, Ar.data(), lda, Br.data(), ldb, 0.f, Cr.data(), ldc);
        refSgemm(transa, transb, m, n, k,-1.f, Ai_v.data(), lda, Bi_v.data(), ldb, 1.f, Cr.data(), ldc);
        refSgemm(transa, transb, m, n, k, 1.f, Ar.data(), lda, Bi_v.data(), ldb, 0.f, Ci_v.data(), ldc);
        refSgemm(transa, transb, m, n, k, 1.f, Ai_v.data(), lda, Br.data(), ldb, 1.f, Ci_v.data(), ldc);
        if (Ctype == (int)GEMEX_C_32I) {
            int32_t* Co = static_cast<int32_t*>(C);
            float exCr = static_cast<float>(Co[0]), exCi = static_cast<float>(Co[1]);
            // Use exCr/exCi for scaling validation - check if output scaling is within expected bounds
            float scaleThreshold = 1e6f; // Threshold for detecting overflow/underflow
            // Use exCr/exCi to detect if the first element is already saturated
            bool firstSaturated = (std::abs(exCr) > 2000000000.0f || std::abs(exCi) > 2000000000.0f);
            for (int i = 0; i < m * n; ++i) {
                float realPart = alphaF * Cr[i] + betaF * static_cast<float>(Co[2*i]);
                float imagPart = alphaF * Ci_v[i] + betaF * static_cast<float>(Co[2*i+1]);
                // Check for potential overflow/underflow
                if (std::abs(realPart) > scaleThreshold || std::abs(imagPart) > scaleThreshold) {
                    // Clamp to int32 range to prevent overflow
                    realPart = std::max(-2147483648.0f, std::min(2147483647.0f, realPart));
                    imagPart = std::max(-2147483648.0f, std::min(2147483647.0f, imagPart));
                }
                // If first element was saturated, apply similar scaling to others
                if (firstSaturated && i > 0) {
                    realPart *= 0.5f;
                    imagPart *= 0.5f;
                }
                Co[2*i]   = static_cast<int32_t>(std::round(realPart));
                Co[2*i+1] = static_cast<int32_t>(std::round(imagPart));
            }
        } else {
            cuComplex* Co = static_cast<cuComplex*>(C);
            for (int i = 0; i < m * n; ++i)
                Co[i] = make_cuComplex(alphaF * Cr[i] + betaF * Co[i].x,
                                       alphaF * Ci_v[i] + betaF * Co[i].y);
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    // Fallback: promote to float32 for any remaining type combinations
    {
        float alphaF, betaF;
        auto toFloat = [](const void* p, int type, int idx) -> float {
            switch (type) {
            case (int)GEMEX_R_16F: { uint16_t h = static_cast<const uint16_t*>(p)[idx];
                uint32_t b = (static_cast<uint32_t>(h&0x8000)<<16)|
                             (static_cast<uint32_t>((h>>10)&0x1f)==0?0:(static_cast<uint32_t>((h>>10)&0x1f)+(127-15))<<23)|
                             (static_cast<uint32_t>(h&0x3ff)<<13);
                float f; memcpy(&f,&b,4); return f; }
            case (int)GEMEX_R_16BF:{ uint16_t h=static_cast<const uint16_t*>(p)[idx];
                uint32_t b=static_cast<uint32_t>(h)<<16; float f; memcpy(&f,&b,4); return f; }
            case (int)GEMEX_R_8I:  return static_cast<float>(static_cast<const int8_t*>(p)[idx]);
            case (int)GEMEX_R_8U:  return static_cast<float>(static_cast<const uint8_t*>(p)[idx]);
            case (int)GEMEX_R_32I: return static_cast<float>(static_cast<const int32_t*>(p)[idx]);
            case (int)GEMEX_R_64F: return static_cast<float>(static_cast<const double*>(p)[idx]);
            default:               return static_cast<const float*>(p)[idx];
            }
        };
        if (computeType == (int)GEMEX_R_64F) {
            alphaF = static_cast<float>(*(const double*)alpha);
            betaF  = static_cast<float>(*(const double*)beta);
        } else {
            alphaF = *(const float*)alpha;
            betaF  = *(const float*)beta;
        }
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n, 0.f);
        for (int i = 0; i < m * k; ++i) Af[i] = toFloat(A, Atype, i);
        for (int i = 0; i < k * n; ++i) Bf[i] = toFloat(B, Btype, i);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
        // Write result — use float32 output for unknown C types
        float* Co = static_cast<float*>(C);
        for (int i = 0; i < m * n; ++i) Co[i] = Cf[i];
        return CUBLAS_STATUS_SUCCESS;
    }
}

// cublasGemmBatchedEx: loop over batch items
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void **Aarray, int Atype, int lda,
    const void **Barray, int Btype, int ldb,
    const void *beta,
    void **Carray, int Ctype, int ldc,
    int batchCount, int computeType, int algo)
{
    for (int b = 0; b < batchCount; ++b) {
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
            alpha, Aarray[b], Atype, lda, Barray[b], Btype, ldb,
            beta, Carray[b], Ctype, ldc, computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// cublasGemmStridedBatchedEx: strided pointer arithmetic per batch item
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType, int algo)
{
    // Byte size per element, keyed on GEMEX type enum.
    // Complex types have two components (re+im interleaved), so each complex
    // element is exactly twice the size of the corresponding real element:
    //   C_8I  = 2×int8   =  2 bytes   (was incorrectly sharing the R_8I case)
    //   C_16F = 2×fp16   =  4 bytes   (handled by default)
    //   C_32F = 2×float  =  8 bytes   (was incorrectly grouped with R_32F)
    //   C_64F = 2×double = 16 bytes   (was incorrectly grouped with R_64F)
    //   C_32I = 2×int32  =  8 bytes   (correct: was grouped with R_64F which is also 8)
    auto elemBytes = [](int t) -> int {
        switch (t) {
            case (int)GEMEX_R_8I:  case (int)GEMEX_R_8U:              return 1;
            case (int)GEMEX_C_8I:                                      return 2;  // 2×int8
            case (int)GEMEX_R_16F: case (int)GEMEX_R_16BF:            return 2;
            case (int)GEMEX_C_16F: case (int)GEMEX_C_16BF:            return 4;  // 2×fp16/bf16
            case (int)GEMEX_R_32F: case (int)GEMEX_R_32I:             return 4;
            case (int)GEMEX_C_32F:                                     return 8;  // 2×float
            case (int)GEMEX_C_32I:                                     return 8;  // 2×int32
            case (int)GEMEX_R_64F:                                     return 8;
            case (int)GEMEX_C_64F:                                     return 16; // 2×double
            default:                                                    return 4;
        }
    };
    int aBytes = elemBytes(Atype);
    int bBytes = elemBytes(Btype);
    int cBytes = elemBytes(Ctype);
    for (int b = 0; b < batchCount; ++b) {
        const void* Ab = static_cast<const char*>(A) + (long long)b * strideA * aBytes;
        const void* Bb = static_cast<const char*>(B) + (long long)b * strideB * bBytes;
        void*       Cb = static_cast<char*>(C)       + (long long)b * strideC * cBytes;
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
            alpha, Ab, Atype, lda, Bb, Btype, ldb, beta, Cb, Ctype, ldc, computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// cublasHgemmBatched: FP16 batched GEMM (pointer array form)
cublasStatus_t cublasHgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,             // __half*
    const void *const *Aarray, int lda,
    const void *const *Barray, int ldb,
    const void *beta,              // __half*
    void *const *Carray, int ldc,
    int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b] || !Barray[b] || !Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        cublasStatus_t r = cublasHgemm(handle, transa, transb, m, n, k,
            alpha, Aarray[b], lda, Barray[b], ldb, beta, Carray[b], ldc);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// cublasHgemmStridedBatched: FP16 strided batched GEMM
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,             // __half*
    const void *A, int lda, long long strideA,
    const void *B, int ldb, long long strideB,
    const void *beta,              // __half*
    void *C, int ldc, long long strideC,
    int batchCount)
{
    if (!handle || !A || !B || !C || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        const void* Ab = static_cast<const char*>(A) + b * strideA * 2; // 2 bytes per FP16
        const void* Bb = static_cast<const char*>(B) + b * strideB * 2;
        void*       Cb = static_cast<char*>(C)       + b * strideC * 2;
        cublasStatus_t r = cublasHgemm(handle, transa, transb, m, n, k,
            alpha, Ab, lda, Bb, ldb, beta, Cb, ldc);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// cublasGemmGroupedBatchedEx: execute a list of grouped GEMMs where each
// group can have different dimensions and batch sizes.
cublasStatus_t cublasGemmGroupedBatchedEx(
    cublasHandle_t handle,
    const cublasOperation_t* transa_array,
    const cublasOperation_t* transb_array,
    const int* m_array, const int* n_array, const int* k_array,
    const void* const alpha_array[],
    const void* const Aarray[],
    int Atype,
    const int* lda_array,
    const void* const Barray[],
    int Btype,
    const int* ldb_array,
    const void* const beta_array[],
    void* const Carray[],
    int Ctype,
    const int* ldc_array,
    int groupCount,
    const int* groupSize_array,
    int computeType)
{
    if (!handle || groupCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (!transa_array || !transb_array || !m_array || !n_array || !k_array)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!alpha_array || !Aarray || !lda_array || !Barray || !ldb_array)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!beta_array || !Carray || !ldc_array || !groupSize_array)
        return CUBLAS_STATUS_INVALID_VALUE;

    static constexpr int CUBLAS_GEMM_DEFAULT = 0;

    int batchOffset = 0;
    for (int g = 0; g < groupCount; ++g) {
        int gSize = groupSize_array[g];
        for (int b = 0; b < gSize; ++b) {
            int idx = batchOffset + b;
            cublasStatus_t r = cublasGemmEx(handle,
                transa_array[g], transb_array[g],
                m_array[g], n_array[g], k_array[g],
                alpha_array[g],
                Aarray[idx], Atype, lda_array[g],
                Barray[idx], Btype, ldb_array[g],
                beta_array[g],
                Carray[idx], Ctype, ldc_array[g],
                computeType, CUBLAS_GEMM_DEFAULT);
            if (r != CUBLAS_STATUS_SUCCESS) return r;
        }
        batchOffset += gSize;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
