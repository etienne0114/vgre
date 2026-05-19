// cuBLAS GemmEx: mixed-precision GEMM (HGEMM, FP16, BF16, INT8, complex)
// Split from cublas_level3.cpp to keep that file under 800 lines.

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"
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
        int exp = ((bits >> 23) & 0xff) - 127 + 15;
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
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
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
    return CUBLAS_STATUS_NOT_SUPPORTED;
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
    int elemSize = (Atype == (int)GEMEX_R_8I) ? 1 : 4;
    for (int b = 0; b < batchCount; ++b) {
        const void* Ab = static_cast<const char*>(A) + b * strideA * elemSize;
        const void* Bb = static_cast<const char*>(B) + b * strideB * elemSize;
        void*       Cb = static_cast<char*>(C)       + b * strideC * 4;
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
            alpha, Ab, Atype, lda, Bb, Btype, ldb, beta, Cb, Ctype, ldc, computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
