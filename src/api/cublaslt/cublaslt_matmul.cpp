// cuBLASLt matmul execution shim — fuses epilogues (ReLU, GELU, bias,
// backward activation gradients) in post-processing for CPU emulation.

#include "cublaslt_state.h"
#include "vgre/common/openmp_helper.h"
#include "vgre/core/math/fp_quant_gemm.h"   // Track 17 — FP8 E4M3/E5M2 codecs

#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

using namespace vgre_lt;

extern void refSgemm(bool tA, bool tB, int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc);
extern void refDgemm(bool tA, bool tB, int M, int N, int K,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb, double beta, double* C, int ldc);

namespace {

// ── Epilogue post-processing templates ───────────────────────────────────────

template<typename T>
void applyRelu(T *data, size_t count) {
    #ifdef _OPENMP
    #pragma omp parallel for if (count > 1024)
    #endif
    for (int i = 0; i < (int)count; ++i) if (data[i] < T(0)) data[i] = T(0);
}

// DRELU: element-wise ReLU gradient — 1 where input > 0, else 0.
template<typename T>
void applyDRelu(T *data, size_t count) {
    #ifdef _OPENMP
    #pragma omp parallel for if (count > 1024)
    #endif
    for (int i = 0; i < (int)count; ++i) data[i] = (data[i] > T(0)) ? T(1) : T(0);
}

template<typename T>
void applyGelu(T *data, size_t count) {
    #ifdef _OPENMP
    #pragma omp parallel for if (count > 1024)
    #endif
    for (int i = 0; i < (int)count; ++i) {
        T x = data[i];
        data[i] = T(0.5) * x * (T(1.0) + std::tanh(std::sqrt(T(2.0) / T(M_PI)) *
                               (x + T(0.044715) * x * x * x)));
    }
}

// DGELU: d/dx[GELU(x)] = 0.5*(1+tanh(cx)) + 0.5*x*sech²(cx)*c*(1+3k*x²)
template<typename T>
void applyDGelu(T *data, size_t count) {
    const T c = static_cast<T>(std::sqrt(2.0 / M_PI));
    const T k = static_cast<T>(0.044715);
    #ifdef _OPENMP
    #pragma omp parallel for if (count > 1024)
    #endif
    for (int i = 0; i < (int)count; ++i) {
        T x   = data[i];
        T cx  = c * (x + k * x * x * x);
        T th  = std::tanh(cx);
        T dcx = c * (T(1) + T(3) * k * x * x);
        data[i] = T(0.5) * (T(1) + th) + T(0.5) * x * (T(1) - th * th) * dcx;
    }
}

template<typename T>
void applyBias(T *data, size_t rows, size_t cols, const T *bias) {
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (rows * cols > 1024)
    #endif
    for (int r = 0; r < (int)rows; ++r)
        for (int c = 0; c < (int)cols; ++c)
            data[r * cols + c] += bias[c];
}

// Column-sum of gradient matrix → bias gradient vector (BGRADA/BGRADB/DRELU_BGRAD/DGELU_BGRAD).
template<typename T>
void applyBiasGrad(const T *data, size_t rows, size_t cols, T *biasGrad) {
    std::fill(biasGrad, biasGrad + cols, T(0));
    #ifdef _OPENMP
    #pragma omp parallel for if (rows > 64)
    #endif
    for (int r = 0; r < (int)rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            biasGrad[c] += data[r * cols + c];
}

} // anonymous namespace

extern "C" {

// ── Matmul execution ─────────────────────────────────────────────────────────

cublasStatus_t cublasLtMatmul(cublasLtHandle_t /*lightHandle*/,
                              cublasLtMatmulDesc_t matmulDesc,
                              const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc,
                              const void *B, cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              void *C, cublasLtMatrixLayout_t Cdesc,
                              void *D, cublasLtMatrixLayout_t Ddesc,
                              const void * /*algo*/, void * /*workspace*/,
                              size_t /*workspaceSizeInBytes*/, void * /*stream*/) {
    if (!matmulDesc || !alpha || !A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto dIt = g_matmulDescs.find(reinterpret_cast<uintptr_t>(matmulDesc));
    auto aIt = g_layouts.find(reinterpret_cast<uintptr_t>(Adesc));
    auto bIt = g_layouts.find(reinterpret_cast<uintptr_t>(Bdesc));
    auto cIt = g_layouts.find(reinterpret_cast<uintptr_t>(Cdesc));
    if (dIt == g_matmulDescs.end() || aIt == g_layouts.end() ||
        bIt == g_layouts.end() || cIt == g_layouts.end())
        return CUBLAS_STATUS_INVALID_VALUE;

    const MatmulDesc   &d  = dIt->second;
    const MatrixLayout &la = aIt->second;
    const MatrixLayout &lb = bIt->second;
    const MatrixLayout &lc = cIt->second;

    int m = static_cast<int>(lc.rows);
    int n = static_cast<int>(lc.cols);
    int k = static_cast<int>(d.transA ? la.rows : la.cols);

    // ── Type dispatch ─────────────────────────────────────────────────────────
    // Use refSgemm/refDgemm directly — the cuBLAS wrappers require a valid
    // cuBLAS handle which cuBLASLt callers don't provide.
    bool tA = d.transA != 0, tB = d.transB != 0;
    if (la.type == CUDA_R_32F && lb.type == CUDA_R_32F && lc.type == CUDA_R_32F) {
        float alphaF = *static_cast<const float*>(alpha);
        float betaF  = *static_cast<const float*>(beta);
        refSgemm(tB, tA, n, m, k, alphaF,
                 static_cast<const float*>(B), static_cast<int>(lb.ld),
                 static_cast<const float*>(A), static_cast<int>(la.ld),
                 betaF, static_cast<float*>(C), static_cast<int>(lc.ld));
    } else if (la.type == CUDA_R_64F && lb.type == CUDA_R_64F && lc.type == CUDA_R_64F) {
        double alphaD = *static_cast<const double*>(alpha);
        double betaD  = *static_cast<const double*>(beta);
        refDgemm(tB, tA, n, m, k, alphaD,
                 static_cast<const double*>(B), static_cast<int>(lb.ld),
                 static_cast<const double*>(A), static_cast<int>(la.ld),
                 betaD, static_cast<double*>(C), static_cast<int>(lc.ld));
    } else {
        // Reduced-precision and mixed types: widen to float, run SGEMM, narrow back.
        float alphaF = *static_cast<const float*>(alpha);
        float betaF  = *static_cast<const float*>(beta);
        std::vector<float> Af(static_cast<size_t>(m * k));
        std::vector<float> Bf(static_cast<size_t>(k * n));
        std::vector<float> Cf(static_cast<size_t>(m * n));

        // Track 17 — FP8 per-tensor dequant scales (CUBLASLT_MATMUL_DESC_*_SCALE_
        // POINTER). The scale is a multiplier: real value = scale · stored_fp8.
        // Folding sA into A's decode and sB into B's decode yields the cuBLASLt
        // contract D = α·sA·sB·(A·B) + β·sC·C. fp8 decoders from vgre::quant.
        const float sA = d.aScalePtr ? *static_cast<const float*>(d.aScalePtr) : 1.0f;
        const float sB = d.bScalePtr ? *static_cast<const float*>(d.bScalePtr) : 1.0f;
        const float sC = d.cScalePtr ? *static_cast<const float*>(d.cScalePtr) : 1.0f;
        const float sD = d.dScalePtr ? *static_cast<const float*>(d.dScalePtr) : 1.0f;
        auto decodeFp8 = [](int type, uint8_t b) -> float {
            return (type == CUDA_R_8F_E5M2) ? vgre::quant::e5m2_to_f32(b)
                                            : vgre::quant::e4m3_to_f32(b);
        };
        const bool aFp8 = (la.type == CUDA_R_8F_E4M3 || la.type == CUDA_R_8F_E5M2);
        const bool bFp8 = (lb.type == CUDA_R_8F_E4M3 || lb.type == CUDA_R_8F_E5M2);

        // Track P3-1 — FP4 (NVFP4/MXFP4): operands are E2M1 packed 2 nibbles/byte,
        // dequantized per K-block (block size from *_SCALE_MODE) by a float scale
        // array, or by a single per-tensor scale when the block size is 0. The
        // micro-scaling contract: real[i,k] = scale[i, k/block] · e2m1(nibble).
        const bool aFp4 = (la.type == CUDA_R_4F_E2M1);
        const bool bFp4 = (lb.type == CUDA_R_4F_E2M1);
        const int  aBlk = d.aScaleBlock, bBlk = d.bScaleBlock;
        const int  aNb  = (aBlk > 0) ? (k + aBlk - 1) / aBlk : 1;  // A scales [M, aNb]
        const float* aBS = static_cast<const float*>(d.aScalePtr);
        const float* bBS = static_cast<const float*>(d.bScalePtr);
        auto e2m1 = [](const uint8_t* p, size_t idx) {
            const uint8_t nib = (idx & 1u) ? static_cast<uint8_t>(p[idx >> 1] >> 4)
                                           : static_cast<uint8_t>(p[idx >> 1] & 0x0Fu);
            return vgre::quant::e2m1_to_f32(nib);
        };

        // FP16 ↔ float helpers
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000u) << 16) |
                ((static_cast<uint32_t>((h >> 10) & 0x1fu) == 0u ? 0u :
                  (static_cast<uint32_t>((h >> 10) & 0x1fu) + 112u)) << 23) |
                (static_cast<uint32_t>(h & 0x3ffu) << 13);
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4);
            uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
            int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) | ((bits >> 13) & 0x3ffu));
        };
        // BF16 ↔ float helpers
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4);
            return static_cast<uint16_t>(bits >> 16);
        };

        // Widen A
        if (la.type == CUDA_R_16F) {
            const uint16_t *p = static_cast<const uint16_t*>(A);
            for (int i = 0; i < m * k; ++i) Af[static_cast<size_t>(i)] = h2f(p[i]);
        } else if (la.type == CUDA_R_16BF) {
            const uint16_t *p = static_cast<const uint16_t*>(A);
            for (int i = 0; i < m * k; ++i) Af[static_cast<size_t>(i)] = bf2f(p[i]);
        } else if (la.type == CUDA_R_8I) {
            const int8_t *p = static_cast<const int8_t*>(A);
            for (int i = 0; i < m * k; ++i) Af[static_cast<size_t>(i)] = static_cast<float>(p[i]);
        } else if (aFp8) {
            const uint8_t *p = static_cast<const uint8_t*>(A);
            for (int i = 0; i < m * k; ++i) Af[static_cast<size_t>(i)] = sA * decodeFp8(la.type, p[i]);
        } else if (aFp4) {
            // A is [m,k]; map each (row,col) to its flat index per the layout so
            // the block-scale lookup (per row, per K-block) is layout-correct.
            const uint8_t *p = static_cast<const uint8_t*>(A);
            const bool col = (la.order == CUBLASLT_ORDER_COL);
            const int64_t lda = la.ld ? la.ld : (col ? m : k);
            for (int mm = 0; mm < m; ++mm)
                for (int kk = 0; kk < k; ++kk) {
                    const size_t idx = col ? static_cast<size_t>(kk) * lda + mm
                                           : static_cast<size_t>(mm) * lda + kk;
                    const float sc = aBS ? (aBlk > 0 ? aBS[mm * aNb + kk / aBlk] : aBS[0]) : 1.0f;
                    Af[idx] = sc * e2m1(p, idx);
                }
        } else {
            memcpy(Af.data(), A, static_cast<size_t>(m * k) * sizeof(float));
        }

        // Widen B
        if (lb.type == CUDA_R_16F) {
            const uint16_t *p = static_cast<const uint16_t*>(B);
            for (int i = 0; i < k * n; ++i) Bf[static_cast<size_t>(i)] = h2f(p[i]);
        } else if (lb.type == CUDA_R_16BF) {
            const uint16_t *p = static_cast<const uint16_t*>(B);
            for (int i = 0; i < k * n; ++i) Bf[static_cast<size_t>(i)] = bf2f(p[i]);
        } else if (lb.type == CUDA_R_8I) {
            const int8_t *p = static_cast<const int8_t*>(B);
            for (int i = 0; i < k * n; ++i) Bf[static_cast<size_t>(i)] = static_cast<float>(p[i]);
        } else if (bFp8) {
            const uint8_t *p = static_cast<const uint8_t*>(B);
            for (int i = 0; i < k * n; ++i) Bf[static_cast<size_t>(i)] = sB * decodeFp8(lb.type, p[i]);
        } else if (bFp4) {
            // B is [k,n]; block scales are [k/block, n].
            const uint8_t *p = static_cast<const uint8_t*>(B);
            const bool col = (lb.order == CUBLASLT_ORDER_COL);
            const int64_t ldb = lb.ld ? lb.ld : (col ? k : n);
            for (int kk = 0; kk < k; ++kk)
                for (int nn = 0; nn < n; ++nn) {
                    const size_t idx = col ? static_cast<size_t>(nn) * ldb + kk
                                           : static_cast<size_t>(kk) * ldb + nn;
                    const float sc = bBS ? (bBlk > 0 ? bBS[(kk / bBlk) * n + nn] : bBS[0]) : 1.0f;
                    Bf[idx] = sc * e2m1(p, idx);
                }
        } else {
            memcpy(Bf.data(), B, static_cast<size_t>(k * n) * sizeof(float));
        }

        // Widen C (for beta accumulation)
        if (lc.type == CUDA_R_16F) {
            const uint16_t *p = static_cast<const uint16_t*>(C);
            for (int i = 0; i < m * n; ++i) Cf[static_cast<size_t>(i)] = h2f(p[i]);
        } else if (lc.type == CUDA_R_16BF) {
            const uint16_t *p = static_cast<const uint16_t*>(C);
            for (int i = 0; i < m * n; ++i) Cf[static_cast<size_t>(i)] = bf2f(p[i]);
        } else if (lc.type == CUDA_R_8I) {
            const int8_t *p = static_cast<const int8_t*>(C);
            for (int i = 0; i < m * n; ++i) Cf[static_cast<size_t>(i)] = static_cast<float>(p[i]);
        } else if (lc.type == CUDA_R_32I) {
            const int32_t *p = static_cast<const int32_t*>(C);
            for (int i = 0; i < m * n; ++i) Cf[static_cast<size_t>(i)] = static_cast<float>(p[i]);
        } else if (lc.type == CUDA_R_8F_E4M3 || lc.type == CUDA_R_8F_E5M2) {
            const uint8_t *p = static_cast<const uint8_t*>(C);
            for (int i = 0; i < m * n; ++i) Cf[static_cast<size_t>(i)] = sC * decodeFp8(lc.type, p[i]);
        } else {
            memcpy(Cf.data(), C, static_cast<size_t>(m * n) * sizeof(float));
        }

        refSgemm(tB, tA, n, m, k, alphaF, Bf.data(), static_cast<int>(lb.ld),
                 Af.data(), static_cast<int>(la.ld), betaF, Cf.data(), static_cast<int>(lc.ld));

        // Narrow C back
        if (lc.type == CUDA_R_16F) {
            uint16_t *p = static_cast<uint16_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = f2h(Cf[i]);
        } else if (lc.type == CUDA_R_16BF) {
            uint16_t *p = static_cast<uint16_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = f2bf(Cf[i]);
        } else if (lc.type == CUDA_R_8I) {
            int8_t *p = static_cast<int8_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = static_cast<int8_t>(Cf[i]);
        } else if (lc.type == CUDA_R_32I) {
            int32_t *p = static_cast<int32_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = static_cast<int32_t>(Cf[i]);
        } else if (lc.type == CUDA_R_8F_E4M3) {
            // FP8 output: quantize the computed result scaled by the D scale.
            uint8_t *p = static_cast<uint8_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = vgre::quant::f32_to_e4m3(Cf[i] * sD);
        } else if (lc.type == CUDA_R_8F_E5M2) {
            uint8_t *p = static_cast<uint8_t*>(C);
            for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) p[i] = vgre::quant::f32_to_e5m2(Cf[i] * sD);
        } else {
            memcpy(C, Cf.data(), static_cast<size_t>(m * n) * sizeof(float));
        }
    }

    // Copy C → D if they differ
    if (D && D != C) {
        auto dIt2 = g_layouts.find(reinterpret_cast<uintptr_t>(Ddesc));
        if (dIt2 != g_layouts.end()) {
            size_t count = static_cast<size_t>(lc.rows * lc.cols);
            if (lc.type == CUDA_R_32F)
                memcpy(D, C, count * sizeof(float));
            else if (lc.type == CUDA_R_64F)
                memcpy(D, C, count * sizeof(double));
            else if (lc.type == CUDA_R_16F || lc.type == CUDA_R_16BF)
                memcpy(D, C, count * sizeof(uint16_t));
            else if (lc.type == CUDA_R_8F_E4M3 || lc.type == CUDA_R_8F_E5M2 ||
                     lc.type == CUDA_R_8I)
                memcpy(D, C, count);   // 1 byte/element
        }
    }

    // Apply epilogue on D (or C if D==nullptr or D==C)
    void *target = (D && D != C) ? D : C;
    size_t count = static_cast<size_t>(lc.rows * lc.cols);

    auto applyEpilogue = [&](auto *fp, auto /*tag*/) {
        using T = std::remove_pointer_t<decltype(fp)>;
        const T *bias = d.biasPtr ? static_cast<const T*>(d.biasPtr) : nullptr;
        switch (d.epilogue) {
        case CUBLASLT_EPILOGUE_BIAS:
            if (bias) applyBias(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), bias);
            break;
        case CUBLASLT_EPILOGUE_RELU:
        case CUBLASLT_EPILOGUE_RELU_AUX:
            applyRelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_RELU_BIAS:
            if (bias) applyBias(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), bias);
            applyRelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_GELU:
        case CUBLASLT_EPILOGUE_GELU_AUX:
            applyGelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_GELU_BIAS:
            if (bias) applyBias(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), bias);
            applyGelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_DRELU:
            applyDRelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_DRELU_BGRAD:
            if (d.epilogueAuxPtr) applyBiasGrad(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), static_cast<T*>(d.epilogueAuxPtr));
            applyDRelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_DGELU:
            applyDGelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_DGELU_BGRAD:
            if (d.epilogueAuxPtr) applyBiasGrad(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), static_cast<T*>(d.epilogueAuxPtr));
            applyDGelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_BGRADA:
        case CUBLASLT_EPILOGUE_BGRADB:
            if (d.epilogueAuxPtr) applyBiasGrad(fp, static_cast<size_t>(lc.rows), static_cast<size_t>(lc.cols), static_cast<T*>(d.epilogueAuxPtr));
            break;
        case CUBLASLT_EPILOGUE_DEFAULT:
        default:
            break;
        }
    };

    if (lc.type == CUDA_R_32F) applyEpilogue(static_cast<float*>(target), 0.0f);
    else if (lc.type == CUDA_R_64F) applyEpilogue(static_cast<double*>(target), 0.0);

    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
