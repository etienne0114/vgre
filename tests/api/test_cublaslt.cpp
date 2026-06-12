// cuBLASLt shim functional tests — matmul with epilogues, algo heuristics,
// FP16 compute path, descriptor lifecycle.

#include "vgre/api/cublaslt_shim.h"
#include "vgre/core/math/fp_quant_gemm.h"   // Track 17 — FP8 codecs for input/reference
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs(static_cast<double>(a)-static_cast<double>(b)) < (eps))

// ── 1. Handle lifecycle ───────────────────────────────────────────────────────
int test_handle() {
    cublasLtHandle_t h = nullptr;
    if (cublasLtCreate(&h)  != CUBLAS_STATUS_SUCCESS || !h) FAIL("create");
    if (cublasLtDestroy(h)  != CUBLAS_STATUS_SUCCESS)       FAIL("destroy");
    PASS("handle lifecycle");
    return 0;
}

// ── 2. Descriptor attribute round-trip ───────────────────────────────────────
int test_desc_attributes() {
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    int transA = 1;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transA, sizeof(transA));
    int readBack = 0;
    size_t written = 0;
    cublasLtMatmulDescGetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &readBack, sizeof(readBack), &written);
    if (readBack != 1 || written != sizeof(int)) FAIL("transA round-trip");

    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(h);
    PASS("descriptor attribute round-trip");
    return 0;
}

// ── 3. Basic float SGEMM: C = A * B ──────────────────────────────────────────
// A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  C = [[19,22],[43,50]]
int test_sgemm_float() {
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);

    cublasLtMatmulDesc_t   desc  = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 2, 2, 2);

    // Column-major storage
    float A[] = {1.f,3.f, 2.f,4.f}; // [[1,2],[3,4]] col-major
    float B[] = {5.f,7.f, 6.f,8.f}; // [[5,6],[7,8]] col-major
    float C[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    auto s = cublasLtMatmul(h, desc, &alpha, A, layA, B, layB, &beta, C, layC,
                             C, layC, nullptr, nullptr, 0, nullptr);
    if (s != CUBLAS_STATUS_SUCCESS) FAIL("matmul status");

    // C col-major: [[19,43],[22,50]]
    if (!NEAR(C[0], 19.f, 1e-4)) FAIL("C[0,0]");
    if (!NEAR(C[1], 43.f, 1e-4)) FAIL("C[1,0]");
    if (!NEAR(C[2], 22.f, 1e-4)) FAIL("C[0,1]");
    if (!NEAR(C[3], 50.f, 1e-4)) FAIL("C[1,1]");

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtDestroy(h);
    PASS("float SGEMM 2×2");
    return 0;
}

// ── 4. SGEMM with RELU epilogue ───────────────────────────────────────────────
// A = [[-1,2],[3,-4]]  B = I  =>  C = A, then ReLU clips negatives to 0
int test_sgemm_relu() {
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);

    cublasLtMatmulDesc_t   desc  = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_RELU;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 2, 2, 2);

    float A[] = {-1.f, 3.f, 2.f, -4.f}; // col-major: [[-1,2],[3,-4]]
    float B[] = {1.f, 0.f, 0.f, 1.f};   // identity col-major
    float C[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmul(h, desc, &alpha, A, layA, B, layB, &beta, C, layC,
                   C, layC, nullptr, nullptr, 0, nullptr);

    // After ReLU: negative values clamped to 0
    if (!NEAR(C[0], 0.f, 1e-4)) FAIL("RELU clamp C[0,0]=-1");
    if (!NEAR(C[1], 3.f, 1e-4)) FAIL("RELU keep C[1,0]=3");
    if (!NEAR(C[2], 2.f, 1e-4)) FAIL("RELU keep C[0,1]=2");
    if (!NEAR(C[3], 0.f, 1e-4)) FAIL("RELU clamp C[1,1]=-4");

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtDestroy(h);
    PASS("SGEMM + RELU epilogue");
    return 0;
}

// ── 5. SGEMM with BIAS epilogue ───────────────────────────────────────────────
// A = I  B = I  alpha=1 beta=0  then bias=[10,20] added per column
int test_sgemm_bias() {
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);

    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    float bias[2] = {10.f, 20.f};
    const float *biasPtr = bias; // pass address of the pointer, not the array
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(void*));

    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 2, 2, 2);

    float A[] = {1.f,0.f, 0.f,1.f}; // identity col-major
    float B[] = {1.f,0.f, 0.f,1.f};
    float C[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmul(h, desc, &alpha, A, layA, B, layB, &beta, C, layC,
                   C, layC, nullptr, nullptr, 0, nullptr);

    // C = I + bias per-col: [[1+10,0+10],[0+20,1+20]] row 0 / row 1
    // col-major: C[0]=1+10=11, C[1]=0+20=20, C[2]=0+10=10, C[3]=1+20=21
    if (!NEAR(C[0], 11.f, 1e-4)) FAIL("BIAS C[0]");
    if (!NEAR(C[1], 20.f, 1e-4)) FAIL("BIAS C[1]");
    if (!NEAR(C[2], 10.f, 1e-4)) FAIL("BIAS C[2]");
    if (!NEAR(C[3], 21.f, 1e-4)) FAIL("BIAS C[3]");

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtDestroy(h);
    PASS("SGEMM + BIAS epilogue");
    return 0;
}

// ── 6. AlgoGetHeuristic + LRU cache ──────────────────────────────────────────
int test_algo_heuristic() {
    cublasLtHandle_t          h    = nullptr;
    cublasLtMatmulDesc_t      desc = nullptr;
    cublasLtMatrixLayout_t    layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    cublasLtCreate(&h);
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 4, 4, 4);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 4, 4, 4);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 4, 4, 4);
    cublasLtMatmulPreferenceCreate(&pref);

    cublasLtMatmulHeuristicResult_t results[4] = {};
    int returnCount = 0;

    // First call — populates LRU cache; may return 1..4 candidates
    auto s1 = cublasLtMatmulAlgoGetHeuristic(h, desc, layA, layB, layC, layC,
                                              pref, 4, results, &returnCount);
    if (s1 != CUBLAS_STATUS_SUCCESS)      FAIL("heuristic first call");
    if (returnCount < 1 || returnCount > 4) FAIL("returnCount out of [1,4]");
    if (results[0].state != CUBLAS_STATUS_SUCCESS) FAIL("algo state");

    // Second call — should hit LRU cache; same count expected
    int returnCount2 = 0;
    auto s2 = cublasLtMatmulAlgoGetHeuristic(h, desc, layA, layB, layC, layC,
                                              pref, 4, results, &returnCount2);
    if (s2 != CUBLAS_STATUS_SUCCESS)        FAIL("heuristic second call (cache)");
    if (returnCount2 < 1 || returnCount2 > 4) FAIL("cache returnCount out of [1,4]");

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtDestroy(h);
    PASS("AlgoGetHeuristic + LRU cache");
    return 0;
}

// ── 7. DRELU epilogue (backward ReLU gradient) ───────────────────────────────
// After GEMM: C = [[2,-1],[0,3]] → DRELU: [[1,0],[0,1]]
int test_drelu() {
    cublasLtHandle_t h = nullptr;
    cublasLtCreate(&h);

    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_DRELU;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep));
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 2, 2, 2);

    // A * B = I, so C = A  (use A=I, B has the values we want post-matmul)
    float A[] = {1.f, 0.f, 0.f, 1.f}; // identity
    float B[] = {2.f, 0.f, -1.f, 3.f}; // col-major: [[2,-1],[0,3]]
    float C[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmul(h, desc, &alpha, A, layA, B, layB, &beta, C, layC,
                   C, layC, nullptr, nullptr, 0, nullptr);

    // DRELU: >0 → 1, <=0 → 0
    if (!NEAR(C[0], 1.f, 1e-4)) FAIL("DRELU C[0]=2>0→1");
    if (!NEAR(C[1], 0.f, 1e-4)) FAIL("DRELU C[1]=0→0");
    if (!NEAR(C[2], 0.f, 1e-4)) FAIL("DRELU C[2]=-1→0");
    if (!NEAR(C[3], 1.f, 1e-4)) FAIL("DRELU C[3]=3>0→1");

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtDestroy(h);
    PASS("DRELU epilogue");
    return 0;
}

// ── 8. Null rejection ─────────────────────────────────────────────────────────
int test_null_rejection() {
    if (cublasLtCreate(nullptr) == CUBLAS_STATUS_SUCCESS) FAIL("null handle");
    PASS("null rejection");
    return 0;
}

// ── 9. Version queries ────────────────────────────────────────────────────────
int test_version_queries() {
    size_t ver = 0, cudart_ver = 0;
    if (cublasLtGetVersion(&ver) != CUBLAS_STATUS_SUCCESS || ver == 0)
        FAIL("cublasLtGetVersion failed or returned 0");
    if (cublasLtGetCudartVersion(&cudart_ver) != CUBLAS_STATUS_SUCCESS || cudart_ver == 0)
        FAIL("cublasLtGetCudartVersion failed or returned 0");
    PASS("version queries (cublasLtGetVersion + cublasLtGetCudartVersion)");
    return 0;
}

// ── 10. Status string helpers ─────────────────────────────────────────────────
int test_status_strings() {
    const char *ok_name = cublasLtGetStatusName(CUBLAS_STATUS_SUCCESS);
    const char *ok_str  = cublasLtGetStatusString(CUBLAS_STATUS_SUCCESS);
    if (!ok_name || ok_name[0] == '\0') FAIL("cublasLtGetStatusName returned empty");
    if (!ok_str  || ok_str[0]  == '\0') FAIL("cublasLtGetStatusString returned empty");
    const char *err_name = cublasLtGetStatusName(CUBLAS_STATUS_INVALID_VALUE);
    if (!err_name || err_name[0] == '\0') FAIL("status name for INVALID_VALUE empty");
    PASS("status string helpers");
    return 0;
}

// ── 11. Plural heuristics alias (cublasLtMatmulAlgoGetHeuristics) ────────────
int test_algo_heuristics_plural() {
    cublasLtHandle_t h = nullptr; cublasLtCreate(&h);
    cublasLtMatmulDesc_t   desc = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_32F, 8, 8, 8);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_32F, 8, 8, 8);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 8, 8, 8);
    cublasLtMatmulPreferenceCreate(&pref);

    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    auto s = cublasLtMatmulAlgoGetHeuristics(h, desc, layA, layB, layC, layC, pref,
                                              1, &result, &returned);
    if (s != CUBLAS_STATUS_SUCCESS || returned != 1)
        FAIL("cublasLtMatmulAlgoGetHeuristics (plural) failed");

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layA); cublasLtMatrixLayoutDestroy(layB); cublasLtMatrixLayoutDestroy(layC);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(h);
    PASS("cublasLtMatmulAlgoGetHeuristics plural alias with LRU cache");
    return 0;
}

// ── Driver ────────────────────────────────────────────────────────────────────
// ── FP8 scaled matmul (Track 17) ─────────────────────────────────────────────
// A,B are FP8 E4M3 with per-tensor dequant scales; output FP32. The scaled GEMM
// must equal the GEMM of the dequantized (scale·stored) matrices. Column-major,
// like test_sgemm_float. Real A=[[2,4],[6,8]] via stored {1,3,2,4}·sA=2; real
// B=[[1,1],[1,1]] via stored {1,1,1,1}·sB=1 → C=[[6,6],[14,14]] (col-major
// {6,14,6,14}). Without the scale the result would be {3,7,3,7}, so this proves
// the scale pointer is honored.
static int fp8_case(cublasLtDatatype_t aType, cublasLtDatatype_t bType) {
    using namespace vgre::quant;
    cublasLtHandle_t h = nullptr; cublasLtCreate(&h);
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t layA=nullptr, layB=nullptr, layC=nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&layA, aType, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layB, bType, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_32F, 2, 2, 2);

    float sA = 2.0f, sB = 1.0f;
    auto enc = [&](int t, float v){ return t==CUDA_R_8F_E5M2 ? f32_to_e5m2(v) : f32_to_e4m3(v); };
    auto dec = [&](int t, uint8_t b){ return t==CUDA_R_8F_E5M2 ? e5m2_to_f32(b) : e4m3_to_f32(b); };
    // stored A (col-major) = {1,3,2,4}; stored B = {1,1,1,1}
    uint8_t A[4] = { enc(aType,1), enc(aType,3), enc(aType,2), enc(aType,4) };
    uint8_t B[4] = { enc(bType,1), enc(bType,1), enc(bType,1), enc(bType,1) };
    float C[4] = {}; float alpha=1.0f, beta=0.0f;

    // The attribute value IS a pointer to the scale, so pass the address of a
    // pointer variable (the setter copies sizeof(void*) bytes as the pointer).
    const void *pA = &sA, *pB = &sB;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA, sizeof(void*));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB, sizeof(void*));

    auto s = cublasLtMatmul(h, desc, &alpha, A, layA, B, layB, &beta, C, layC,
                            C, layC, nullptr, nullptr, 0, nullptr);
    if (s != CUBLAS_STATUS_SUCCESS) FAIL("fp8 matmul status");

    // Reference: GEMM of the dequantized matrices (col-major).
    float Ar[4], Br[4];
    for (int i=0;i<4;++i){ Ar[i]=sA*dec(aType,A[i]); Br[i]=sB*dec(bType,B[i]); }
    // col-major 2x2: Cref[r + c*2] = sum_k Ar[r + k*2]*Br[k + c*2]
    float Cref[4];
    for (int r=0;r<2;++r) for (int c=0;c<2;++c){
        float acc=0; for(int kk=0;kk<2;++kk) acc += Ar[r+kk*2]*Br[kk+c*2];
        Cref[r+c*2]=acc;
    }
    for (int i=0;i<4;++i) if (!NEAR(C[i], Cref[i], 1e-3)) FAIL("fp8 result != dequant reference");
    // Sanity: the scale must matter (unscaled would give C[0]=3, not 6).
    if (NEAR(C[0], 3.0f, 1e-3)) FAIL("fp8 A-scale was not applied");

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(layA); cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC); cublasLtDestroy(h);
    return 0;
}

int test_fp8_matmul() {
    if (fp8_case(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3)) return 1;
    PASS("FP8 E4M3 scaled matmul == dequant reference");
    if (fp8_case(CUDA_R_8F_E5M2, CUDA_R_8F_E5M2)) return 1;
    PASS("FP8 E5M2 scaled matmul == dequant reference");
    if (fp8_case(CUDA_R_8F_E4M3, CUDA_R_8F_E5M2)) return 1;  // mixed A=E4M3, B=E5M2
    PASS("FP8 mixed E4M3×E5M2 scaled matmul == dequant reference");
    return 0;
}

int main() {
    std::cout << "=== cuBLASLt Shim Functional Tests ===\n";
    int rc = 0;
    rc |= test_handle();
    rc |= test_null_rejection();
    rc |= test_desc_attributes();
    rc |= test_sgemm_float();
    rc |= test_sgemm_relu();
    rc |= test_sgemm_bias();
    rc |= test_algo_heuristic();
    rc |= test_drelu();
    rc |= test_version_queries();
    rc |= test_status_strings();
    rc |= test_algo_heuristics_plural();
    rc |= test_fp8_matmul();
    if (rc == 0) std::cout << "\nAll cuBLASLt tests passed!\n";
    return rc;
}
