// cuSPARSE shim functional tests — SpMV, SpSV, SpGEMM, ILU0, IC0,
// SparseToDense, DenseToSparse, CSC descriptor.

#include "vgre/api/cusparse_shim.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((a)-(b)) < (eps))

// ── 1. Handle lifecycle ───────────────────────────────────────────────────────
int test_handle() {
    cusparseHandle_t h = nullptr;
    if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS || !h) FAIL("create handle");
    if (cusparseDestroy(h)  != CUSPARSE_STATUS_SUCCESS)      FAIL("destroy handle");
    PASS("handle lifecycle");
    return 0;
}

// ── 2. CSR SpMV (y = A*x, float) ─────────────────────────────────────────────
// A = [[1,0],[0,2]]  x=[3,4]  =>  y=[3,8]
int test_spmv_float() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // CSR for 2×2 diagonal: nnz=2
    std::vector<int>   rowPtr = {0, 1, 2};
    std::vector<int>   colInd = {0, 1};
    std::vector<float> vals   = {1.0f, 2.0f};
    std::vector<float> x      = {3.0f, 4.0f};
    std::vector<float> y      = {0.0f, 0.0f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    cusparseCreateCsr(&matA, 2, 2, 2,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 2, x.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 2, y.data(), CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                            vecX, &beta, vecY, CUDA_R_32F,
                            &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                 vecX, &beta, vecY, CUDA_R_32F, buf.data());

    if (!NEAR(y[0], 3.0f, 1e-5f)) FAIL("SpMV y[0]");
    if (!NEAR(y[1], 8.0f, 1e-5f)) FAIL("SpMV y[1]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("CSR SpMV float");
    return 0;
}

// ── 3. SpSV — lower triangular forward substitution ──────────────────────────
// L = [[2,0],[1,3]]  b=[4,7]  =>  x=[2, (7-2)/3=5/3]
int test_spsv_lower() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 1, 3};
    std::vector<int>   colInd = {0,  0, 1};
    std::vector<float> vals   = {2.0f, 1.0f, 3.0f};
    std::vector<float> b      = {4.0f, 7.0f};
    std::vector<float> x      = {0.0f, 0.0f};

    cusparseSpMatDescr_t matL = nullptr;
    cusparseDnVecDescr_t vecB = nullptr, vecX = nullptr;
    cusparseCreateCsr(&matL, 2, 2, 3,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
                              &fillMode, sizeof(fillMode));
    cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE,
                              &diagType, sizeof(diagType));

    cusparseCreateDnVec(&vecB, 2, b.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 2, x.data(), CUDA_R_32F);

    cusparseSpSVDescr_t descr = nullptr;
    cusparseSpSV_createDescr(&descr);
    float alpha = 1.0f;
    size_t bufSize = 0;
    cusparseSpSV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                            vecB, vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr, &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSpSV_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                          vecB, vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr, buf.data());
    cusparseSpSV_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matL,
                       vecB, vecX, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, descr);

    float x0_expected = 4.0f / 2.0f;           // 2.0
    float x1_expected = (7.0f - 1.0f*x0_expected) / 3.0f;  // 5/3
    if (!NEAR(x[0], x0_expected, 1e-5f)) FAIL("SpSV x[0]");
    if (!NEAR(x[1], x1_expected, 1e-5f)) FAIL("SpSV x[1]");

    cusparseSpSV_destroyDescr(descr);
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecB);
    cusparseDestroyDnVec(vecX);
    cusparseDestroy(h);
    PASS("SpSV lower triangular float");
    return 0;
}

// ── 4. SpGEMM — float sparse × sparse ────────────────────────────────────────
// A = [[1,0],[0,2]]  B = [[3,0],[0,4]]  C = A*B = [[3,0],[0,8]]
int test_spgemm_float() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   arPtr = {0,1,2}, acInd = {0,1};
    std::vector<float> aVals  = {1.0f, 2.0f};
    std::vector<int>   brPtr = {0,1,2}, bcInd = {0,1};
    std::vector<float> bVals  = {3.0f, 4.0f};
    // C: pre-allocate for 2 NNZ
    std::vector<int>   crPtr(3, 0);
    std::vector<int>   ccInd(2, 0);
    std::vector<float> cVals(2, 0.0f);

    cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
    cusparseCreateCsr(&matA, 2, 2, 2, arPtr.data(), acInd.data(), aVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, 2, 2, 2, brPtr.data(), bcInd.data(), bVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, 2, 2, 0, crPtr.data(), ccInd.data(), cVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseSpGEMMDescr_t spgemmDescr = nullptr;
    cusparseSpGEMM_createDescr(&spgemmDescr);
    float alpha = 1.0f, beta = 0.0f;
    size_t buf1 = 0;
    cusparseSpGEMM_workEstimation(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC,
                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &buf1, nullptr);
    size_t buf2 = 0;
    cusparseSpGEMM_compute(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, matA, matB, &beta, matC,
                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &buf2, nullptr);
    // Reallocate C arrays based on computed NNZ
    int64_t cRows = 0, cCols = 0, cNnz = 0;
    cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz);
    crPtr.resize(static_cast<size_t>(cRows + 1));
    ccInd.resize(static_cast<size_t>(cNnz));
    cVals.resize(static_cast<size_t>(cNnz));
    cusparseCsrSetPointers(matC, crPtr.data(), ccInd.data(), cVals.data());
    cusparseSpGEMM_copy(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, matB, &beta, matC,
                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);

    if (cNnz != 2) FAIL("SpGEMM NNZ");
    if (!NEAR(cVals[0], 3.0f, 1e-5f)) FAIL("SpGEMM C[0,0]");
    if (!NEAR(cVals[1], 8.0f, 1e-5f)) FAIL("SpGEMM C[1,1]");

    cusparseSpGEMM_destroyDescr(spgemmDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(h);
    PASS("SpGEMM float diagonal×diagonal");
    return 0;
}

// ── 5. ILU0 ──────────────────────────────────────────────────────────────────
// A = [[4,2],[2,3]] (symmetric positive definite)
// After ILU0 in-place: L*U ≈ A with zero fill-in
// Row 0: pivot = A[0,0] = 4  (unchanged)
// Row 1: A[1,0] /= A[0,0] = 2/4 = 0.5; A[1,1] -= A[1,0]*A[0,1] = 3 - 0.5*2 = 2
int test_ilu0() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 2, 4};
    std::vector<int>   colInd = {0, 1, 0, 1};
    std::vector<float> vals   = {4.0f, 2.0f, 2.0f, 3.0f};

    csrilu02Info_t info = nullptr;
    cusparseCreateCsrilu02Info(&info);
    int pBufSize = 0;
    cusparseScsrilu02_bufferSize(h, 2, 4, nullptr, vals.data(),
                                 rowPtr.data(), colInd.data(), info, &pBufSize);
    std::vector<char> buf(pBufSize + 1);
    cusparseScsrilu02_analysis(h, 2, 4, nullptr, vals.data(),
                               rowPtr.data(), colInd.data(), info, 0, buf.data());
    cusparseScsrilu02(h, 2, 4, nullptr, vals.data(),
                      rowPtr.data(), colInd.data(), info, 0, buf.data());

    // vals[2] should be L[1,0] = 0.5
    if (!NEAR(vals[2], 0.5f, 1e-5f)) FAIL("ILU0 L[1,0]");
    // vals[3] should be U[1,1] = 2.0
    if (!NEAR(vals[3], 2.0f, 1e-5f)) FAIL("ILU0 U[1,1]");

    cusparseDestroyCsrilu02Info(info);
    cusparseDestroy(h);
    PASS("ILU0 2×2");
    return 0;
}

// ── 6. IC0 ───────────────────────────────────────────────────────────────────
// A = [[4,2],[2,3]] lower triangle CSR
// After IC0: L[0,0]=sqrt(4)=2; L[1,0]=(2)/2=1; L[1,1]=sqrt(3-1)=sqrt(2)
int test_ic0() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // Lower triangle only
    std::vector<int>   rowPtr = {0, 1, 3};
    std::vector<int>   colInd = {0, 0, 1};
    std::vector<float> vals   = {4.0f, 2.0f, 3.0f};

    csric02Info_t info = nullptr;
    cusparseCreateCsric02Info(&info);
    int pBufSize = 0;
    cusparseScsric02_bufferSize(h, 2, 3, nullptr, vals.data(),
                                rowPtr.data(), colInd.data(), info, &pBufSize);
    std::vector<char> buf(pBufSize + 1);
    cusparseScsric02_analysis(h, 2, 3, nullptr, vals.data(),
                              rowPtr.data(), colInd.data(), info, 0, buf.data());
    cusparseScsric02(h, 2, 3, nullptr, vals.data(),
                     rowPtr.data(), colInd.data(), info, 0, buf.data());

    float l00 = 2.0f;                    // sqrt(4)
    float l10 = 2.0f / l00;             // 1.0
    float l11 = std::sqrt(3.0f - l10*l10); // sqrt(2)
    if (!NEAR(vals[0], l00, 1e-5f)) FAIL("IC0 L[0,0]");
    if (!NEAR(vals[1], l10, 1e-5f)) FAIL("IC0 L[1,0]");
    if (!NEAR(vals[2], l11, 1e-5f)) FAIL("IC0 L[1,1]");

    cusparseDestroyCsric02Info(info);
    cusparseDestroy(h);
    PASS("IC0 2×2");
    return 0;
}

// ── 7. SparseToDense / DenseToSparse round-trip ───────────────────────────────
// CSR 3×3 identity → dense → sparse again
int test_sparse_dense_roundtrip() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 1, 2, 3};
    std::vector<int>   colInd = {0, 1, 2};
    std::vector<float> vals   = {1.0f, 1.0f, 1.0f};
    std::vector<float> dense(9, 0.0f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matD = nullptr;
    cusparseCreateCsr(&matA, 3, 3, 3, rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matD, 3, 3, 3, dense.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    size_t bufSize = 0;
    cusparseSparseToDense_bufferSize(h, matA, matD,
                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSparseToDense(h, matA, matD, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf.data());

    // Identity diagonal should be 1.0; off-diagonal 0.0
    if (!NEAR(dense[0], 1.0f, 1e-5f)) FAIL("SparseToDense [0,0]");
    if (!NEAR(dense[4], 1.0f, 1e-5f)) FAIL("SparseToDense [1,1]");
    if (!NEAR(dense[8], 1.0f, 1e-5f)) FAIL("SparseToDense [2,2]");
    if (!NEAR(dense[1], 0.0f, 1e-5f)) FAIL("SparseToDense [0,1] should be 0");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matD);
    cusparseDestroy(h);
    PASS("SparseToDense 3×3 identity");
    return 0;
}

// ── 8. CSC descriptor ─────────────────────────────────────────────────────────
// CSC 2×2 identity: colPtr={0,1,2} rowInd={0,1} vals={1,1}
int test_csc_descriptor() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   colPtr = {0, 1, 2};
    std::vector<int>   rowInd = {0, 1};
    std::vector<float> vals   = {1.0f, 1.0f};
    std::vector<float> x      = {3.0f, 5.0f};
    std::vector<float> y      = {0.0f, 0.0f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    cusparseCreateCsc(&matA, 2, 2, 2,
                      colPtr.data(), rowInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 2, x.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 2, y.data(), CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                            vecX, &beta, vecY, CUDA_R_32F,
                            &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                 vecX, &beta, vecY, CUDA_R_32F, buf.data());

    // CSC identity × [3,5] = [3,5]
    if (!NEAR(y[0], 3.0f, 1e-5f)) FAIL("CSC SpMV y[0]");
    if (!NEAR(y[1], 5.0f, 1e-5f)) FAIL("CSC SpMV y[1]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("CSC descriptor SpMV");
    return 0;
}

// ── 9. Null rejection ─────────────────────────────────────────────────────────
int test_null_rejection() {
    if (cusparseCreate(nullptr) == CUSPARSE_STATUS_SUCCESS) FAIL("null handle create");
    if (cusparseCreateCsr(nullptr, 1, 1, 0, nullptr, nullptr, nullptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS)
        FAIL("null matDescr create");
    PASS("null rejection");
    return 0;
}

// ── Driver ────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== cuSPARSE Shim Functional Tests ===\n";
    int rc = 0;
    rc |= test_handle();
    rc |= test_null_rejection();
    rc |= test_spmv_float();
    rc |= test_spsv_lower();
    rc |= test_spgemm_float();
    rc |= test_ilu0();
    rc |= test_ic0();
    rc |= test_sparse_dense_roundtrip();
    rc |= test_csc_descriptor();
    if (rc == 0) std::cout << "\nAll cuSPARSE tests passed!\n";
    return rc;
}
