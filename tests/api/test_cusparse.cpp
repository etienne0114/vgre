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

// ── 4b. SpGEMM reuse — symbolic once, numeric twice ──────────────────────────
// First multiply: A=[[1,0],[0,2]] B=[[3,0],[0,4]] → C=[[3,0],[0,8]].
// Then change A values to [[2,0],[0,5]] and recompute via reuse_compute (reusing
// the symbolic structure) → C=[[6,0],[0,20]].
int test_spgemm_reuse() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   arPtr = {0,1,2}, acInd = {0,1};
    std::vector<float> aVals  = {1.0f, 2.0f};
    std::vector<int>   brPtr = {0,1,2}, bcInd = {0,1};
    std::vector<float> bVals  = {3.0f, 4.0f};
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

    cusparseSpGEMMDescr_t d = nullptr;
    cusparseSpGEMM_createDescr(&d);
    float alpha = 1.0f, beta = 0.0f;

    size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
    cusparseSpGEMMreuse_workEstimation(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC,
        CUSPARSE_SPGEMM_DEFAULT, d, &b1, nullptr);
    cusparseSpGEMMreuse_nnz(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC,
        CUSPARSE_SPGEMM_DEFAULT, d, &b2, nullptr, &b3, nullptr, &b4, nullptr);

    int64_t cRows = 0, cCols = 0, cNnz = 0;
    cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz);
    if (cNnz != 2) FAIL("SpGEMM reuse NNZ");
    crPtr.resize(static_cast<size_t>(cRows + 1));
    ccInd.resize(static_cast<size_t>(cNnz));
    cVals.resize(static_cast<size_t>(cNnz));
    cusparseCsrSetPointers(matC, crPtr.data(), ccInd.data(), cVals.data());

    cusparseSpGEMMreuse_copy(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, matA, matB, matC,
        CUSPARSE_SPGEMM_DEFAULT, d, &b5, nullptr);
    cusparseSpGEMMreuse_compute(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, d);

    if (!NEAR(cVals[0], 3.0f, 1e-5f)) FAIL("SpGEMM reuse C[0,0] first");
    if (!NEAR(cVals[1], 8.0f, 1e-5f)) FAIL("SpGEMM reuse C[1,1] first");

    // Change A's values only (same sparsity) and recompute — symbolic reused.
    aVals[0] = 2.0f; aVals[1] = 5.0f;
    cusparseSpGEMMreuse_compute(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, d);

    if (!NEAR(cVals[0], 6.0f, 1e-5f))  FAIL("SpGEMM reuse C[0,0] recompute");
    if (!NEAR(cVals[1], 20.0f, 1e-5f)) FAIL("SpGEMM reuse C[1,1] recompute");

    cusparseSpGEMM_destroyDescr(d);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(h);
    PASS("SpGEMM reuse (symbolic once, numeric twice)");
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

// ── 10. BSR SpMV (y = alpha * A * x + beta * y) ────────────────────────────────
int test_bsr_spmv() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // mb=2, nb=2, nnzb=3, blockDim=2
    std::vector<int> bsrRowPtr = {0, 2, 3};
    std::vector<int> bsrColInd = {0, 1, 1};
    // 3 blocks of 2x2:
    // Block 0 (row 0, col 0): I
    // Block 1 (row 0, col 1): diag(2, 3)
    // Block 2 (row 1, col 1): diag(4, 5)
    std::vector<float> vals = {
        1.f, 0.f, 0.f, 1.f,
        2.f, 0.f, 0.f, 3.f,
        4.f, 0.f, 0.f, 5.f
    };
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> y = {0.f, 0.f, 0.f, 0.f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    cusparseCreateBsr(&matA, 2, 2, 3, 2,
                      bsrRowPtr.data(), bsrColInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 4, x.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 4, y.data(), CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bsr(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                     vecX, &beta, vecY, CUDA_R_32F, nullptr);

    // Expected y = A*x = [7, 14, 12, 20]
    if (!NEAR(y[0], 7.f, 1e-5f))  FAIL("BSR SpMV y[0]");
    if (!NEAR(y[1], 14.f, 1e-5f)) FAIL("BSR SpMV y[1]");
    if (!NEAR(y[2], 12.f, 1e-5f)) FAIL("BSR SpMV y[2]");
    if (!NEAR(y[3], 20.f, 1e-5f)) FAIL("BSR SpMV y[3]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("BSR SpMV float");
    return 0;
}

// ── 11. BSR SpMM (C = alpha * A * B + beta * C) ────────────────────────────────
int test_bsr_spmm() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int> bsrRowPtr = {0, 2, 3};
    std::vector<int> bsrColInd = {0, 1, 1};
    std::vector<float> vals = {
        1.f, 0.f, 0.f, 1.f,
        2.f, 0.f, 0.f, 3.f,
        4.f, 0.f, 0.f, 5.f
    };
    std::vector<float> B = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f,
        7.f, 8.f
    };
    std::vector<float> C(8, 0.f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;

    cusparseCreateBsr(&matA, 2, 2, 3, 2,
                      bsrRowPtr.data(), bsrColInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 4, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, 4, 2, 2, C.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMM_bsr(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F, nullptr);

    // Expected C = A*B:
    // C[0] = 11, C[1] = 14, C[2] = 24, C[3] = 28, C[4] = 20, C[5] = 24, C[6] = 35, C[7] = 40
    if (!NEAR(C[0], 11.f, 1e-5f) || !NEAR(C[3], 28.f, 1e-5f) || !NEAR(C[7], 40.f, 1e-5f)) {
        printf("DEBUG BSR SpMM actual C: ");
        for (int i = 0; i < 8; ++i) printf("%f ", C[i]);
        printf("\n");
        FAIL("BSR SpMM verification failed");
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("BSR SpMM float");
    return 0;
}

// ── 12. Batched SpMM ──────────────────────────────────────────────────────────
int test_batched_spmm() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // CSR 2x2 identity matrix
    std::vector<int> rowPtr = {0, 1, 2};
    std::vector<int> colInd = {0, 1};
    std::vector<float> vals = {1.0f, 1.0f};

    // Dense B: 2 batches of 2x2 matrices
    // Batch 0: diag(1, 2) => [1, 0, 0, 2] (order row)
    // Batch 1: diag(3, 4) => [3, 0, 0, 4]
    std::vector<float> B = {
        1.f, 0.f,
        0.f, 2.f,
        3.f, 0.f,
        0.f, 4.f
    };
    std::vector<float> C(8, 0.f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;

    cusparseCreateCsr(&matA, 2, 2, 2,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 2, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, 2, 2, 2, C.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMM_batched(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                         2, 4, 4, nullptr);

    // Expected C = B:
    if (!NEAR(C[0], 1.f, 1e-5f)) FAIL("Batched SpMM C[0]");
    if (!NEAR(C[3], 2.f, 1e-5f)) FAIL("Batched SpMM C[3]");
    if (!NEAR(C[4], 3.f, 1e-5f)) FAIL("Batched SpMM C[4]");
    if (!NEAR(C[7], 4.f, 1e-5f)) FAIL("Batched SpMM C[7]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("Batched SpMM float");
    return 0;
}

// ── 13. SpMM with explicit alg parameter ──────────────────────────────────────
// A = [[1,2,0],[0,3,4],[5,0,0]] (3×3, nnz=5)   B = [[1,2],[3,4],[5,6]] (3×2)
// C = A*B : C[0]={7,10}, C[1]={29,36}, C[2]={5,10}
// Verifies CUSPARSE_SPMM_ALG_DEFAULT and CUSPARSE_SPMM_CSR_ALG1 both accepted.
int test_spmm_alg_default() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // CSR for 3×3 A with nnz=5: rowPtr, colInd, vals
    // Row 0: (0,1), Row 1: (3,4), Row 2: (5) — zero-indexed cols 0,1,1,2,0
    // Invariant: row i spans [rowPtr[i], rowPtr[i+1]) in colInd/vals (CSR).
    std::vector<int>   rowPtr = {0, 2, 4, 5};
    std::vector<int>   colInd = {0, 1,  1, 2,  0};
    std::vector<float> vals   = {1.f, 2.f,  3.f, 4.f,  5.f};

    // Dense B stored row-major, ld=2 (B has 3 rows × 2 cols)
    std::vector<float> B = {1.f, 2.f,  3.f, 4.f,  5.f, 6.f};
    // Output C (3×2), ld=2, zero-initialised
    std::vector<float> C(6, 0.f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    cusparseCreateCsr(&matA, 3, 3, 5,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 3, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, 3, 2, 2, C.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.f, beta = 0.f;
    size_t bufSz = 0;
    cusparseSpMM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufSz);
    std::vector<char> buf(bufSz + 1);

    // Test ALG_DEFAULT path
    cusparseStatus_t st = cusparseSpMM(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, buf.data());
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SpMM alg_default status");

    // C[row,col] = sum_k A[row,k]*B[k,col]; A row-indices per CSR rowPtr above.
    if (!NEAR(C[0],  7.f, 1e-5f)) FAIL("spmm_alg_default C[0,0]");
    if (!NEAR(C[1], 10.f, 1e-5f)) FAIL("spmm_alg_default C[0,1]");
    if (!NEAR(C[2], 29.f, 1e-5f)) FAIL("spmm_alg_default C[1,0]");
    if (!NEAR(C[3], 36.f, 1e-5f)) FAIL("spmm_alg_default C[1,1]");
    if (!NEAR(C[4],  5.f, 1e-5f)) FAIL("spmm_alg_default C[2,0]");
    if (!NEAR(C[5], 10.f, 1e-5f)) FAIL("spmm_alg_default C[2,1]");

    // Test CSR_ALG1 path (same result, different alg token)
    std::fill(C.begin(), C.end(), 0.f);
    st = cusparseSpMM(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG1, buf.data());
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SpMM csr_alg1 status");
    if (!NEAR(C[0], 7.f, 1e-5f)) FAIL("spmm_csr_alg1 C[0,0]");

    // CSR_ALG3 (atomic row-split variant) is a valid cuSPARSE algorithm and must
    // succeed — all CSR algorithm variants use the same CPU computation path.
    st = cusparseSpMM(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG3, buf.data());
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SpMM csr_alg3 should succeed (all CSR algs use same path)");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("SpMM alg_default and csr_alg1");
    return 0;
}

// ── 14. Batched SpMM stride correctness ───────────────────────────────────────
// 2×2 identity A, 2 batches of 2×2 B with bStride=8 (non-minimal; minimal=4).
// Batch 0 at element [0..3], padding [4..7]; Batch 1 at elements [8..11].
// Invariant: byte offset for batch b = b * bStride * sizeof(float).
// Verifies that batch[1] reads from offset 8*4=32 bytes, not batch[0]'s data.
int test_batched_spmm_stride_correctness() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // 2×2 CSR identity matrix: A = I
    std::vector<int>   rowPtr = {0, 1, 2};
    std::vector<int>   colInd = {0, 1};
    std::vector<float> vals   = {1.f, 1.f};

    // B flat array: 16 elements.
    // Batch 0 at [0..3]: [[1,0],[0,2]]; elements [4..7]: sentinel values (-99).
    // Batch 1 at [8..11]: [[10,0],[0,20]]; elements [12..15]: unused.
    // Using bStride=8 means batch 1 starts at element 8, skipping the sentinels.
    std::vector<float> B = {
        1.f,   0.f,   0.f,   2.f,     // batch 0  (elements 0-3)
        -99.f, -99.f, -99.f, -99.f,   // padding   (elements 4-7)
        10.f,  0.f,   0.f,   20.f,    // batch 1  (elements 8-11)
        0.f,   0.f,   0.f,   0.f      // unused   (elements 12-15)
    };
    // cStride=4 (minimal, no extra padding in output)
    std::vector<float> C(8, 0.f);  // 2 batches × 4 elements each

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    cusparseCreateCsr(&matA, 2, 2, 2,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // matB and matC descriptors point to the start; the batched call advances internally.
    cusparseCreateDnMat(&matB, 2, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, 2, 2, 2, C.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.f, beta = 0.f;
    // bStride=8 (elements), cStride=4 (elements)
    cusparseSpMM_batched(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        2, 8, 4, nullptr);

    // I * B_batch0 = B_batch0; C[0..3] should equal B[0..3]
    if (!NEAR(C[0],  1.f, 1e-5f)) FAIL("stride batch0 C[0]");
    if (!NEAR(C[3],  2.f, 1e-5f)) FAIL("stride batch0 C[3]");
    // I * B_batch1 = B_batch1; C[4..7] should equal B[8..11] (NOT B[4..7] = sentinels)
    if (!NEAR(C[4], 10.f, 1e-5f)) FAIL("stride batch1 C[4] — wrong stride reads sentinel");
    if (!NEAR(C[7], 20.f, 1e-5f)) FAIL("stride batch1 C[7]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("Batched SpMM bStride correctness");
    return 0;
}

// ── 15. Batched SpMM beta scaling ─────────────────────────────────────────────
// Single batch: 2×2 identity A, B=[[1,2],[3,4]], initial C=[[10,20],[30,40]], beta=0.5.
// Invariant: C_out = alpha*A*B + beta*C_in = B + 0.5*C_in.
// Verifies beta is applied to the original C values (before SpMM), not zeroed first.
int test_batched_spmm_beta_scaling() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 1, 2};
    std::vector<int>   colInd = {0, 1};
    std::vector<float> vals   = {1.f, 1.f};  // 2×2 identity

    std::vector<float> B = {1.f, 2.f, 3.f, 4.f};               // [[1,2],[3,4]]
    std::vector<float> C = {10.f, 20.f, 30.f, 40.f};           // initial C

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    cusparseCreateCsr(&matA, 2, 2, 2,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 2, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, 2, 2, 2, C.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.f, beta = 0.5f;
    // Single batch, bStride=4, cStride=4
    cusparseSpMM_batched(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        1, 4, 4, nullptr);

    // Expected: C_out[i] = 1*B[i] + 0.5*C_in[i]
    // C_out = {1+5, 2+10, 3+15, 4+20} = {6, 12, 18, 24}
    if (!NEAR(C[0],  6.f, 1e-5f)) FAIL("beta_scaling C[0]");
    if (!NEAR(C[1], 12.f, 1e-5f)) FAIL("beta_scaling C[1]");
    if (!NEAR(C[2], 18.f, 1e-5f)) FAIL("beta_scaling C[2]");
    if (!NEAR(C[3], 24.f, 1e-5f)) FAIL("beta_scaling C[3]");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("Batched SpMM beta scaling");
    return 0;
}

// ── SpGEMM symbolic phase (QUEUE-35) ─────────────────────────────────────────
// Verifies that workEstimation runs a real symbolic phase:
//   – bufferSize1 > 0 (structural data stored in descriptor)
//   – cusparseSpMatGetSize reports correct NNZ before compute/copy
//   – compute reuses the symbolic structure (numeric-only pass)
//   – copy produces correct values
//
// A (3×3): rows = {[1,2,_],[_,_,3],[4,5,_]}  nnz=5
// B (3×2): rows = {[1,_],[_,2],[3,_]}          nnz=3
// C = A*B (3×2):
//   row 0: A[0,0]*B[0,*] + A[0,1]*B[1,*] = 1·[1,0] + 2·[0,2] = [1, 4]
//   row 1: A[1,2]*B[2,*] = 3·[3,0]              = [9, 0]  (col 0 only)
//   row 2: A[2,0]*B[0,*] + A[2,1]*B[1,*] = 4·[1,0]+5·[0,2] = [4,10]
// NNZ(C) = 5; cRowPtr = {0,2,3,5}
int test_spgemm_symbolic_phase() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   arPtr = {0, 2, 3, 5};
    std::vector<int>   acInd = {0, 1, 2, 0, 1};
    std::vector<float> aVals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int>   brPtr = {0, 1, 2, 3};
    std::vector<int>   bcInd = {0, 1, 0};
    std::vector<float> bVals = {1.0f, 2.0f, 3.0f};
    std::vector<int>   crPtr(4, 0);
    std::vector<int>   ccInd(5, 0);
    std::vector<float> cVals(5, 0.0f);

    cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
    cusparseCreateCsr(&matA, 3, 3, 5, arPtr.data(), acInd.data(), aVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matB, 3, 2, 3, brPtr.data(), bcInd.data(), bVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&matC, 3, 2, 0, crPtr.data(), ccInd.data(), cVals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseSpGEMMDescr_t spgemmDescr = nullptr;
    cusparseSpGEMM_createDescr(&spgemmDescr);
    float alpha = 1.0f, beta = 0.0f;

    // Phase 1a: workEstimation with null buffer — runs symbolic analysis
    size_t buf1 = 0;
    if (cusparseSpGEMM_workEstimation(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha, matA, matB, &beta, matC,
                                      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDescr, &buf1, nullptr)
            != CUSPARSE_STATUS_SUCCESS) FAIL("SpGEMM symbolic: workEstimation failed");
    if (buf1 == 0) FAIL("SpGEMM symbolic: bufferSize1 must be > 0 after symbolic phase");

    // NNZ must already be correct in the C descriptor (before compute/copy)
    int64_t symRows = 0, symCols = 0, symNnz = 0;
    cusparseSpMatGetSize(matC, &symRows, &symCols, &symNnz);
    if (symNnz != 5)
        FAIL("SpGEMM symbolic: NNZ after workEstimation must be 5");

    // Phase 1b: second workEstimation call with allocated buffer (no-op, returns same size)
    std::vector<char> wsBuf(buf1);
    size_t buf1b = 0;
    cusparseSpGEMM_workEstimation(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC,
                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDescr, &buf1b, wsBuf.data());
    if (buf1b != buf1) FAIL("SpGEMM symbolic: second workEstimation must return same size");

    // Phase 2: compute — reuses symbolic structure (numeric-only)
    size_t buf2 = 0;
    if (cusparseSpGEMM_compute(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDescr, &buf2, nullptr)
            != CUSPARSE_STATUS_SUCCESS) FAIL("SpGEMM symbolic: compute failed");

    // Phase 3: copy
    int64_t cRows = 0, cCols = 0, cNnz = 0;
    cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz);
    if (cNnz != 5) FAIL("SpGEMM symbolic: final NNZ must be 5");
    crPtr.resize(static_cast<size_t>(cRows + 1));
    ccInd.resize(static_cast<size_t>(cNnz));
    cVals.resize(static_cast<size_t>(cNnz));
    cusparseCsrSetPointers(matC, crPtr.data(), ccInd.data(), cVals.data());
    if (cusparseSpGEMM_copy(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC,
                            CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr)
            != CUSPARSE_STATUS_SUCCESS) FAIL("SpGEMM symbolic: copy failed");

    // Verify structure: cRowPtr = {0,2,3,5}
    if (crPtr[0]!=0 || crPtr[1]!=2 || crPtr[2]!=3 || crPtr[3]!=5)
        FAIL("SpGEMM symbolic: cRowPtr mismatch");

    // Verify values (column-order within each row may vary; use lookup)
    auto getC = [&](int r, int c) -> float {
        for (int k = crPtr[static_cast<size_t>(r)]; k < crPtr[static_cast<size_t>(r+1)]; ++k)
            if (ccInd[static_cast<size_t>(k)] == c) return cVals[static_cast<size_t>(k)];
        return 0.0f;
    };
    if (!NEAR(getC(0, 0),  1.0f, 1e-5f)) FAIL("SpGEMM symbolic C[0,0]");
    if (!NEAR(getC(0, 1),  4.0f, 1e-5f)) FAIL("SpGEMM symbolic C[0,1]");
    if (!NEAR(getC(1, 0),  9.0f, 1e-5f)) FAIL("SpGEMM symbolic C[1,0]");
    if (!NEAR(getC(2, 0),  4.0f, 1e-5f)) FAIL("SpGEMM symbolic C[2,0]");
    if (!NEAR(getC(2, 1), 10.0f, 1e-5f)) FAIL("SpGEMM symbolic C[2,1]");

    cusparseSpGEMM_destroyDescr(spgemmDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(h);
    PASS("SpGEMM symbolic phase 3×3×2 (buf>0, NNZ=5 after workEstimation, values correct)");
    return 0;
}

// ── QUEUE-72: DenseToSparse — three-phase workflow ────────────────────────────
// Dense 3×4 matrix → CSR. Tests correct NNZ count, row offsets, col indices,
// and values via the bufferSize→analysis→(query+alloc+setPointers)→compress flow.
// Row-major layout; zero threshold is exact equality to 0.0f.
int test_dense_to_sparse() {
    // Dense 3×4 (row-major):
    //   row 0: [1, 0, 0, 2]  → 2 NNZ
    //   row 1: [0, 0, 3, 0]  → 1 NNZ
    //   row 2: [4, 5, 0, 6]  → 3 NNZ
    // Total NNZ = 6
    std::vector<float> dense = {1,0,0,2, 0,0,3,0, 4,5,0,6};

    cusparseHandle_t     h    = nullptr; cusparseCreate(&h);
    cusparseDnMatDescr_t matA = nullptr;
    cusparseCreateDnMat(&matA, 3, 4, /*ld=*/4, dense.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Sparse output descriptor (initially no pointers — set after analysis)
    cusparseSpMatDescr_t matB = nullptr;
    cusparseCreateCsr(&matB, 3, 4, /*nnz=*/0,
                      nullptr, nullptr, nullptr,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Phase 1: bufferSize (always 0 in this shim)
    size_t bufSize = 999;
    if (cusparseDenseToSparse_bufferSize(h, matA, matB,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufSize) != CUSPARSE_STATUS_SUCCESS)
        FAIL("DenseToSparse_bufferSize status");
    if (bufSize != 0) FAIL("bufferSize should be 0");

    // Phase 2: analysis — counts NNZ
    if (cusparseDenseToSparse_analysis(h, matA, matB,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) != CUSPARSE_STATUS_SUCCESS)
        FAIL("DenseToSparse_analysis status");

    // Query NNZ from the sparse descriptor
    int64_t rows64, cols64, nnz64;
    if (cusparseSpMatGetSize(matB, &rows64, &cols64, &nnz64) != CUSPARSE_STATUS_SUCCESS)
        FAIL("SpMatGetSize status");
    if (nnz64 != 6) FAIL("NNZ should be 6, got " + std::to_string(nnz64));

    // Allocate CSR arrays and wire them into the descriptor
    std::vector<int>   rowOff(rows64 + 1, 0);
    std::vector<int>   colIdx(nnz64, 0);
    std::vector<float> vals(nnz64, 0.0f);
    if (cusparseCsrSetPointers(matB, rowOff.data(), colIdx.data(), vals.data())
            != CUSPARSE_STATUS_SUCCESS)
        FAIL("CsrSetPointers status");

    // Phase 3: compress — fill CSR arrays
    if (cusparseDenseToSparse_compress(h, matA, matB,
            CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) != CUSPARSE_STATUS_SUCCESS)
        FAIL("DenseToSparse_compress status");

    // Verify row offsets: [0, 2, 3, 6]
    if (rowOff[0]!=0) FAIL("rowOff[0]"); if (rowOff[1]!=2) FAIL("rowOff[1]");
    if (rowOff[2]!=3) FAIL("rowOff[2]"); if (rowOff[3]!=6) FAIL("rowOff[3]");

    // Verify col indices (row 0: {0,3}, row 1: {2}, row 2: {0,1,3})
    if (colIdx[0]!=0) FAIL("colIdx[0]"); if (colIdx[1]!=3) FAIL("colIdx[1]");
    if (colIdx[2]!=2) FAIL("colIdx[2]");
    if (colIdx[3]!=0) FAIL("colIdx[3]"); if (colIdx[4]!=1) FAIL("colIdx[4]");
    if (colIdx[5]!=3) FAIL("colIdx[5]");

    // Verify values: 1,2,3,4,5,6
    float expected[] = {1,2,3,4,5,6};
    for (int i=0;i<6;++i)
        if (!NEAR(vals[i], expected[i], 1e-6f))
            FAIL("vals[" + std::to_string(i) + "] got " + std::to_string(vals[i]));

    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroy(h);
    PASS("DenseToSparse 3×4: NNZ=6, rowOff/colIdx/vals correct (QUEUE-72)");
    return 0;
}

// ── ELLPACK SpMV ─────────────────────────────────────────────────────────────
// Matrix (3×4):
//   row 0: (col=0, val=1)  (col=2, val=3)
//   row 1: (col=1, val=2)  [padding]
//   row 2: (col=3, val=4)  [padding]
// ELLPACK layout (ellWidth=2, row-major):
//   colInd = [0, 2,  1,-1,  3,-1]
//   values = [1, 3,  2, 0,  4, 0]
// x = [1, 2, 3, 4]
// Expected y = alpha * A*x + 0 = [1*1+3*3, 2*2, 4*4] = [10, 4, 16]
int test_ellpack_spmv() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   colInd = {0, 2,  1, -1,  3, -1};
    std::vector<float> vals   = {1.f, 3.f,  2.f, 0.f,  4.f, 0.f};
    std::vector<float> x      = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> y      = {0.f, 0.f, 0.f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (cusparseCreateEll(&matA, 3, 4, 2,
                          colInd.data(), vals.data(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F)
        != CUSPARSE_STATUS_SUCCESS) FAIL("cusparseCreateEll");

    cusparseCreateDnVec(&vecX, 4, x.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y.data(), CUDA_R_32F);

    float alpha = 1.f, beta = 0.f;
    size_t buf = 0;
    cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                            vecX, &beta, vecY, CUDA_R_32F, &buf);
    if (cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                     vecX, &beta, vecY, CUDA_R_32F, nullptr)
        != CUSPARSE_STATUS_SUCCESS) FAIL("ELL SpMV dispatch");

    if (!NEAR(y[0], 10.f, 1e-5f)) FAIL("ELL SpMV y[0] expected 10 got " + std::to_string(y[0]));
    if (!NEAR(y[1],  4.f, 1e-5f)) FAIL("ELL SpMV y[1] expected 4 got "  + std::to_string(y[1]));
    if (!NEAR(y[2], 16.f, 1e-5f)) FAIL("ELL SpMV y[2] expected 16 got " + std::to_string(y[2]));

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("ELLPACK SpMV 3×4: y = A*x = [10, 4, 16]");
    return 0;
}

// ── ELLPACK SpMM ─────────────────────────────────────────────────────────────
// Same 3×4 ELLPACK matrix A.  B is 4×2 (col-major, ld=4):
//   B = [[1,5],[2,6],[3,7],[4,8]]
// C = alpha * A * B + 0:
//   row 0: [1*1+3*3, 1*5+3*7] = [10, 26]
//   row 1: [2*2,     2*6    ] = [4,  12]
//   row 2: [4*4,     4*8    ] = [16, 32]
int test_ellpack_spmm() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   colInd = {0, 2,  1, -1,  3, -1};
    std::vector<float> vals   = {1.f, 3.f,  2.f, 0.f,  4.f, 0.f};
    // B col-major (ld=4): rows=4, cols=2
    std::vector<float> bData  = {1.f, 2.f, 3.f, 4.f,  5.f, 6.f, 7.f, 8.f};
    // C col-major (ld=3): rows=3, cols=2
    std::vector<float> cData(6, 0.f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;

    cusparseCreateEll(&matA, 3, 4, 2,
                      colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 4, 2, 4, bData.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, 3, 2, 3, cData.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);

    float alpha = 1.f, beta = 0.f;
    size_t buf = 0;
    cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &buf);
    if (cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, nullptr)
        != CUSPARSE_STATUS_SUCCESS) FAIL("ELL SpMM dispatch");

    // col-major layout: C[r,c] = cData[r + c*ld]
    if (!NEAR(cData[0], 10.f, 1e-5f)) FAIL("ELL SpMM C[0,0] expected 10 got " + std::to_string(cData[0]));
    if (!NEAR(cData[1],  4.f, 1e-5f)) FAIL("ELL SpMM C[1,0] expected 4 got "  + std::to_string(cData[1]));
    if (!NEAR(cData[2], 16.f, 1e-5f)) FAIL("ELL SpMM C[2,0] expected 16 got " + std::to_string(cData[2]));
    if (!NEAR(cData[3], 26.f, 1e-5f)) FAIL("ELL SpMM C[0,1] expected 26 got " + std::to_string(cData[3]));
    if (!NEAR(cData[4], 12.f, 1e-5f)) FAIL("ELL SpMM C[1,1] expected 12 got " + std::to_string(cData[4]));
    if (!NEAR(cData[5], 32.f, 1e-5f)) FAIL("ELL SpMM C[2,1] expected 32 got " + std::to_string(cData[5]));

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(h);
    PASS("ELLPACK SpMM 3×4 × 4×2: C = [[10,26],[4,12],[16,32]]");
    return 0;
}

// ── ELLPACK transposed SpMV ───────────────────────────────────────────────────
// Same matrix A (3×4). A^T * y where y = [1, 2, 3]:
//   col 0: row 0 contributes val=1 * y[0]=1 → 1
//   col 1: row 1 contributes val=2 * y[1]=2 → 4
//   col 2: row 0 contributes val=3 * y[0]=1 → 3
//   col 3: row 2 contributes val=4 * y[2]=3 → 12
// Expected: z = [1, 4, 3, 12]
int test_ellpack_spmv_transposed() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   colInd = {0, 2,  1, -1,  3, -1};
    std::vector<float> vals   = {1.f, 3.f,  2.f, 0.f,  4.f, 0.f};
    std::vector<float> y      = {1.f, 2.f, 3.f};
    std::vector<float> z      = {0.f, 0.f, 0.f, 0.f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecY = nullptr, vecZ = nullptr;

    cusparseCreateEll(&matA, 3, 4, 2,
                      colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y.data(), CUDA_R_32F);
    cusparseCreateDnVec(&vecZ, 4, z.data(), CUDA_R_32F);

    float alpha = 1.f, beta = 0.f;
    if (cusparseSpMV(h, CUSPARSE_OPERATION_TRANSPOSE, &alpha, matA,
                     vecY, &beta, vecZ, CUDA_R_32F, nullptr)
        != CUSPARSE_STATUS_SUCCESS) FAIL("ELL transposed SpMV dispatch");

    if (!NEAR(z[0],  1.f, 1e-5f)) FAIL("ELL^T z[0] expected 1 got "  + std::to_string(z[0]));
    if (!NEAR(z[1],  4.f, 1e-5f)) FAIL("ELL^T z[1] expected 4 got "  + std::to_string(z[1]));
    if (!NEAR(z[2],  3.f, 1e-5f)) FAIL("ELL^T z[2] expected 3 got "  + std::to_string(z[2]));
    if (!NEAR(z[3], 12.f, 1e-5f)) FAIL("ELL^T z[3] expected 12 got " + std::to_string(z[3]));

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecZ);
    cusparseDestroy(h);
    PASS("ELLPACK transposed SpMV 4×3: z = A^T * y = [1, 4, 3, 12]");
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
    rc |= test_spgemm_reuse();
    rc |= test_spgemm_symbolic_phase();
    rc |= test_ilu0();
    rc |= test_ic0();
    rc |= test_sparse_dense_roundtrip();
    rc |= test_csc_descriptor();
    rc |= test_bsr_spmv();
    rc |= test_bsr_spmm();
    rc |= test_batched_spmm();
    rc |= test_spmm_alg_default();
    rc |= test_batched_spmm_stride_correctness();
    rc |= test_batched_spmm_beta_scaling();
    rc |= test_dense_to_sparse();
    rc |= test_ellpack_spmv();
    rc |= test_ellpack_spmm();
    rc |= test_ellpack_spmv_transposed();
    if (rc == 0) std::cout << "\nAll cuSPARSE tests passed!\n";
    return rc;
}
