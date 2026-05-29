// cuSPARSE SpSM and SDDMM functional tests.
// SpSM: sparse triangular solve with matrix RHS (multi-column).
// SDDMM: sampled dense-dense matrix multiplication.

#include "vgre/api/cusparse_shim.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((a)-(b)) < (eps))

// ── 1. SpSM descriptor lifecycle ─────────────────────────────────────────────
int test_spsm_descr_lifecycle() {
    cusparseHandle_t h = nullptr;
    if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS || !h) FAIL("create handle");

    cusparseSpSMDescr_t descr = nullptr;
    if (cusparseSpSM_createDescr(&descr) != CUSPARSE_STATUS_SUCCESS || !descr)
        FAIL("SpSM createDescr");
    if (cusparseSpSM_destroyDescr(descr) != CUSPARSE_STATUS_SUCCESS)
        FAIL("SpSM destroyDescr");

    cusparseDestroy(h);
    PASS("SpSM descriptor lifecycle");
    return 0;
}

// ── 2. SpSM: lower triangular solve with 2-column RHS ────────────────────────
// L = [[2,0],[1,3]]  (lower triangular, non-unit)
// B (RHS matrix, 2 columns): col0=[4,7], col1=[6,9]
// L * X = B  =>  X col0: x0=4/2=2, x1=(7-1*2)/3=5/3
//               X col1: x0=6/2=3, x1=(9-1*3)/3=2
int test_spsm_lower_triangular() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // CSR for L = [[2,0],[1,3]] (lower triangle: store both diag and lower off-diag)
    std::vector<int>    rowPtr = {0, 1, 3};
    std::vector<int>    colInd = {0, 0, 1};
    std::vector<float>  vals   = {2.0f, 1.0f, 3.0f};

    // RHS matrix B: 2 rows, 2 cols, column-major: col0=[4,7] col1=[6,9]
    std::vector<float>  B = {4.0f, 7.0f, 6.0f, 9.0f};
    std::vector<float>  X(4, 0.0f);

    cusparseSpMatDescr_t matL = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matX = nullptr;
    cusparseCreateCsr(&matL, 2, 2, 3,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matB, 2, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matX, 2, 2, 2, X.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);

    cusparseSpSMDescr_t descr = nullptr;
    cusparseSpSM_createDescr(&descr);

    float alpha = 1.0f;
    size_t bufSize = 0;
    cusparseSpSM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr, &bufSize);

    std::vector<char> buf(bufSize + 1);
    cusparseSpSM_analysis(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr, buf.data());

    cusparseSpSM_solve(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr);

    // X col0: [2.0, 5/3]  col1: [3.0, 2.0]  (column-major layout)
    if (!NEAR(X[0], 2.0f,       1e-5f)) FAIL("SpSM X[0,0] expected 2.0");
    if (!NEAR(X[1], 5.0f/3.0f,  1e-4f)) FAIL("SpSM X[1,0] expected 5/3");
    if (!NEAR(X[2], 3.0f,       1e-5f)) FAIL("SpSM X[0,1] expected 3.0");
    if (!NEAR(X[3], 2.0f,       1e-5f)) FAIL("SpSM X[1,1] expected 2.0");

    cusparseSpSM_destroyDescr(descr);
    cusparseDestroySpMat(matL);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matX);
    cusparseDestroy(h);
    PASS("SpSM 2×2 lower triangular 2-column RHS");
    return 0;
}

// ── 3. SpSM: upper triangular 3×3 single column ──────────────────────────────
// U = [[4,2,1],[0,3,5],[0,0,6]]  x_ref=[1,1,1]  b = U*x_ref = [7,8,6]
// U * x = b  => x[2]=6/6=1, x[1]=(8-5)/3=1, x[0]=(7-2-1)/4=1
int test_spsm_upper_triangular() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 3, 5, 6};
    std::vector<int>   colInd = {0, 1, 2, 1, 2, 2};
    std::vector<float> vals   = {4.0f, 2.0f, 1.0f, 3.0f, 5.0f, 6.0f};

    std::vector<float> B = {7.0f, 8.0f, 6.0f}; // single column
    std::vector<float> X(3, 0.0f);

    cusparseSpMatDescr_t matU = nullptr;
    cusparseDnMatDescr_t matB = nullptr, matX = nullptr;
    cusparseCreateCsr(&matU, 3, 3, 6,
                      rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Mark as upper triangular so the solver uses backward substitution
    cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_UPPER;
    cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
    cusparseCreateDnMat(&matB, 3, 1, 3, B.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matX, 3, 1, 3, X.data(), CUDA_R_32F, CUSPARSE_ORDER_COL);

    cusparseSpSMDescr_t descr = nullptr;
    cusparseSpSM_createDescr(&descr);

    float alpha = 1.0f;
    size_t bufSize = 0;
    cusparseSpSM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr, &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSpSM_analysis(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr, buf.data());
    cusparseSpSM_solve(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, matB, matX,
        CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, descr);

    if (!NEAR(X[0], 1.0f, 1e-4f)) FAIL("SpSM upper X[0] expected 1.0 got " << X[0]);
    if (!NEAR(X[1], 1.0f, 1e-4f)) FAIL("SpSM upper X[1] expected 1.0 got " << X[1]);
    if (!NEAR(X[2], 1.0f, 1e-4f)) FAIL("SpSM upper X[2] expected 1.0 got " << X[2]);

    cusparseSpSM_destroyDescr(descr);
    cusparseDestroySpMat(matU);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matX);
    cusparseDestroy(h);
    PASS("SpSM 3×3 upper triangular single-column");
    return 0;
}

// ── 4. SDDMM buffer-size and preprocess return SUCCESS ───────────────────────
int test_sddmm_lifecycle() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // Sparse C: 3×3 identity pattern
    std::vector<int>   rowPtr = {0, 1, 2, 3};
    std::vector<int>   colInd = {0, 1, 2};
    std::vector<float> vals   = {0.0f, 0.0f, 0.0f};
    // Dense A: 3×2, B: 3×2 (opB=T so B^T is 2×3)
    std::vector<float> A = {1,2, 3,4, 5,6};
    std::vector<float> B = {1,0, 0,1, 1,1};

    cusparseSpMatDescr_t matC = nullptr;
    cusparseDnMatDescr_t matA = nullptr, matBm = nullptr;
    cusparseCreateCsr(&matC, 3, 3, 3, rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matA, 3, 2, 2, A.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matBm, 3, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    cusparseStatus_t st = cusparseSDDMM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SDDMM bufferSize error");

    std::vector<char> buf(bufSize + 1);
    st = cusparseSDDMM_preprocess(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buf.data());
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SDDMM preprocess error");

    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matBm);
    cusparseDestroy(h);
    PASS("SDDMM lifecycle (bufferSize + preprocess)");
    return 0;
}

// ── 5. SDDMM correctness: C = alpha*(C_pat ⊙ (A*B^T)) + beta*C ───────────────
// A = [[1,0],[0,1],[1,1]] (3×2)  B = [[1,0],[0,1],[1,1]] (3×2)
// A*B^T (3×3): [1,0,1; 0,1,1; 1,1,2]
// Sparse C pattern (diagonal): sample (0,0),(1,1),(2,2)
// Expected C values: 1, 1, 2  (alpha=1, beta=0)
int test_sddmm_correctness() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 1, 2, 3};
    std::vector<int>   colInd = {0, 1, 2};
    std::vector<float> vals   = {0.0f, 0.0f, 0.0f};

    std::vector<float> A = {1.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f};
    std::vector<float> B = {1.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f};

    cusparseSpMatDescr_t matC = nullptr;
    cusparseDnMatDescr_t matA = nullptr, matBm = nullptr;
    cusparseCreateCsr(&matC, 3, 3, 3, rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matA, 3, 2, 2, A.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matBm, 3, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    cusparseSDDMM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSDDMM_preprocess(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buf.data());
    cusparseStatus_t st = cusparseSDDMM(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buf.data());
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SDDMM solve error");

    // diagonal of A*B^T = [dot(A[0],B[0]), dot(A[1],B[1]), dot(A[2],B[2])] = [1,1,2]
    if (!NEAR(vals[0], 1.0f, 1e-5f)) FAIL("SDDMM vals[0] expected 1.0 got " << vals[0]);
    if (!NEAR(vals[1], 1.0f, 1e-5f)) FAIL("SDDMM vals[1] expected 1.0 got " << vals[1]);
    if (!NEAR(vals[2], 2.0f, 1e-5f)) FAIL("SDDMM vals[2] expected 2.0 got " << vals[2]);

    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matBm);
    cusparseDestroy(h);
    PASS("SDDMM diagonal pattern correctness");
    return 0;
}

// ── 6. SDDMM beta accumulation ───────────────────────────────────────────────
// Same as above but beta=2, initial vals=[10,10,10]
// Expected: C[i] = 1*sampled + 2*10 => [21, 21, 22]
int test_sddmm_beta_accumulate() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   rowPtr = {0, 1, 2, 3};
    std::vector<int>   colInd = {0, 1, 2};
    std::vector<float> vals   = {10.0f, 10.0f, 10.0f};

    std::vector<float> A = {1.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f};
    std::vector<float> B = {1.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f};

    cusparseSpMatDescr_t matC = nullptr;
    cusparseDnMatDescr_t matA = nullptr, matBm = nullptr;
    cusparseCreateCsr(&matC, 3, 3, 3, rowPtr.data(), colInd.data(), vals.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnMat(&matA, 3, 2, 2, A.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matBm, 3, 2, 2, B.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.0f, beta = 2.0f;
    size_t bufSize = 0;
    cusparseSDDMM_bufferSize(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufSize);
    std::vector<char> buf(bufSize + 1);
    cusparseSDDMM_preprocess(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buf.data());
    cusparseSDDMM(h,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, matA, matBm, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, buf.data());

    if (!NEAR(vals[0], 21.0f, 1e-4f)) FAIL("SDDMM beta vals[0] expected 21.0 got " << vals[0]);
    if (!NEAR(vals[1], 21.0f, 1e-4f)) FAIL("SDDMM beta vals[1] expected 21.0 got " << vals[1]);
    if (!NEAR(vals[2], 22.0f, 1e-4f)) FAIL("SDDMM beta vals[2] expected 22.0 got " << vals[2]);

    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matBm);
    cusparseDestroy(h);
    PASS("SDDMM beta accumulation");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_spsm_descr_lifecycle();
    failures += test_spsm_lower_triangular();
    failures += test_spsm_upper_triangular();
    failures += test_sddmm_lifecycle();
    failures += test_sddmm_correctness();
    failures += test_sddmm_beta_accumulate();
    if (failures == 0)
        std::cout << "All SpSM/SDDMM tests passed.\n";
    return failures;
}
