// cuSPARSE sparse-vector operation correctness tests.
// Tests: SpVV, Axpby, Gather, Scatter, Rot.

#include "vgre/api/cusparse_shim.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((float)(a)-(float)(b)) < (float)(eps))

// ── 1. cusparseSpVV: sparse dot product ──────────────────────────────────────
// x_sparse = [3.0, 4.0] at indices [1, 3]
// y_dense  = [0, 2, 0, 5]
// result = 3*2 + 4*5 = 26
int test_spvv() {
    cusparseHandle_t h = nullptr;
    if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS) FAIL("create handle");

    std::vector<int>   xIdx  = {1, 3};
    std::vector<float> xVals = {3.0f, 4.0f};
    std::vector<float> yVals = {0.0f, 2.0f, 0.0f, 5.0f};
    float result = 0.0f;

    cusparseDnVecDescr_t vecY = nullptr;
    cusparseCreateDnVec(&vecY, 4, yVals.data(), CUDA_R_32F);

    cusparseSpVecDescr_t vecX = nullptr;
    cusparseCreateSpVec(&vecX, 4, 2,
        xIdx.data(), xVals.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    size_t bufSize = 0;
    cusparseSpVV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        vecX, vecY, &result, CUDA_R_32F, CUSPARSE_SPVV_ALG_DEFAULT, &bufSize);

    cusparseStatus_t st = cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        vecX, vecY, &result, CUDA_R_32F, CUSPARSE_SPVV_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("SpVV failed");
    if (!NEAR(result, 26.0f, 1e-5f)) FAIL("SpVV result wrong: expected 26, got " + std::to_string(result));

    cusparseDestroySpVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("cusparseSpVV dot product = 26");
    return 0;
}

// ── 2. cusparseAxpby: y[x_idx[i]] = alpha*x[i] + beta*y[x_idx[i]] ───────────
// alpha=2, beta=1, x=[1,2] at [0,2], y=[10,0,10]
// y[0] = 2*1 + 1*10 = 12
// y[2] = 2*2 + 1*10 = 14
// y[1] unchanged = 0
int test_axpby() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   xIdx  = {0, 2};
    std::vector<float> xVals = {1.0f, 2.0f};
    std::vector<float> yVals = {10.0f, 0.0f, 10.0f};
    float alpha = 2.0f, beta = 1.0f;

    cusparseDnVecDescr_t vecY = nullptr;
    cusparseCreateDnVec(&vecY, 3, yVals.data(), CUDA_R_32F);

    cusparseSpVecDescr_t vecX = nullptr;
    cusparseCreateSpVec(&vecX, 3, 2,
        xIdx.data(), xVals.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseStatus_t st = cusparseAxpby(h, &alpha, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_AXPBY_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("Axpby failed");
    if (!NEAR(yVals[0], 12.0f, 1e-5f)) FAIL("Axpby y[0] wrong");
    if (!NEAR(yVals[1],  0.0f, 1e-5f)) FAIL("Axpby y[1] should be unchanged");
    if (!NEAR(yVals[2], 14.0f, 1e-5f)) FAIL("Axpby y[2] wrong");

    cusparseDestroySpVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("cusparseAxpby y=[12,0,14]");
    return 0;
}

// ── 3. cusparseGather: x[i] = y[x_idx[i]] ───────────────────────────────────
// y=[5,6,7], indices=[2,0] → x.values should become [7, 5]
int test_gather() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   xIdx  = {2, 0};
    std::vector<float> xVals = {0.0f, 0.0f};
    std::vector<float> yVals = {5.0f, 6.0f, 7.0f};

    cusparseDnVecDescr_t vecY = nullptr;
    cusparseCreateDnVec(&vecY, 3, yVals.data(), CUDA_R_32F);

    cusparseSpVecDescr_t vecX = nullptr;
    cusparseCreateSpVec(&vecX, 3, 2,
        xIdx.data(), xVals.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseStatus_t st = cusparseGather(h, vecY, vecX,
        CUSPARSE_GATHER_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("Gather failed");
    if (!NEAR(xVals[0], 7.0f, 1e-5f)) FAIL("Gather xVals[0] wrong");
    if (!NEAR(xVals[1], 5.0f, 1e-5f)) FAIL("Gather xVals[1] wrong");

    cusparseDestroySpVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("cusparseGather xVals=[7,5]");
    return 0;
}

// ── 4. cusparseScatter: y[x_idx[i]] = x[i] ───────────────────────────────────
// x=[9,8] at [0,2], y=[0,0,0] → y=[9,0,8]
int test_scatter() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   xIdx  = {0, 2};
    std::vector<float> xVals = {9.0f, 8.0f};
    std::vector<float> yVals = {0.0f, 0.0f, 0.0f};

    cusparseDnVecDescr_t vecY = nullptr;
    cusparseCreateDnVec(&vecY, 3, yVals.data(), CUDA_R_32F);

    cusparseSpVecDescr_t vecX = nullptr;
    cusparseCreateSpVec(&vecX, 3, 2,
        xIdx.data(), xVals.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseStatus_t st = cusparseScatter(h, vecX, vecY,
        CUSPARSE_SCATTER_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("Scatter failed");
    if (!NEAR(yVals[0], 9.0f, 1e-5f)) FAIL("Scatter y[0] wrong");
    if (!NEAR(yVals[1], 0.0f, 1e-5f)) FAIL("Scatter y[1] should be 0");
    if (!NEAR(yVals[2], 8.0f, 1e-5f)) FAIL("Scatter y[2] wrong");

    cusparseDestroySpVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("cusparseScatter y=[9,0,8]");
    return 0;
}

// ── 5. cusparseRot: Givens rotation ──────────────────────────────────────────
// c=0.6, s=0.8, x=[5.0] at idx 0, y=[0.0]
// x' = 0.6*5 + 0.8*0 = 3.0
// y' = -0.8*5 + 0.6*0 = -4.0
int test_rot() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    std::vector<int>   xIdx  = {0};
    std::vector<float> xVals = {5.0f};
    std::vector<float> yVals = {0.0f};
    float c = 0.6f, s = 0.8f;

    cusparseDnVecDescr_t vecY = nullptr;
    cusparseCreateDnVec(&vecY, 1, yVals.data(), CUDA_R_32F);

    cusparseSpVecDescr_t vecX = nullptr;
    cusparseCreateSpVec(&vecX, 1, 1,
        xIdx.data(), xVals.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseStatus_t st = cusparseRot(h, &c, &s, vecX, vecY,
        CUDA_R_32F, CUDA_R_32F, CUSPARSE_ROT_ALG_DEFAULT);
    if (st != CUSPARSE_STATUS_SUCCESS) FAIL("Rot failed");
    if (!NEAR(xVals[0],  3.0f, 1e-5f)) FAIL("Rot x' wrong");
    if (!NEAR(yVals[0], -4.0f, 1e-5f)) FAIL("Rot y' wrong");

    cusparseDestroySpVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(h);
    PASS("cusparseRot x'=3, y'=-4");
    return 0;
}

int main() {
    int r = 0;
    r |= test_spvv();
    r |= test_axpby();
    r |= test_gather();
    r |= test_scatter();
    r |= test_rot();
    if (r == 0) std::cout << "All cuSPARSE SpVec tests PASSED\n";
    return r;
}
