/**
 * Test: cuDNN OpTensor, ReduceTensor, TransformTensor (P3.12, P3.13, P3.14)
 *
 * Verifies elementwise tensor operations and reductions.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnOpTensorDescriptor_t;
typedef void* cudnnReduceTensorDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };
enum cudnnNanPropagation_t { CUDNN_NOT_PROPAGATE_NAN = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

// OpTensor
enum cudnnOpTensorOp_t {
    CUDNN_OP_TENSOR_ADD = 0,
    CUDNN_OP_TENSOR_MUL = 1,
    CUDNN_OP_TENSOR_MIN = 2,
    CUDNN_OP_TENSOR_MAX = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT = 5
};
cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t d);
cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t d,
    cudnnOpTensorOp_t op, cudnnDataType_t compType, cudnnNanPropagation_t nanOpt);
cudnnStatus_t cudnnOpTensor(cudnnHandle_t,
    cudnnOpTensorDescriptor_t opDesc,
    const void* alpha1, cudnnTensorDescriptor_t aDesc, const void* A,
    const void* alpha2, cudnnTensorDescriptor_t bDesc, const void* B,
    const void* beta,  cudnnTensorDescriptor_t cDesc, void* C);

// ReduceTensor
enum cudnnReduceTensorOp_t {
    CUDNN_REDUCE_TENSOR_ADD = 0,
    CUDNN_REDUCE_TENSOR_MUL = 1,
    CUDNN_REDUCE_TENSOR_MIN = 2,
    CUDNN_REDUCE_TENSOR_MAX = 3,
    CUDNN_REDUCE_TENSOR_AMAX = 4,
    CUDNN_REDUCE_TENSOR_AVG = 5,
    CUDNN_REDUCE_TENSOR_NORM1 = 6,
    CUDNN_REDUCE_TENSOR_NORM2 = 7
};
cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t d);
cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t d,
    cudnnReduceTensorOp_t op, cudnnDataType_t compType, cudnnNanPropagation_t nanOpt,
    cudnnReduceTensorOp_t idxCompType, cudnnDataType_t outType);
cudnnStatus_t cudnnReduceTensor(cudnnHandle_t,
    cudnnReduceTensorDescriptor_t reduceDesc,
    void* indices, size_t indicesSizeInBytes,
    void* workspace, size_t workspaceSizeInBytes,
    const void* alpha, cudnnTensorDescriptor_t aDesc, const void* A,
    const void* beta, cudnnTensorDescriptor_t cDesc, void* C);

// TransformTensor
cudnnStatus_t cudnnTransformTensor(cudnnHandle_t,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y);
}

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuDNN OpTensor / ReduceTensor / TransformTensor ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    int N=1, C=2, H=2, W=2;
    int totalEls = N*C*H*W;
    std::vector<float> a(totalEls), b(totalEls), c(totalEls, 0.f);
    for (int i=0; i<totalEls; ++i) { a[i] = static_cast<float>(i+1); b[i] = static_cast<float>(i+2); }

    cudnnTensorDescriptor_t aDesc, bDesc, cDesc, scalarDesc;
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&cDesc);
    cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&scalarDesc);
    cudnnSetTensor4dDescriptor(scalarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,1,1,1);

    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;

    // ── OpTensor ADD ────────────────────────────────────────────────────
    ++total;
    {
        cudnnOpTensorDescriptor_t opDesc;
        cudnnCreateOpTensorDescriptor(&opDesc);
        cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
        std::fill(c.begin(), c.end(), 0.f);
        auto r = cudnnOpTensor(handle, opDesc, &alpha1, aDesc, a.data(), &alpha2, bDesc, b.data(), &beta, cDesc, c.data());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        for (int i=0; i<totalEls && ok; ++i) {
            if (!approx_eq(c[i], a[i]+b[i], 1e-3f)) {
                std::cerr << "FAIL [OpTensor ADD] c[" << i << "]=" << c[i] << " expected=" << a[i]+b[i] << "\n";
                ok = false;
            }
        }
        if (ok) { std::cout << "PASS [OpTensor ADD]\n"; ++pass; }
        cudnnDestroyOpTensorDescriptor(opDesc);
    }

    // ── OpTensor MUL ────────────────────────────────────────────────────
    ++total;
    {
        cudnnOpTensorDescriptor_t opDesc;
        cudnnCreateOpTensorDescriptor(&opDesc);
        cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
        std::fill(c.begin(), c.end(), 0.f);
        auto r = cudnnOpTensor(handle, opDesc, &alpha1, aDesc, a.data(), &alpha2, bDesc, b.data(), &beta, cDesc, c.data());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        for (int i=0; i<totalEls && ok; ++i) {
            if (!approx_eq(c[i], a[i]*b[i], 1e-3f)) {
                std::cerr << "FAIL [OpTensor MUL] c[" << i << "]=" << c[i] << " expected=" << a[i]*b[i] << "\n";
                ok = false;
            }
        }
        if (ok) { std::cout << "PASS [OpTensor MUL]\n"; ++pass; }
        cudnnDestroyOpTensorDescriptor(opDesc);
    }

    // ── OpTensor MAX ────────────────────────────────────────────────────
    ++total;
    {
        cudnnOpTensorDescriptor_t opDesc;
        cudnnCreateOpTensorDescriptor(&opDesc);
        cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
        std::fill(c.begin(), c.end(), 0.f);
        auto r = cudnnOpTensor(handle, opDesc, &alpha1, aDesc, a.data(), &alpha2, bDesc, b.data(), &beta, cDesc, c.data());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        for (int i=0; i<totalEls && ok; ++i) {
            if (!approx_eq(c[i], std::max(a[i], b[i]), 1e-3f)) {
                std::cerr << "FAIL [OpTensor MAX] c[" << i << "]=" << c[i] << " expected=" << std::max(a[i],b[i]) << "\n";
                ok = false;
            }
        }
        if (ok) { std::cout << "PASS [OpTensor MAX]\n"; ++pass; }
        cudnnDestroyOpTensorDescriptor(opDesc);
    }

    // ── ReduceTensor ADD ────────────────────────────────────────────────
    ++total;
    {
        cudnnReduceTensorDescriptor_t redDesc;
        cudnnCreateReduceTensorDescriptor(&redDesc);
        cudnnSetReduceTensorDescriptor(redDesc, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT);
        float result = 99.f;
        auto r = cudnnReduceTensor(handle, redDesc, nullptr, 0, nullptr, 0,
                                    &alpha1, aDesc, a.data(), &beta, scalarDesc, &result);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        double expected = 0;
        for (float v : a) expected += v;
        if (ok && !approx_eq(result, static_cast<float>(expected), 1e-3f)) {
            std::cerr << "FAIL [Reduce ADD] result=" << result << " expected=" << expected << "\n";
            ok = false;
        }
        if (ok) { std::cout << "PASS [ReduceTensor ADD]\n"; ++pass; }
        cudnnDestroyReduceTensorDescriptor(redDesc);
    }

    // ── ReduceTensor AVG ────────────────────────────────────────────────
    ++total;
    {
        cudnnReduceTensorDescriptor_t redDesc;
        cudnnCreateReduceTensorDescriptor(&redDesc);
        cudnnSetReduceTensorDescriptor(redDesc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT);
        float result = 99.f;
        auto r = cudnnReduceTensor(handle, redDesc, nullptr, 0, nullptr, 0,
                                    &alpha1, aDesc, a.data(), &beta, scalarDesc, &result);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        double expected = 0;
        for (float v : a) expected += v;
        expected /= totalEls;
        if (ok && !approx_eq(result, static_cast<float>(expected), 1e-3f)) {
            std::cerr << "FAIL [Reduce AVG] result=" << result << " expected=" << expected << "\n";
            ok = false;
        }
        if (ok) { std::cout << "PASS [ReduceTensor AVG]\n"; ++pass; }
        cudnnDestroyReduceTensorDescriptor(redDesc);
    }

    // ── TransformTensor ─────────────────────────────────────────────────
    ++total;
    {
        std::vector<float> y(totalEls, 5.f);
        auto r = cudnnTransformTensor(handle, &alpha1, aDesc, a.data(), &beta, cDesc, y.data());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        for (int i=0; i<totalEls && ok; ++i) {
            if (!approx_eq(y[i], a[i], 1e-3f)) {
                std::cerr << "FAIL [Transform] y[" << i << "]=" << y[i] << " expected=" << a[i] << "\n";
                ok = false;
            }
        }
        if (ok) { std::cout << "PASS [TransformTensor]\n"; ++pass; }
    }

    // ── Null pointer rejection ──────────────────────────────────────────
    ++total;
    {
        auto r = cudnnOpTensor(handle, nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyTensorDescriptor(scalarDesc);
    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
