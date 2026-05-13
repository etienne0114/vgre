/**
 * Test: cuDNN Pooling Backward (P3.7)
 *
 * Verifies cudnnPoolingBackward computes dx correctly
 * for MAX and AVERAGE pooling modes.
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
typedef void* cudnnPoolingDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };
enum cudnnPoolingMode_t {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
};

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* d);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t d);
cudnnStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t d, cudnnPoolingMode_t mode, int /*nanOpt*/,
    int winH, int winW, int padH, int padW, int strH, int strW);

cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t, cudnnPoolingDescriptor_t pd,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    cudnnTensorDescriptor_t yDesc, void* y);

cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t, cudnnPoolingDescriptor_t pd,
    const void* alpha,
    cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx);
}

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuDNN Pooling Backward (P3.7) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    // ── Test 1: MAX pooling backward ──────────────────────────────────────
    ++total;
    {
        int N=1, C=1, H=4, W=4;
        float xData[16] = {
            1,2,3,4,
            5,6,7,8,
            9,10,11,12,
            13,14,15,16
        };
        float dyData[4] = {1,2,3,4};

        cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,2,2);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,2,2);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, 0, 2,2, 0,0, 2,2);

        std::vector<float> y(4, 0.f), dx(16, 0.f);
        float alpha = 1.0f, beta = 0.0f;
        auto r1 = cudnnPoolingForward(handle, poolDesc, &alpha, xDesc, xData, &beta, yDesc, y.data());
        bool ok = (r1 == CUDNN_STATUS_SUCCESS);

        if (ok) {
            auto r2 = cudnnPoolingBackward(handle, poolDesc, &alpha,
                                            yDesc, y.data(), dyDesc, dyData,
                                            xDesc, xData, &beta, dxDesc, dx.data());
            ok = (r2 == CUDNN_STATUS_SUCCESS);

            if (ok) {
                // For MAX pool 2x2 stride 2:
                // (0,0): max=6 at (1,1) -> dx[1*4+1]=1
                // (0,1): max=8 at (1,3) -> dx[1*4+3]=2
                // (1,0): max=14 at (3,1) -> dx[3*4+1]=3
                // (1,1): max=16 at (3,3) -> dx[3*4+3]=4
                bool match =
                    approx_eq(dx[0], 0.f) && approx_eq(dx[1], 0.f) &&
                    approx_eq(dx[2], 0.f) && approx_eq(dx[3], 0.f) &&
                    approx_eq(dx[4], 0.f) && approx_eq(dx[5], 1.f) &&
                    approx_eq(dx[6], 0.f) && approx_eq(dx[7], 2.f) &&
                    approx_eq(dx[8], 0.f) && approx_eq(dx[9], 0.f) &&
                    approx_eq(dx[10], 0.f) && approx_eq(dx[11], 0.f) &&
                    approx_eq(dx[12], 0.f) && approx_eq(dx[13], 3.f) &&
                    approx_eq(dx[14], 0.f) && approx_eq(dx[15], 4.f);
                if (!match) {
                    std::cerr << "FAIL [MAX pool] dx values incorrect\n";
                    for (int i=0; i<16; ++i)
                        std::cerr << "  dx[" << i << "]=" << dx[i] << "\n";
                    ok = false;
                }
            } else {
                std::cerr << "FAIL [MAX pool] backward status=" << r2 << "\n";
            }
        } else {
            std::cerr << "FAIL [MAX pool] forward status=" << r1 << "\n";
        }

        if (ok) { std::cout << "PASS [MAX pool backward]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyPoolingDescriptor(poolDesc);
    }

    // ── Test 2: AVERAGE pooling backward ──────────────────────────────────
    ++total;
    {
        int N=1, C=1, H=4, W=4;
        float xData[16];
        for (int i=0; i<16; ++i) xData[i] = static_cast<float>(i);
        float dyData[4] = {1,1,1,1};

        cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
        cudnnPoolingDescriptor_t poolDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,2,2);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,2,2);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreatePoolingDescriptor(&poolDesc);
        cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 0, 2,2, 0,0, 2,2);

        std::vector<float> y(4, 0.f), dx(16, 0.f);
        float alpha = 1.0f, beta = 0.0f;
        auto r1 = cudnnPoolingForward(handle, poolDesc, &alpha, xDesc, xData, &beta, yDesc, y.data());
        bool ok = (r1 == CUDNN_STATUS_SUCCESS);

        if (ok) {
            auto r2 = cudnnPoolingBackward(handle, poolDesc, &alpha,
                                            yDesc, y.data(), dyDesc, dyData,
                                            xDesc, xData, &beta, dxDesc, dx.data());
            ok = (r2 == CUDNN_STATUS_SUCCESS);

            if (ok) {
                // AVG 2x2 no pad: each window has 4 elements, dy=1
                // dx for each window element = 1/4 = 0.25
                bool match = true;
                for (int i=0; i<16 && match; ++i) {
                    if (!approx_eq(dx[i], 0.25f, 1e-3f)) {
                        std::cerr << "FAIL [AVG pool] dx[" << i << "]=" << dx[i]
                                  << " expected=0.25\n";
                        match = false;
                    }
                }
                ok = match;
            } else {
                std::cerr << "FAIL [AVG pool] backward status=" << r2 << "\n";
            }
        } else {
            std::cerr << "FAIL [AVG pool] forward status=" << r1 << "\n";
        }

        if (ok) { std::cout << "PASS [AVG pool backward]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
        cudnnDestroyPoolingDescriptor(poolDesc);
    }

    // ── Test 3: Null pointer rejection ────────────────────────────────────
    ++total;
    {
        auto r = cudnnPoolingBackward(handle, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, nullptr, nullptr);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
