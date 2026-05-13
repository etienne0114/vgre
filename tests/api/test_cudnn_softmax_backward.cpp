/**
 * Test: cuDNN Softmax Backward (P3.6)
 *
 * Verifies cudnnSoftmaxBackward computes dx correctly
 * by comparing with reference formula:
 *   dx_i = y_i * (dy_i - sum_j(dy_j * y_j))
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t,
    int algo, int mode,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y);

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t,
    int algo, int mode,
    const void* alpha,
    cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx);
}

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuDNN Softmax Backward (P3.6) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    // ── Test 1: CHANNEL mode ────────────────────────────────────────────
    ++total;
    {
        int N=2, C=3, H=2, W=2;
        int HW = H*W;

        std::vector<float> x(N*C*HW);
        std::vector<float> y(N*C*HW, 0.f);
        std::vector<float> dy(N*C*HW);
        std::vector<float> dx(N*C*HW, 0.f);

        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i % 5) - 2);
        for (size_t i = 0; i < dy.size(); ++i) dy[i] = static_cast<float>((i % 3) - 1);

        cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);

        float alpha = 1.0f, beta = 0.0f;
        auto r1 = cudnnSoftmaxForward(handle, 0, 1, &alpha, xDesc, x.data(), &beta, yDesc, y.data());

        bool ok = (r1 == CUDNN_STATUS_SUCCESS);
        if (ok) {
            auto r2 = cudnnSoftmaxBackward(handle, 0, 1, &alpha,
                                              yDesc, y.data(), dyDesc, dy.data(),
                                              &beta, dxDesc, dx.data());
            ok = (r2 == CUDNN_STATUS_SUCCESS);

            if (ok) {
                // Reference: dx_i = y_i * (dy_i - sum_j(dy_j * y_j))
                for (int n=0; n<N && ok; ++n)
                for (int hw=0; hw<HW && ok; ++hw) {
                    float sum_dy_y = 0.f;
                    for (int c=0; c<C; ++c) {
                        int idx = n*C*HW + c*HW + hw;
                        sum_dy_y += dy[idx] * y[idx];
                    }
                    for (int c=0; c<C; ++c) {
                        int idx = n*C*HW + c*HW + hw;
                        float ref = y[idx] * (dy[idx] - sum_dy_y);
                        if (!approx_eq(dx[idx], ref, 1e-3f)) {
                            std::cerr << "FAIL [CHANNEL] dx[" << idx << "]=" << dx[idx]
                                      << " expected=" << ref << "\n";
                            ok = false; break;
                        }
                    }
                }
            } else {
                std::cerr << "FAIL [CHANNEL] backward status=" << r2 << "\n";
            }
        } else {
            std::cerr << "FAIL [CHANNEL] forward status=" << r1 << "\n";
        }

        if (ok) { std::cout << "PASS [CHANNEL mode]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
    }

    // ── Test 2: INSTANCE mode ───────────────────────────────────────────
    ++total;
    {
        int N=2, C=2, H=2, W=2;
        int HW = H*W;

        std::vector<float> x(N*C*HW);
        std::vector<float> y(N*C*HW, 0.f);
        std::vector<float> dy(N*C*HW);
        std::vector<float> dx(N*C*HW, 0.f);

        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i % 5) - 2);
        for (size_t i = 0; i < dy.size(); ++i) dy[i] = static_cast<float>((i % 3) - 1);

        cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&dxDesc);
        cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);

        float alpha = 1.0f, beta = 0.0f;
        auto r1 = cudnnSoftmaxForward(handle, 0, 0, &alpha, xDesc, x.data(), &beta, yDesc, y.data());

        bool ok = (r1 == CUDNN_STATUS_SUCCESS);
        if (ok) {
            auto r2 = cudnnSoftmaxBackward(handle, 0, 0, &alpha,
                                              yDesc, y.data(), dyDesc, dy.data(),
                                              &beta, dxDesc, dx.data());
            ok = (r2 == CUDNN_STATUS_SUCCESS);

            if (ok) {
                int VEC = C*HW;
                for (int n=0; n<N && ok; ++n) {
                    float sum_dy_y = 0.f;
                    for (int i=0; i<VEC; ++i)
                        sum_dy_y += dy[n*VEC+i] * y[n*VEC+i];
                    for (int i=0; i<VEC; ++i) {
                        float ref = y[n*VEC+i] * (dy[n*VEC+i] - sum_dy_y);
                        if (!approx_eq(dx[n*VEC+i], ref, 1e-3f)) {
                            std::cerr << "FAIL [INSTANCE] dx[" << n*VEC+i << "]=" << dx[n*VEC+i]
                                      << " expected=" << ref << "\n";
                            ok = false; break;
                        }
                    }
                }
            } else {
                std::cerr << "FAIL [INSTANCE] backward status=" << r2 << "\n";
            }
        } else {
            std::cerr << "FAIL [INSTANCE] forward status=" << r1 << "\n";
        }

        if (ok) { std::cout << "PASS [INSTANCE mode]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(dyDesc);
        cudnnDestroyTensorDescriptor(dxDesc);
    }

    // ── Test 3: Null pointer rejection ──────────────────────────────────
    ++total;
    {
        auto r = cudnnSoftmaxBackward(handle, 0, 1, nullptr,
                                       nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, nullptr);
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
