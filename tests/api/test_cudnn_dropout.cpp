/**
 * Test: cuDNN Dropout Forward/Backward (P3.8)
 *
 * Verifies cudnnDropoutForward and cudnnDropoutBackward:
 * 1. dropout=0 (no dropout): y == x, backward passes dy through
 * 2. dropout=1.0 (all dropout): y == 0, backward produces 0
 * 3. General case: forward stores mask in reserveSpace, backward
 *    uses mask to zero dropped positions and scale survivors.
 * 4. ReserveSpace size query.
 * 5. Null pointer rejection.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnDropoutDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* d);
cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t d);
cudnnStatus_t cudnnSetDropoutDescriptor(
    cudnnDropoutDescriptor_t desc, cudnnHandle_t handle,
    float dropout, void* states, size_t stateSizeInBytes,
    unsigned long long seed);

cudnnStatus_t cudnnGetDropoutReserveSpaceSize(
    cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes);

cudnnStatus_t cudnnDropoutForward(
    cudnnHandle_t handle,
    cudnnDropoutDescriptor_t desc,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);

cudnnStatus_t cudnnDropoutBackward(
    cudnnHandle_t handle,
    cudnnDropoutDescriptor_t desc,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t dxDesc, void* dx,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);
}

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuDNN Dropout Forward/Backward (P3.8) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    int N=1, C=2, H=3, W=3;
    int count = N*C*H*W;

    std::vector<float> x(count);
    for (int i=0; i<count; ++i) x[i] = static_cast<float>(i % 5 - 2);

    cudnnTensorDescriptor_t xDesc, yDesc, dyDesc, dxDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&dxDesc);
    cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);

    // ── Test 1: ReserveSpace size query ─────────────────────────────────
    ++total;
    {
        size_t sz = 0;
        auto r = cudnnGetDropoutReserveSpaceSize(xDesc, &sz);
        if (r == CUDNN_STATUS_SUCCESS && sz == count * sizeof(uint8_t)) {
            std::cout << "PASS [ReserveSpace size]\n"; ++pass;
        } else {
            std::cerr << "FAIL [ReserveSpace size] status=" << r
                      << " sz=" << sz << " expected=" << count*sizeof(uint8_t) << "\n";
        }
    }

    // ── Test 2: dropout=0 (no dropout) ────────────────────────────────────
    ++total;
    {
        cudnnDropoutDescriptor_t dropDesc;
        cudnnCreateDropoutDescriptor(&dropDesc);
        cudnnSetDropoutDescriptor(dropDesc, handle, 0.0f, nullptr, 0, 12345ULL);

        std::vector<float> y(count, 99.f);
        std::vector<uint8_t> reserve(count, 255);
        auto r = cudnnDropoutForward(handle, dropDesc, xDesc, x.data(),
                                     yDesc, y.data(), reserve.data(), reserve.size());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            for (int i=0; i<count && ok; ++i) {
                if (!approx_eq(y[i], x[i], 1e-4f)) {
                    std::cerr << "FAIL [dropout=0] y[" << i << "]=" << y[i]
                              << " expected=" << x[i] << "\n";
                    ok = false;
                }
            }
        }
        if (ok) {
            // backward should pass dy through unchanged
            std::vector<float> dy(count);
            for (int i=0; i<count; ++i) dy[i] = static_cast<float>((i%3)-1);
            std::vector<float> dx(count, 99.f);
            auto r2 = cudnnDropoutBackward(handle, dropDesc, dyDesc, dy.data(),
                                           dxDesc, dx.data(), reserve.data(), reserve.size());
            ok = (r2 == CUDNN_STATUS_SUCCESS);
            if (ok) {
                for (int i=0; i<count && ok; ++i) {
                    if (!approx_eq(dx[i], dy[i], 1e-4f)) {
                        std::cerr << "FAIL [dropout=0 backward] dx[" << i << "]=" << dx[i]
                                  << " expected=" << dy[i] << "\n";
                        ok = false;
                    }
                }
            }
        }
        if (ok) { std::cout << "PASS [dropout=0]\n"; ++pass; }
        else { std::cerr << "FAIL [dropout=0] status=" << r << "\n"; }
        cudnnDestroyDropoutDescriptor(dropDesc);
    }

    // ── Test 3: dropout=1.0 (all dropped) ─────────────────────────────────
    ++total;
    {
        cudnnDropoutDescriptor_t dropDesc;
        cudnnCreateDropoutDescriptor(&dropDesc);
        cudnnSetDropoutDescriptor(dropDesc, handle, 1.0f, nullptr, 0, 12345ULL);

        std::vector<float> y(count, 99.f);
        std::vector<uint8_t> reserve(count, 255);
        auto r = cudnnDropoutForward(handle, dropDesc, xDesc, x.data(),
                                     yDesc, y.data(), reserve.data(), reserve.size());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            for (int i=0; i<count && ok; ++i) {
                if (!approx_eq(y[i], 0.0f, 1e-4f)) {
                    std::cerr << "FAIL [dropout=1.0] y[" << i << "]=" << y[i]
                              << " expected=0\n";
                    ok = false;
                }
            }
        }
        if (ok) {
            std::vector<float> dy(count);
            for (int i=0; i<count; ++i) dy[i] = static_cast<float>((i%3)-1);
            std::vector<float> dx(count, 99.f);
            auto r2 = cudnnDropoutBackward(handle, dropDesc, dyDesc, dy.data(),
                                           dxDesc, dx.data(), reserve.data(), reserve.size());
            ok = (r2 == CUDNN_STATUS_SUCCESS);
            if (ok) {
                for (int i=0; i<count && ok; ++i) {
                    if (!approx_eq(dx[i], 0.0f, 1e-4f)) {
                        std::cerr << "FAIL [dropout=1.0 backward] dx[" << i << "]=" << dx[i]
                                  << " expected=0\n";
                        ok = false;
                    }
                }
            }
        }
        if (ok) { std::cout << "PASS [dropout=1.0]\n"; ++pass; }
        else { std::cerr << "FAIL [dropout=1.0] status=" << r << "\n"; }
        cudnnDestroyDropoutDescriptor(dropDesc);
    }

    // ── Test 4: General dropout consistency (forward mask -> backward) ────
    ++total;
    {
        cudnnDropoutDescriptor_t dropDesc;
        cudnnCreateDropoutDescriptor(&dropDesc);
        cudnnSetDropoutDescriptor(dropDesc, handle, 0.5f, nullptr, 0, 42ULL);

        std::vector<float> y(count, 0.f);
        std::vector<uint8_t> reserve(count, 0);
        auto r = cudnnDropoutForward(handle, dropDesc, xDesc, x.data(),
                                     yDesc, y.data(), reserve.data(), reserve.size());
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            // Verify y matches x * scale * mask
            float scale = 2.0f;  // 1/(1-0.5)
            for (int i=0; i<count && ok; ++i) {
                float expected = reserve[i] ? (x[i] * scale) : 0.0f;
                if (!approx_eq(y[i], expected, 1e-4f)) {
                    std::cerr << "FAIL [dropout=0.5 forward] y[" << i << "]=" << y[i]
                              << " expected=" << expected << " mask=" << (int)reserve[i] << "\n";
                    ok = false;
                }
            }
        }
        if (ok) {
            std::vector<float> dy(count);
            for (int i=0; i<count; ++i) dy[i] = static_cast<float>((i%4)-1);
            std::vector<float> dx(count, 0.f);
            auto r2 = cudnnDropoutBackward(handle, dropDesc, dyDesc, dy.data(),
                                           dxDesc, dx.data(), reserve.data(), reserve.size());
            ok = (r2 == CUDNN_STATUS_SUCCESS);
            if (ok) {
                float scale = 2.0f;
                for (int i=0; i<count && ok; ++i) {
                    float expected = reserve[i] ? (dy[i] * scale) : 0.0f;
                    if (!approx_eq(dx[i], expected, 1e-4f)) {
                        std::cerr << "FAIL [dropout=0.5 backward] dx[" << i << "]=" << dx[i]
                                  << " expected=" << expected << " mask=" << (int)reserve[i] << "\n";
                        ok = false;
                    }
                }
            }
        }
        if (ok) { std::cout << "PASS [dropout=0.5 consistency]\n"; ++pass; }
        else { std::cerr << "FAIL [dropout=0.5] status=" << r << "\n"; }
        cudnnDestroyDropoutDescriptor(dropDesc);
    }

    // ── Test 5: Null pointer rejection ──────────────────────────────────
    ++total;
    {
        auto r = cudnnDropoutForward(handle, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, 0);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyTensorDescriptor(dxDesc);
    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
