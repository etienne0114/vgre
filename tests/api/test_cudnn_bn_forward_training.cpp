/**
 * Test: cuDNN Batch Normalization Forward Training (P3.3)
 *
 * Verifies cudnnBatchNormalizationForwardTraining computes
 * per-batch statistics, normalizes, applies scale/shift,
 * and optionally saves mean/invVariance and updates running stats.
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

typedef void* cudnnTensorDescriptor_t; // bnDesc alias

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL        = 1
};

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance);
}

static bool approx_eq(float a, float b, float eps = 1e-3f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

static bool runSpatialTest(cudnnHandle_t handle, const char* name, int& pass, int& total) {
    ++total;
    int N=2, C=3, H=2, W=2;
    int HW = H*W;

    std::vector<float> x(N*C*HW);
    std::vector<float> y(N*C*HW, 0.f);
    std::vector<float> scale(C), bias(C);
    std::vector<float> saveMean(C), saveInvVar(C);
    std::vector<float> runningMean(C, 1.f), runningVar(C, 1.f);

    // Simple data
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i);
    for (int c = 0; c < C; ++c) { scale[c] = 1.0f; bias[c] = 0.0f; }

    cudnnTensorDescriptor_t xDesc, yDesc, bnDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&bnDesc);
    cudnnSetTensor4dDescriptor(bnDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,C,1,1);

    float alpha = 1.0f, beta = 0.0f;
    auto r = cudnnBatchNormalizationForwardTraining(
        handle, CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        xDesc, x.data(),
        yDesc, y.data(),
        bnDesc,
        scale.data(), bias.data(),
        0.1, // exponentialAverageFactor
        runningMean.data(), runningVar.data(),
        1e-5,
        saveMean.data(), saveInvVar.data());

    bool ok = (r == CUDNN_STATUS_SUCCESS);
    if (ok) {
        // Reference: compute mean/var manually
        for (int c = 0; c < C && ok; ++c) {
            double sum = 0, sumSq = 0;
            for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HW; ++hw) {
                float v = x[(n*C + c)*HW + hw];
                sum += v;
            }
            float mean = static_cast<float>(sum / (N*HW));
            for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HW; ++hw) {
                float d = x[(n*C + c)*HW + hw] - mean;
                sumSq += d*d;
            }
            float var = static_cast<float>(sumSq / (N*HW));
            float invVar = 1.f / std::sqrt(var + 1e-5f);

            // Check saved mean/invVar
            if (!approx_eq(saveMean[c], mean, 1e-3f)) {
                std::cerr << "FAIL [" << name << "] saveMean[" << c << "]=" << saveMean[c]
                          << " expected=" << mean << "\n";
                ok = false; break;
            }
            if (!approx_eq(saveInvVar[c], invVar, 1e-3f)) {
                std::cerr << "FAIL [" << name << "] saveInvVar[" << c << "]=" << saveInvVar[c]
                          << " expected=" << invVar << "\n";
                ok = false; break;
            }

            // Check output
            for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                float xhat = (x[idx] - mean) * invVar;
                if (!approx_eq(y[idx], xhat, 1e-3f)) {
                    std::cerr << "FAIL [" << name << "] y[" << idx << "]=" << y[idx]
                              << " expected=" << xhat << "\n";
                    ok = false; break;
                }
            }
        }
    } else {
        std::cerr << "FAIL [" << name << "] status=" << r << "\n";
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(bnDesc);

    if (ok) { std::cout << "PASS [" << name << "]\n"; ++pass; }
    return ok;
}

int main() {
    std::cout << "=== Test: cuDNN BN Forward Training (P3.3) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    // ── Test 1: Spatial mode ──────────────────────────────────────────────
    runSpatialTest(handle, "Spatial NCHW", pass, total);

    // ── Test 2: Null pointer rejection ──────────────────────────────────
    ++total;
    {
        auto r = cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL,
            nullptr, nullptr,
            nullptr, nullptr,
            nullptr, nullptr,
            nullptr,
            nullptr, nullptr,
            0.1, nullptr, nullptr,
            1e-5, nullptr, nullptr);
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
