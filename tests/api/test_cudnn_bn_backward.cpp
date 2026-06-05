/**
 * Test: cuDNN Batch Normalization Backward (P3.4)
 *
 * Verifies cudnnBatchNormalizationBackward computes dx, dScale, dBias
 * correctly by comparing with a reference implementation.
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

cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t,
    cudnnBatchNormMode_t mode,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t dxDesc, void* dx,
    cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void* bnScale,
    void* dBnScaleResult, void* dBnBiasResult,
    double epsilon,
    const void* savedMean, const void* savedInvVariance);
}

static bool approx_eq(float a, float b, float eps = 1e-3f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuDNN BN Backward (P3.4) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    // ── Test 1: Spatial mode end-to-end (forward then backward) ────────────
    ++total;
    {
        int N=2, C=3, H=2, W=2;
        int HW = H*W;

        std::vector<float> x(N*C*HW);
        std::vector<float> y(N*C*HW, 0.f);
        std::vector<float> scale(C), bias(C);
        std::vector<float> saveMean(C), saveInvVar(C);
        std::vector<float> runningMean(C, 0.f), runningVar(C, 1.f);

        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i % 7 - 3);
        for (int c = 0; c < C; ++c) { scale[c] = 1.0f + 0.1f*c; bias[c] = 0.2f*c; }

        cudnnTensorDescriptor_t xDesc, yDesc, bnDesc;
        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&yDesc);
        cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&bnDesc);
        cudnnSetTensor4dDescriptor(bnDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,C,1,1);

        float alpha = 1.0f, beta = 0.0f;
        auto r1 = cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            xDesc, x.data(),
            yDesc, y.data(),
            bnDesc,
            scale.data(), bias.data(),
            0.1,
            runningMean.data(), runningVar.data(),
            1e-5,
            saveMean.data(), saveInvVar.data());

        bool ok = (r1 == CUDNN_STATUS_SUCCESS);

        if (ok) {
            std::vector<float> dy(N*C*HW);
            std::vector<float> dx(N*C*HW, 0.f);
            std::vector<float> dScale(C, 0.f), dBias(C, 0.f);

            for (size_t i = 0; i < dy.size(); ++i) dy[i] = static_cast<float>((i % 5) - 2);

            auto r2 = cudnnBatchNormalizationBackward(
                handle, CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta, &alpha, &beta,
                xDesc, x.data(),
                yDesc, dy.data(),
                xDesc, dx.data(),
                bnDesc,
                scale.data(),
                dScale.data(), dBias.data(),
                1e-5,
                saveMean.data(), saveInvVar.data());

            ok = (r2 == CUDNN_STATUS_SUCCESS);

            if (ok) {
                // Reference computation
                std::vector<float> ref_dx(N*C*HW, 0.f);
                std::vector<float> ref_dScale(C, 0.f), ref_dBias(C, 0.f);

                for (int c=0; c<C; ++c) {
                    double sum_dy = 0, sum_dy_xhat = 0;
                    for (int n=0; n<N; ++n)
                    for (int hw=0; hw<HW; ++hw) {
                        int idx = (n*C + c)*HW + hw;
                        float xhat = (x[idx] - saveMean[c]) * saveInvVar[c];
                        sum_dy += dy[idx];
                        sum_dy_xhat += dy[idx] * xhat;
                    }
                    double mean_dy = sum_dy / (N*HW);
                    double mean_dy_xhat = sum_dy_xhat / (N*HW);

                    for (int n=0; n<N; ++n)
                    for (int hw=0; hw<HW; ++hw) {
                        int idx = (n*C + c)*HW + hw;
                        float xhat = (x[idx] - saveMean[c]) * saveInvVar[c];
                        float val = dy[idx] - static_cast<float>(mean_dy) - xhat * static_cast<float>(mean_dy_xhat);
                        ref_dx[idx] = scale[c] * saveInvVar[c] * val;
                    }

                    double dscale = 0, dbias = 0;
                    for (int n=0; n<N; ++n)
                    for (int hw=0; hw<HW; ++hw) {
                        int idx = (n*C + c)*HW + hw;
                        float xhat = (x[idx] - saveMean[c]) * saveInvVar[c];
                        dscale += dy[idx] * xhat;
                        dbias  += dy[idx];
                    }
                    ref_dScale[c] = static_cast<float>(dscale);
                    ref_dBias[c]  = static_cast<float>(dbias);
                }

                for (size_t i = 0; i < dx.size() && ok; ++i) {
                    if (!approx_eq(dx[i], ref_dx[i], 1e-3f)) {
                        std::cerr << "FAIL [Spatial] dx[" << i << "]=" << dx[i]
                                  << " expected=" << ref_dx[i] << "\n";
                        ok = false; break;
                    }
                }
                for (int c = 0; c < C && ok; ++c) {
                    if (!approx_eq(dScale[c], ref_dScale[c], 1e-3f)) {
                        std::cerr << "FAIL [Spatial] dScale[" << c << "]=" << dScale[c]
                                  << " expected=" << ref_dScale[c] << "\n";
                        ok = false; break;
                    }
                    if (!approx_eq(dBias[c], ref_dBias[c], 1e-3f)) {
                        std::cerr << "FAIL [Spatial] dBias[" << c << "]=" << dBias[c]
                                  << " expected=" << ref_dBias[c] << "\n";
                        ok = false; break;
                    }
                }
            } else {
                std::cerr << "FAIL [Spatial] backward status=" << r2 << "\n";
            }
        } else {
            std::cerr << "FAIL [Spatial] forward status=" << r1 << "\n";
        }

        if (ok) { std::cout << "PASS [Spatial end-to-end]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc);
        cudnnDestroyTensorDescriptor(yDesc);
        cudnnDestroyTensorDescriptor(bnDesc);
    }

    // ── Test 2: Null pointer rejection ──────────────────────────────────
    ++total;
    {
        auto r = cudnnBatchNormalizationBackward(
            handle, CUDNN_BATCHNORM_SPATIAL,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr,
            nullptr, nullptr,
            nullptr, nullptr,
            nullptr,
            nullptr, nullptr, nullptr,
            1e-5, nullptr, nullptr);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    // ── Test 3: Numerical gradient check (central difference) ────────────
    // Loss = dot(w, y) with non-uniform w; NOT BN-invariant (sum(w*xhat) ≠ const).
    // h=1e-2 is the float32-optimal step (h=1e-4 is too small: signal ~3e-4 ≈ float noise).
    // Analytical dx uses fixed dy=w through the BN backward formula.
    // Math: dx = scale/std*(dy - mean(dy) - xhat*mean(dy*xhat)); m=N*H*W
    ++total;
    {
        const float h = 1e-2f;  // float32 optimal: noise ≈ 8e-5, approx_err ≈ 1e-4
        const int N=2, C=1, H=2, W=2;  // C=1 to isolate channel maths
        const int HW = H*W, SZ = N*C*HW;

        std::vector<float> x(SZ), sc(C), bi(C);
        for (int i=0; i<SZ; ++i) x[i] = static_cast<float>((i%7) - 3) * 0.5f;
        sc[0] = 1.5f; bi[0] = 0.3f;

        // Non-uniform dy weights (alternating sign + magnitude variation)
        std::vector<float> dy(SZ);
        for (int i=0; i<SZ; ++i) dy[i] = static_cast<float>(i%3) - 1.0f + 0.1f*i;

        cudnnTensorDescriptor_t xDesc3, bnDesc3;
        cudnnCreateTensorDescriptor(&xDesc3);
        cudnnSetTensor4dDescriptor(xDesc3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
        cudnnCreateTensorDescriptor(&bnDesc3);
        cudnnSetTensor4dDescriptor(bnDesc3, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,C,1,1);

        // Forward pass to get saved mean/inv-std
        std::vector<float> y0(SZ), sm0(C), siv0(C);
        {
            std::vector<float> rm(C, 0.f), rv(C, 1.f);
            float a=1.f, b=0.f;
            cudnnBatchNormalizationForwardTraining(
                handle, CUDNN_BATCHNORM_SPATIAL,
                &a, &b,
                xDesc3, x.data(),
                xDesc3, y0.data(),
                bnDesc3, sc.data(), bi.data(),
                0.1, rm.data(), rv.data(),
                1e-5,
                sm0.data(), siv0.data());
        }

        // Analytical backward with fixed dy weights
        std::vector<float> dx_an(SZ, 0.f), dSc(C, 0.f), dBi(C, 0.f);
        {
            float a=1.f, b=0.f;
            cudnnBatchNormalizationBackward(
                handle, CUDNN_BATCHNORM_SPATIAL,
                &a, &b, &a, &b,
                xDesc3, x.data(),
                xDesc3, dy.data(),
                xDesc3, dx_an.data(),
                bnDesc3, sc.data(),
                dSc.data(), dBi.data(),
                1e-5, sm0.data(), siv0.data());
        }

        // Numerical gradient: L = dot(dy, y(x))  →  dL/dx[i] via central diff
        bool ok = true;
        for (int i=0; i<SZ && ok; ++i) {
            std::vector<float> xp = x, xm = x;
            xp[i] += h; xm[i] -= h;

            std::vector<float> yp(SZ,0.f), ym(SZ,0.f), smp(C), sivp(C), smm(C), sivm(C);
            std::vector<float> rm1(C,0.f), rv1(C,1.f), rm2(C,0.f), rv2(C,1.f);
            float a=1.f, b=0.f;
            cudnnBatchNormalizationForwardTraining(
                handle, CUDNN_BATCHNORM_SPATIAL,
                &a, &b, xDesc3, xp.data(), xDesc3, yp.data(),
                bnDesc3, sc.data(), bi.data(), 0.1, rm1.data(), rv1.data(),
                1e-5, smp.data(), sivp.data());
            cudnnBatchNormalizationForwardTraining(
                handle, CUDNN_BATCHNORM_SPATIAL,
                &a, &b, xDesc3, xm.data(), xDesc3, ym.data(),
                bnDesc3, sc.data(), bi.data(), 0.1, rm2.data(), rv2.data(),
                1e-5, smm.data(), sivm.data());

            float Lp = 0.f, Lm = 0.f;
            for (int j=0; j<SZ; ++j) { Lp += dy[j]*yp[j]; Lm += dy[j]*ym[j]; }
            float dx_num_i = (Lp - Lm) / (2.f * h);

            float err = std::abs(dx_an[i] - dx_num_i);
            if (err > 1e-3f) {
                std::cerr << "FAIL [NumGrad] i=" << i
                          << " dx_an=" << dx_an[i]
                          << " dx_num=" << dx_num_i
                          << " err=" << err << "\n";
                ok = false;
            }
        }

        if (ok) { std::cout << "PASS [Numerical gradient dX, tol 1e-3]\n"; ++pass; }

        cudnnDestroyTensorDescriptor(xDesc3);
        cudnnDestroyTensorDescriptor(bnDesc3);
    }

    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
