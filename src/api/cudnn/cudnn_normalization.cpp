// cuDNN 8.5+ Normalization API
//
// Implements cudnnNormalizationForwardTraining, ForwardInference, and Backward.
// Two modes:
//   CUDNN_NORM_PER_CHANNEL    → per-channel batch normalization (delegates to BN math)
//   CUDNN_NORM_PER_ACTIVATION → per-element layer normalization over all non-batch dims

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"

#include <cmath>
#include <vector>

extern "C" {

cudnnStatus_t cudnnNormalizationForwardInference(
    cudnnHandle_t /*handle*/,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t /*algo*/,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    double epsilon,
    const cudnnTensorDescriptor_t /*zDesc*/, const void * /*z*/,
    const cudnnActivationDescriptor_t actDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    void * /*workspace*/, size_t /*workspaceSizeInBytes*/)
{
    if (!xDesc || !yDesc || !x || !y || !normScale || !normBias)
        return CUDNN_STATUS_INVALID_VALUE;
    auto *t = reinterpret_cast<TensorDesc*>(xDesc);
    int N = t->n, C = t->c, HW = t->h * t->w;
    float a = *(const float*)alpha, b_scale = *(const float*)beta;
    const float *xf = static_cast<const float*>(x);
    float *yf = static_cast<float*>(y);
    const float *sc = static_cast<const float*>(normScale);
    const float *bi = static_cast<const float*>(normBias);
    const float *mu = static_cast<const float*>(estimatedMean);
    const float *va = static_cast<const float*>(estimatedVariance);

    // Decode the activation descriptor (null → identity / no activation)
    const ActDesc *act = static_cast<const ActDesc*>(actDesc);
    const bool fuse_act = (normOps == CUDNN_NORM_OPS_NORM_ACTIVATION) && (act != nullptr);

    if (mode == CUDNN_NORM_PER_CHANNEL) {
        // Fused BN+ReLU: single pass O(N·C·H·W), no intermediate buffer
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
        for (int hw = 0; hw < HW; ++hw) {
            int idx = (n * C + c) * HW + hw;
            float xhat = (xf[idx] - mu[c]) / std::sqrt(va[c] + static_cast<float>(epsilon));
            float yval = sc[c] * xhat + bi[c];
            // ReLU fusion: single pass, no intermediate buffer
            if (fuse_act) yval = applyActivation(yval, *act);
            yf[idx] = a * yval + b_scale * yf[idx];
        }
    } else {
        // Layer norm inference: normalize over all C*HW elements per sample
        int D = C * HW;
        #ifdef _OPENMP
        #pragma omp parallel for if (N > 1)
        #endif
        for (int n = 0; n < N; ++n) {
            const float *xn = xf + n * D;
            float *yn = yf + n * D;
            // Compute mean and variance over D elements
            double sum = 0.0, sum2 = 0.0;
            for (int i = 0; i < D; ++i) { sum += xn[i]; sum2 += static_cast<double>(xn[i]) * xn[i]; }
            float mean_n = static_cast<float>(sum / D);
            float var_n  = static_cast<float>(sum2 / D - static_cast<double>(mean_n) * mean_n);
            float inv_std = 1.0f / std::sqrt(var_n + static_cast<float>(epsilon));
            for (int i = 0; i < D; ++i) {
                float xhat = (xn[i] - mean_n) * inv_std;
                float yval = sc[i] * xhat + bi[i];
                // Fused BN+ReLU: single pass O(N·C·H·W), no intermediate buffer
                if (fuse_act) yval = applyActivation(yval, *act);
                yn[i] = a * yval + b_scale * yn[i];
            }
        }
    }
    (void)normScaleBiasDesc; (void)normMeanVarDesc; (void)yDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationForwardTraining(
    cudnnHandle_t /*handle*/,
    cudnnNormMode_t mode,
    cudnnNormOps_t normOps,
    cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    const cudnnActivationDescriptor_t /*actDesc*/,
    const cudnnTensorDescriptor_t /*zDesc*/, const void * /*z*/,
    const cudnnTensorDescriptor_t yDesc, void *y,
    void * /*workspace*/, size_t /*workspaceSizeInBytes*/,
    void * /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    if (!xDesc || !yDesc || !x || !y || !normScale || !normBias)
        return CUDNN_STATUS_INVALID_VALUE;
    auto *t = reinterpret_cast<TensorDesc*>(xDesc);
    int N = t->n, C = t->c, HW = t->h * t->w;
    float a = *(const float*)alpha, b_scale = *(const float*)beta;
    const float *xf = static_cast<const float*>(x);
    float *yf = static_cast<float*>(y);
    const float *sc = static_cast<const float*>(normScale);
    const float *bi = static_cast<const float*>(normBias);

    if (mode == CUDNN_NORM_PER_CHANNEL) {
        // Per-channel batch norm training: statistics over N*HW per channel
        // Uses Welford online algorithm for numerically stable mean/variance
        int group = N * HW;
        std::vector<float> mean(C, 0.f), var(C, 0.f);
        for (int c = 0; c < C; ++c) {
            // Welford online algorithm for numerically stable mean/variance
            // Invariant: M2_n = M2_{n-1} + (x-mean_{n-1})(x-mean_n)
            double welford_mean = 0.0, M2 = 0.0;
            int count = 0;
            for (int hw = 0; hw < HW; ++hw) {
                for (int n = 0; n < N; ++n) {
                    float xval = xf[(n * C + c) * HW + hw];
                    ++count;
                    double delta = xval - welford_mean;
                    welford_mean += delta / count;
                    double delta2 = xval - welford_mean;
                    // Welford invariant: M2_n = M2_{n-1} + (x-mean_{n-1})(x-mean_n)
                    M2 += delta * delta2;
                }
            }
            mean[c] = static_cast<float>(welford_mean);
            // Population variance (consistent with two-pass BN training)
            var[c]  = (count > 1) ? static_cast<float>(M2 / count) : 0.f;
        }
        float ema = static_cast<float>(exponentialAverageFactor);
        float ema1 = 1.0f - ema;
        if (resultRunningMean) {
            float *rm = static_cast<float*>(resultRunningMean);
            for (int c = 0; c < C; ++c) rm[c] = ema * mean[c] + ema1 * rm[c];
        }
        if (resultRunningVariance) {
            float *rv = static_cast<float*>(resultRunningVariance);
            float corr = (group > 1) ? static_cast<float>(group) / (group - 1) : 1.0f;
            for (int c = 0; c < C; ++c) rv[c] = ema * var[c] * corr + ema1 * rv[c];
        }
        if (resultSaveMean)       memcpy(resultSaveMean,       mean.data(), C * sizeof(float));
        if (resultSaveInvVariance) {
            float *siv = static_cast<float*>(resultSaveInvVariance);
            for (int c = 0; c < C; ++c) siv[c] = 1.0f / std::sqrt(var[c] + static_cast<float>(epsilon));
        }
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
        for (int hw = 0; hw < HW; ++hw) {
            int idx = (n * C + c) * HW + hw;
            float inv_std = 1.0f / std::sqrt(var[c] + static_cast<float>(epsilon));
            float xhat = (xf[idx] - mean[c]) * inv_std;
            yf[idx] = a * (sc[c] * xhat + bi[c]) + b_scale * yf[idx];
        }
    } else {
        // Per-activation (layer norm): normalize per sample over C*HW
        int D = C * HW;
        std::vector<float> save_mean(N, 0.f), save_inv_var(N, 0.f);
        #ifdef _OPENMP
        #pragma omp parallel for if (N > 1)
        #endif
        for (int n = 0; n < N; ++n) {
            const float *xn = xf + n * D;
            float *yn = yf + n * D;
            double s = 0.0, s2 = 0.0;
            for (int i = 0; i < D; ++i) { s += xn[i]; s2 += static_cast<double>(xn[i]) * xn[i]; }
            float mean_n = static_cast<float>(s / D);
            float var_n  = static_cast<float>(s2 / D - static_cast<double>(mean_n) * mean_n);
            float inv_std = 1.0f / std::sqrt(var_n + static_cast<float>(epsilon));
            save_mean[n] = mean_n;
            save_inv_var[n] = inv_std;
            for (int i = 0; i < D; ++i) {
                float xhat = (xn[i] - mean_n) * inv_std;
                yn[i] = a * (sc[i] * xhat + bi[i]) + b_scale * yn[i];
            }
        }
        if (resultSaveMean)        memcpy(resultSaveMean,        save_mean.data(),    N * sizeof(float));
        if (resultSaveInvVariance) memcpy(resultSaveInvVariance, save_inv_var.data(), N * sizeof(float));
    }
    (void)normScaleBiasDesc; (void)normMeanVarDesc; (void)yDesc;
    (void)normOps; (void)algo;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnNormalizationBackward(
    cudnnHandle_t /*handle*/,
    cudnnNormMode_t mode,
    cudnnNormOps_t /*normOps*/,
    cudnnNormAlgo_t /*algo*/,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t /*yDesc*/, const void * /*y*/,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t /*dzDesc*/, void * /*dz*/,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    void *resultBnScaleDiff, void *resultBnBiasDiff,
    double epsilon,
    const cudnnTensorDescriptor_t /*savedMeanVarDesc*/,
    const void *savedMean, const void *savedInvVariance,
    const cudnnActivationDescriptor_t /*actDesc*/,
    void * /*workspace*/, size_t /*workspaceSizeInBytes*/,
    void * /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    if (!xDesc || !dyDesc || !dxDesc || !x || !dy || !dx || !normScale)
        return CUDNN_STATUS_INVALID_VALUE;
    auto *t = reinterpret_cast<TensorDesc*>(xDesc);
    int N = t->n, C = t->c, HW = t->h * t->w;
    float adD = *(const float*)alphaDataDiff,  bdD = *(const float*)betaDataDiff;
    float adP = *(const float*)alphaParamDiff, bdP = *(const float*)betaParamDiff;
    const float *xf  = static_cast<const float*>(x);
    const float *dyf = static_cast<const float*>(dy);
    float *dxf = static_cast<float*>(dx);
    const float *sc = static_cast<const float*>(normScale);

    if (mode == CUDNN_NORM_PER_CHANNEL) {
        // Per-channel backward: dscale, dbias, dx — same as BN backward
        const float *mu_s  = static_cast<const float*>(savedMean);
        const float *ivar_s = static_cast<const float*>(savedInvVariance);
        if (!mu_s || !ivar_s) return CUDNN_STATUS_INVALID_VALUE;
        int group = N * HW;
        std::vector<float> dsc(C, 0.f), dbi(C, 0.f);
        for (int c = 0; c < C; ++c) {
            for (int n = 0; n < N; ++n) for (int hw = 0; hw < HW; ++hw) {
                int idx = (n * C + c) * HW + hw;
                float xhat = (xf[idx] - mu_s[c]) * ivar_s[c];
                dsc[c] += dyf[idx] * xhat;
                dbi[c] += dyf[idx];
            }
        }
        if (resultBnScaleDiff) {
            float *ds = static_cast<float*>(resultBnScaleDiff);
            for (int c = 0; c < C; ++c) ds[c] = adP * dsc[c] + bdP * ds[c];
        }
        if (resultBnBiasDiff) {
            float *db = static_cast<float*>(resultBnBiasDiff);
            for (int c = 0; c < C; ++c) db[c] = adP * dbi[c] + bdP * db[c];
        }
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            double sum_dy = 0.0, sum_dy_xhat = 0.0;
            for (int hw = 0; hw < HW; ++hw) {
                int idx = (n * C + c) * HW + hw;
                float xhat = (xf[idx] - mu_s[c]) * ivar_s[c];
                sum_dy      += dyf[idx];
                sum_dy_xhat += dyf[idx] * xhat;
            }
            for (int hw = 0; hw < HW; ++hw) {
                int idx = (n * C + c) * HW + hw;
                float xhat = (xf[idx] - mu_s[c]) * ivar_s[c];
                float dx_val = sc[c] * ivar_s[c] / group *
                    (group * dyf[idx] - static_cast<float>(sum_dy) - xhat * static_cast<float>(sum_dy_xhat));
                dxf[idx] = adD * dx_val + bdD * dxf[idx];
            }
        }
    } else {
        // Per-activation (layer norm) backward
        const float *inv_std_s = static_cast<const float*>(savedInvVariance);
        const float *mean_s    = static_cast<const float*>(savedMean);
        if (!inv_std_s || !mean_s) return CUDNN_STATUS_INVALID_VALUE;
        int D = C * HW;
        std::vector<float> dsc(D, 0.f), dbi_v(D, 0.f);
        for (int n = 0; n < N; ++n) {
            const float *xn  = xf  + n * D;
            const float *dyn = dyf + n * D;
            float inv_std = inv_std_s[n];
            float mean_n  = mean_s[n];
            for (int i = 0; i < D; ++i) {
                float xhat = (xn[i] - mean_n) * inv_std;
                dsc[i]  += dyn[i] * xhat;
                dbi_v[i] += dyn[i];
            }
        }
        if (resultBnScaleDiff) {
            float *ds = static_cast<float*>(resultBnScaleDiff);
            for (int i = 0; i < D; ++i) ds[i] = adP * dsc[i] + bdP * ds[i];
        }
        if (resultBnBiasDiff) {
            float *db = static_cast<float*>(resultBnBiasDiff);
            for (int i = 0; i < D; ++i) db[i] = adP * dbi_v[i] + bdP * db[i];
        }
        #ifdef _OPENMP
        #pragma omp parallel for if (N > 1)
        #endif
        for (int n = 0; n < N; ++n) {
            const float *xn  = xf  + n * D;
            const float *dyn = dyf + n * D;
            float *dxn = dxf + n * D;
            float inv_std = inv_std_s[n];
            float mean_n  = mean_s[n];
            double sum_dy = 0.0, sum_dy_xhat = 0.0;
            for (int i = 0; i < D; ++i) {
                float xhat = (xn[i] - mean_n) * inv_std;
                sum_dy      += dyn[i];
                sum_dy_xhat += dyn[i] * xhat;
            }
            for (int i = 0; i < D; ++i) {
                float xhat = (xn[i] - mean_n) * inv_std;
                float dx_val = sc[i] * inv_std / D *
                    (D * dyn[i] - static_cast<float>(sum_dy) - xhat * static_cast<float>(sum_dy_xhat));
                dxn[i] = adD * dx_val + bdD * dxn[i];
            }
        }
    }
    (void)normScaleBiasDesc; (void)dxDesc; (void)normBias;
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
