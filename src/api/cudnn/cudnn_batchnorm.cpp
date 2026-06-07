// cuDNN batch normalization: inference, training, backward

#include "cudnn_internal.h"

#include "vgre/common/openmp_helper.h"

extern "C" {

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t, int /*mode*/,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnDesc*/,
    const void* scale, const void* bias,
    const void* mean, const void* var, double eps)
{
    auto* t=(TensorDesc*)xDesc;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;
    const float* sc=(const float*)scale; const float* bi=(const float*)bias;
    const float* mu=(const float*)mean; const float* va=(const float*)var;
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
    #endif
    for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int hw=0; hw<HW; ++hw) {
        int idx = (n*C + c)*HW + hw;
        float xhat = (xf[idx] - mu[c]) / sqrtf((float)va[c] + (float)eps);
        yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
    }
    (void)yDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnScaleBiasMeanVarDesc*/,
    const void* scale, const void* bias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta || !scale || !bias)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* t=(TensorDesc*)xDesc;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;
    const float* sc=(const float*)scale; const float* bi=(const float*)bias;

    std::vector<float> mean(C, 0.f), var(C, 0.f);
    int count = N * HW;

    if (mode == CUDNN_BATCHNORM_SPATIAL) {
        // Welford online algorithm: single pass, numerically stable mean/variance
        #ifdef _OPENMP
        #pragma omp parallel for if (C > 4)
        #endif
        for (int c=0; c<C; ++c) {
            // Welford online algorithm for numerically stable mean/variance
            double welford_mean = 0.0, M2 = 0.0;
            int wcount = 0;
            for (int hw=0; hw<HW; ++hw) {
                for (int n=0; n<N; ++n) {
                    float xval = xf[(n*C + c)*HW + hw];
                    ++wcount;
                    double delta = xval - welford_mean;
                    welford_mean += delta / wcount;
                    double delta2 = xval - welford_mean;
                    // Welford invariant: M2_n = M2_{n-1} + (x-mean_{n-1})(x-mean_n)
                    M2 += delta * delta2;
                }
            }
            mean[c] = static_cast<float>(welford_mean);
            // Population variance (count == N*HW, consistent with original formulation)
            var[c]  = (wcount > 1) ? static_cast<float>(M2 / wcount) : 0.f;
        }
    } else {
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            double sum = 0.0;
            for (int n=0; n<N; ++n)
                sum += xf[(n*C + c)*HW + hw];
            mean[c*HW + hw] = static_cast<float>(sum / N);
        }
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            double sumSq = 0.0;
            for (int n=0; n<N; ++n) {
                float d = xf[(n*C + c)*HW + hw] - mean[c*HW + hw];
                sumSq += d * d;
            }
            var[c*HW + hw] = static_cast<float>(sumSq / N);
        }
    }

    if (mode == CUDNN_BATCHNORM_SPATIAL) {
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c) {
            float invVar = 1.f / sqrtf(var[c] + (float)epsilon);
            for (int hw=0; hw<HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                float xhat = (xf[idx] - mean[c]) * invVar;
                yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
            }
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            int idx = (n*C + c)*HW + hw;
            float invVar = 1.f / sqrtf(var[c*HW + hw] + (float)epsilon);
            float xhat = (xf[idx] - mean[c*HW + hw]) * invVar;
            yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
        }
    }

    if (resultSaveMean && resultSaveInvVariance) {
        float* sm = (float*)resultSaveMean;
        float* siv = (float*)resultSaveInvVariance;
        if (mode == CUDNN_BATCHNORM_SPATIAL) {
            for (int c=0; c<C; ++c) {
                sm[c] = mean[c];
                siv[c] = 1.f / sqrtf(var[c] + (float)epsilon);
            }
        } else {
            for (int i=0; i<C*HW; ++i) {
                sm[i] = mean[i];
                siv[i] = 1.f / sqrtf(var[i] + (float)epsilon);
            }
        }
    }

    if (resultRunningMean && resultRunningVariance) {
        float* rm = (float*)resultRunningMean;
        float* rv = (float*)resultRunningVariance;
        float factor = static_cast<float>(exponentialAverageFactor);
        if (mode == CUDNN_BATCHNORM_SPATIAL) {
            for (int c=0; c<C; ++c) {
                rm[c] = (1.f - factor) * rm[c] + factor * mean[c];
                rv[c] = (1.f - factor) * rv[c] + factor * var[c];
            }
        } else {
            for (int i=0; i<C*HW; ++i) {
                rm[i] = (1.f - factor) * rm[i] + factor * mean[i];
                rv[i] = (1.f - factor) * rv[i] + factor * var[i];
            }
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t,
    cudnnBatchNormMode_t mode,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t dxDesc, void* dx,
    cudnnTensorDescriptor_t /*dBnScaleBiasDesc*/,
    const void* bnScale,
    void* dBnScaleResult, void* dBnBiasResult,
    double /*epsilon*/,
    const void* savedMean, const void* savedInvVariance)
{
    if (!xDesc || !dyDesc || !dxDesc || !x || !dy || !dx || !bnScale ||
        !savedMean || !savedInvVariance)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* t=(TensorDesc*)xDesc;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float aD=*(const float*)alphaDataDiff, bD=*(const float*)betaDataDiff;
    float aP=*(const float*)alphaParamDiff, bP=*(const float*)betaParamDiff;
    const float* xf=(const float*)x;
    const float* dyf=(const float*)dy;
    float* dxf=(float*)dx;
    const float* sc=(const float*)bnScale;
    const float* sm=(const float*)savedMean;
    const float* siv=(const float*)savedInvVariance;
    float* dSc=(float*)dBnScaleResult;
    float* dBi=(float*)dBnBiasResult;

    int count = N * HW;

    if (mode == CUDNN_BATCHNORM_SPATIAL) {
        std::vector<float> mean_dy(C, 0.f), mean_dy_xhat(C, 0.f);
        #ifdef _OPENMP
        #pragma omp parallel for if (C > 4)
        #endif
        for (int c=0; c<C; ++c) {
            double sum_dy = 0, sum_dy_xhat = 0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                float xhat = (xf[idx] - sm[c]) * siv[c];
                sum_dy += dyf[idx];
                sum_dy_xhat += dyf[idx] * xhat;
            }
            mean_dy[c] = static_cast<float>(sum_dy / count);
            mean_dy_xhat[c] = static_cast<float>(sum_dy_xhat / count);
        }

        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            int idx = (n*C + c)*HW + hw;
            float xhat = (xf[idx] - sm[c]) * siv[c];
            float val = dyf[idx] - mean_dy[c] - xhat * mean_dy_xhat[c];
            dxf[idx] = aD * (sc[c] * siv[c] * val) + bD * dxf[idx];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if (C > 4)
        #endif
        for (int c=0; c<C; ++c) {
            double dscale = 0, dbias = 0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                float xhat = (xf[idx] - sm[c]) * siv[c];
                dscale += dyf[idx] * xhat;
                dbias  += dyf[idx];
            }
            dSc[c] = aP * static_cast<float>(dscale) + bP * dSc[c];
            dBi[c] = aP * static_cast<float>(dbias)  + bP * dBi[c];
        }
    } else {
        std::vector<float> mean_dy(C*HW, 0.f), mean_dy_xhat(C*HW, 0.f);
        #ifdef _OPENMP
        #pragma omp parallel for if (C * HW > 256)
        #endif
        for (int i=0; i<C*HW; ++i) {
            double sum_dy = 0, sum_dy_xhat = 0;
            for (int n=0; n<N; ++n) {
                int idx = n*C*HW + i;
                float xhat = (xf[idx] - sm[i]) * siv[i];
                sum_dy += dyf[idx];
                sum_dy_xhat += dyf[idx] * xhat;
            }
            mean_dy[i] = static_cast<float>(sum_dy / N);
            mean_dy_xhat[i] = static_cast<float>(sum_dy_xhat / N);
        }

        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) if (N * C * HW > 1024)
        #endif
        for (int n=0; n<N; ++n)
        for (int i=0; i<C*HW; ++i) {
            int idx = n*C*HW + i;
            float xhat = (xf[idx] - sm[i]) * siv[i];
            float val = dyf[idx] - mean_dy[i] - xhat * mean_dy_xhat[i];
            int c = i / HW;
            dxf[idx] = aD * (sc[c] * siv[i] * val) + bD * dxf[idx];
        }

        #ifdef _OPENMP
        #pragma omp parallel for if (C > 4)
        #endif
        for (int c=0; c<C; ++c) {
            double dscale = 0, dbias = 0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                int i = c*HW + hw;
                float xhat = (xf[idx] - sm[i]) * siv[i];
                dscale += dyf[idx] * xhat;
                dbias  += dyf[idx];
            }
            dSc[c] = aP * static_cast<float>(dscale) + bP * dSc[c];
            dBi[c] = aP * static_cast<float>(dbias)  + bP * dBi[c];
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize ──────────────────
// Returns the required workspace size for the Ex variant.  VGRE's in-place
// algorithm needs no extra workspace — returns 0.
cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t /*mode*/,
    cudnnBatchNormOps_t /*bnOps*/,
    cudnnTensorDescriptor_t /*xDesc*/,
    cudnnTensorDescriptor_t /*zDesc*/,
    cudnnTensorDescriptor_t /*yDesc*/,
    cudnnTensorDescriptor_t /*bnScaleBiasMeanVarDesc*/,
    cudnnActivationDescriptor_t /*activationDesc*/,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnGetBatchNormalizationBackwardExWorkspaceSize ─────────────────────────
cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t /*mode*/,
    cudnnBatchNormOps_t /*bnOps*/,
    cudnnTensorDescriptor_t /*xDesc*/,
    cudnnTensorDescriptor_t /*yDesc*/,
    cudnnTensorDescriptor_t /*dyDesc*/,
    cudnnTensorDescriptor_t /*dzDesc*/,
    cudnnTensorDescriptor_t /*dxDesc*/,
    cudnnTensorDescriptor_t /*dBnScaleBiasDesc*/,
    cudnnActivationDescriptor_t /*activationDesc*/,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnGetBatchNormalizationTrainingExReserveSpaceSize ──────────────────────
cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t /*mode*/,
    cudnnBatchNormOps_t /*bnOps*/,
    cudnnActivationDescriptor_t /*activationDesc*/,
    cudnnTensorDescriptor_t /*xDesc*/,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnBatchNormalizationForwardTrainingEx ──────────────────────────────────
// Extended BN forward: optionally fuses activation (ReLU) after normalisation.
// bnOps = 0 (BN only), 1 (BN + activation), 2 (BN + add + activation).
// VGRE implements full per-mode logic; the workspace/reserveSpace are unused.
cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t zDesc, const void* z,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* scale, const void* bias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void* /*workspace*/, size_t /*workSpaceSizeInBytes*/,
    void* /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    // Step 1: run standard BN forward (computes y = BN(x), saves stats).
    cudnnStatus_t s = cudnnBatchNormalizationForwardTraining(
        handle, mode, alpha, beta,
        xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, scale, bias,
        exponentialAverageFactor,
        resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance);
    if (s != CUDNN_STATUS_SUCCESS) return s;

    auto* td = static_cast<TensorDesc*>(yDesc);
    int total = td->n * td->c * td->h * td->w;
    float* yf = static_cast<float*>(y);

    // Step 2: optional add (bnOps == 2) — y += z element-wise.
    if ((bnOps == 2) && z && zDesc) {
        const float* zf = static_cast<const float*>(z);
        for (int i = 0; i < total; ++i) yf[i] += zf[i];
    }

    // Step 3: optional activation (bnOps 1 or 2).
    if (bnOps >= 1 && activationDesc) {
        auto* act = static_cast<ActDesc*>(activationDesc);
        for (int i = 0; i < total; ++i) {
            float v = yf[i];
            switch (act->mode) {
                case CUDNN_ACTIVATION_RELU:
                    yf[i] = v > 0.f ? v : 0.f;
                    break;
                case CUDNN_ACTIVATION_TANH:
                    yf[i] = std::tanh(v);
                    break;
                case CUDNN_ACTIVATION_SIGMOID:
                    yf[i] = 1.f / (1.f + std::exp(-v));
                    break;
                case CUDNN_ACTIVATION_CLIPPED_RELU:
                    yf[i] = std::max(0.f, std::min(v, static_cast<float>(act->coeff)));
                    break;
                case CUDNN_ACTIVATION_ELU:
                    yf[i] = v >= 0.f ? v : static_cast<float>(act->coeff) * (std::exp(v) - 1.f);
                    break;
                default: break; // IDENTITY — no-op
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnBatchNormalizationBackwardEx ─────────────────────────────────────────
// Extended BN backward: mirrors ForwardTrainingEx but for gradients.
// When bnOps >= 1, dz (gradient w.r.t. the skip/z input) is written.
// The activation backward is applied to dy before entering BN backward.
cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    cudnnTensorDescriptor_t xDesc,   const void* x,
    cudnnTensorDescriptor_t yDesc,   const void* y,
    cudnnTensorDescriptor_t dyDesc,  const void* dy,
    cudnnTensorDescriptor_t dzDesc,  void* dz,
    cudnnTensorDescriptor_t dxDesc,  void* dx,
    cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void* scale,
    const void* bias,
    void* dScale, void* dBias,
    double epsilon,
    const void* savedMean, const void* savedInvVariance,
    cudnnActivationDescriptor_t activationDesc,
    void* /*workspace*/, size_t /*workSpaceSizeInBytes*/,
    void* /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    auto* td = static_cast<TensorDesc*>(xDesc);
    int total = td->n * td->c * td->h * td->w;

    // Step 1: activation backward — produce d_bn (gradient into BN).
    // If no activation, d_bn == dy.
    std::vector<float> d_bn_storage;
    const float* d_bn = static_cast<const float*>(dy);

    if (bnOps >= 1 && activationDesc && y) {
        d_bn_storage.resize(static_cast<size_t>(total));
        const float* yf  = static_cast<const float*>(y);
        const float* dyf = static_cast<const float*>(dy);
        auto* act = static_cast<ActDesc*>(activationDesc);
        for (int i = 0; i < total; ++i) {
            float v = yf[i], g = dyf[i];
            switch (act->mode) {
                case CUDNN_ACTIVATION_RELU:
                    d_bn_storage[i] = v > 0.f ? g : 0.f; break;
                case CUDNN_ACTIVATION_TANH:
                    d_bn_storage[i] = g * (1.f - v * v); break;
                case CUDNN_ACTIVATION_SIGMOID:
                    d_bn_storage[i] = g * v * (1.f - v); break;
                case CUDNN_ACTIVATION_ELU:
                    d_bn_storage[i] = v >= 0.f ? g : g * (v + static_cast<float>(act->coeff)); break;
                default:
                    d_bn_storage[i] = g; break;
            }
        }
        d_bn = d_bn_storage.data();
    }

    // Step 2: standard BN backward (computes dx, dScale, dBias).
    // Pass d_bn as the dy parameter (we need a non-const void* cast for the API).
    cudnnStatus_t s = cudnnBatchNormalizationBackward(
        handle, mode,
        alphaDataDiff, betaDataDiff,
        alphaParamDiff, betaParamDiff,
        xDesc, x, dyDesc, static_cast<const void*>(d_bn), dxDesc, dx,
        dBnScaleBiasDesc, scale, dScale, dBias,
        epsilon, savedMean, savedInvVariance);
    if (s != CUDNN_STATUS_SUCCESS) return s;

    // Step 3: for bnOps == 2, dz = d_bn (gradient w.r.t. the addend z).
    if (bnOps == 2 && dz && dzDesc) {
        float* dzf = static_cast<float*>(dz);
        for (int i = 0; i < total; ++i) dzf[i] = d_bn[i];
    }

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
