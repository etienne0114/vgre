// cuDNN batch normalization: inference, training, backward

#include "cudnn_internal.h"

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
        for (int c=0; c<C; ++c) {
            double sum = 0.0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw)
                sum += xf[(n*C + c)*HW + hw];
            mean[c] = static_cast<float>(sum / count);
        }
        for (int c=0; c<C; ++c) {
            double sumSq = 0.0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw) {
                float d = xf[(n*C + c)*HW + hw] - mean[c];
                sumSq += d * d;
            }
            var[c] = static_cast<float>(sumSq / count);
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

        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            int idx = (n*C + c)*HW + hw;
            float xhat = (xf[idx] - sm[c]) * siv[c];
            float val = dyf[idx] - mean_dy[c] - xhat * mean_dy_xhat[c];
            dxf[idx] = aD * (sc[c] * siv[c] * val) + bD * dxf[idx];
        }

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

        for (int n=0; n<N; ++n)
        for (int i=0; i<C*HW; ++i) {
            int idx = n*C*HW + i;
            float xhat = (xf[idx] - sm[i]) * siv[i];
            float val = dyf[idx] - mean_dy[i] - xhat * mean_dy_xhat[i];
            int c = i / HW;
            dxf[idx] = aD * (sc[c] * siv[i] * val) + bD * dxf[idx];
        }

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

} // extern "C"
