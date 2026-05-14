// cuDNN activation forward + backward

#include "cudnn_internal.h"

extern "C" {

cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,  cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta) return CUDNN_STATUS_INVALID_VALUE;
    auto* tx=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc; auto* act=(ActDesc*)actDesc;
    int Nx = tx->n*tx->c*tx->h*tx->w;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    if (Nx != Ny) return CUDNN_STATUS_INVALID_VALUE;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;

    bool isInt8 = (tx->dtype == CUDNN_DATA_INT8 || tx->dtype == CUDNN_DATA_INT8x4 ||
                   tx->dtype == CUDNN_DATA_INT8x32);
    std::vector<float> xFloat;
    if (isInt8) {
        float scale = (a > 1e-8f) ? a : (1.f / 128.f);
        xFloat.resize(Nx);
        const int8_t* xi = (const int8_t*)x;
        for (int i = 0; i < Nx; ++i) xFloat[i] = vgre_dequant_i8(xi[i], scale);
        xf = xFloat.data();
    }

    std::vector<float> tmp(Nx, 0.f);
    for (int i = 0; i < Nx; ++i)
        tmp[i] = applyActivation(xf[i], *act);

    if (isInt8) {
        float outScale = (a > 1e-8f) ? a : (1.f / 128.f);
        float invOutScale = 1.f / outScale;
        int8_t* yi = (int8_t*)y;
        for (int i = 0; i < Nx; ++i)
            yi[i] = vgre_quant_f32_to_i8(tmp[i], invOutScale);
    } else {
        for (int i = 0; i < Nx; ++i)
            yf[i] = a * tmp[i] + b * yf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
    const void* alpha, cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx)
{
    if (!xDesc || !yDesc || !dyDesc || !dxDesc || !x || !y || !dy || !dx || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* tx=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc; auto* tdy=(TensorDesc*)dyDesc; auto* tdx=(TensorDesc*)dxDesc;
    auto* act=(ActDesc*)actDesc;
    int Nx = tx->n*tx->c*tx->h*tx->w;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    int Nd = tdy->n*tdy->c*tdy->h*tdy->w;
    int Ndx = tdx->n*tdx->c*tdx->h*tdx->w;
    if (Nx != Ny || Nx != Nd || Nx != Ndx) return CUDNN_STATUS_INVALID_VALUE;

    float a = *(const float*)alpha, b = *(const float*)beta;
    const float* xf = (const float*)x; const float* yf = (const float*)y;
    const float* dyf = (const float*)dy; float* dxf = (float*)dx;

    bool isInt8 = (tx->dtype == CUDNN_DATA_INT8 || tx->dtype == CUDNN_DATA_INT8x4 ||
                   tx->dtype == CUDNN_DATA_INT8x32);
    std::vector<float> xFloat, yFloat, dyFloat;
    if (isInt8) {
        float scale = (a > 1e-8f) ? a : (1.f / 128.f);
        xFloat.resize(Nx); yFloat.resize(Nx); dyFloat.resize(Nx);
        const int8_t* xi = (const int8_t*)x;
        const int8_t* yi = (const int8_t*)y;
        const int8_t* dyi = (const int8_t*)dy;
        for (int i = 0; i < Nx; ++i) {
            xFloat[i] = vgre_dequant_i8(xi[i], scale);
            yFloat[i] = vgre_dequant_i8(yi[i], scale);
            dyFloat[i] = vgre_dequant_i8(dyi[i], scale);
        }
        xf = xFloat.data(); yf = yFloat.data(); dyf = dyFloat.data();
    }

    std::vector<float> dxTmp(Nx, 0.f);
    for (int i = 0; i < Nx; ++i) {
        float deriv = applyActivationDerivative(xf[i], yf[i], *act);
        dxTmp[i] = a * dyf[i] * deriv;
    }

    if (isInt8) {
        float outScale = (a > 1e-8f) ? a : (1.f / 128.f);
        float invOutScale = 1.f / outScale;
        int8_t* dxi = (int8_t*)dx;
        for (int i = 0; i < Nx; ++i)
            dxi[i] = vgre_quant_f32_to_i8(dxTmp[i], invOutScale);
    } else {
        for (int i = 0; i < Nx; ++i)
            dxf[i] = dxTmp[i] + b * dxf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
