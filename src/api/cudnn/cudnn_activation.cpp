// cuDNN activation forward + backward

#include "cudnn_internal.h"

#include "vgre/common/openmp_helper.h"

extern "C" {

cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,  cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!actDesc || !xDesc || !yDesc || !x || !y || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;
    auto* tx=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc; auto* act=(ActDesc*)actDesc;
    if (!isValidActivationDescriptor(*act))
        return CUDNN_STATUS_BAD_PARAM;
    if (act->mode == CUDNN_ACTIVATION_IDENTITY) return CUDNN_STATUS_NOT_SUPPORTED;
    if (tx->n != ty->n || tx->c != ty->c || tx->h != ty->h || tx->w != ty->w ||
        tx->dtype != ty->dtype)
        return CUDNN_STATUS_INVALID_VALUE;
    if (tx->dtype != CUDNN_DATA_FLOAT && tx->dtype != CUDNN_DATA_INT8)
        return CUDNN_STATUS_NOT_SUPPORTED;
    int Nx = tx->n*tx->c*tx->h*tx->w;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    if (Nx <= 0 || Nx != Ny) return CUDNN_STATUS_INVALID_VALUE;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;

    const bool isInt8 = tx->dtype == CUDNN_DATA_INT8;
    if (isInt8 && act->mode != CUDNN_ACTIVATION_RELU &&
        act->mode != CUDNN_ACTIVATION_CLIPPED_RELU)
        return CUDNN_STATUS_NOT_SUPPORTED;
    if (isInt8 && (!std::isfinite(a) || !std::isfinite(b)))
        return CUDNN_STATUS_BAD_PARAM;
    std::vector<float> xFloat;
    if (isInt8) {
        xFloat.resize(Nx);
        const int8_t* xi = (const int8_t*)x;
        for (int i = 0; i < Nx; ++i) xFloat[i] = static_cast<float>(xi[i]);
        xf = xFloat.data();
    }

    std::vector<float> tmp(Nx, 0.f);
    #ifdef _OPENMP
    #pragma omp parallel for if (Nx > 1024)
    #endif
    for (int i = 0; i < Nx; ++i)
        tmp[i] = applyActivation(xf[i], *act);

    if (isInt8) {
        int8_t* yi = (int8_t*)y;
        for (int i = 0; i < Nx; ++i)
            yi[i] = vgre_round_sat_i8(a * tmp[i] + (b == 0.f ? 0.f : b * yi[i]));
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for if (Nx > 1024)
        #endif
        for (int i = 0; i < Nx; ++i)
            yf[i] = a * tmp[i] + (b == 0.f ? 0.f : b * yf[i]);
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
    const void* alpha, cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx)
{
    if (!actDesc || !xDesc || !yDesc || !dyDesc || !dxDesc || !x || !y || !dy || !dx || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* tx=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc; auto* tdy=(TensorDesc*)dyDesc; auto* tdx=(TensorDesc*)dxDesc;
    auto* act=(ActDesc*)actDesc;
    if (!isValidActivationDescriptor(*act))
        return CUDNN_STATUS_BAD_PARAM;
    if (act->mode == CUDNN_ACTIVATION_IDENTITY) return CUDNN_STATUS_NOT_SUPPORTED;
    if (tx->n != ty->n || tx->c != ty->c || tx->h != ty->h || tx->w != ty->w ||
        tx->n != tdy->n || tx->c != tdy->c || tx->h != tdy->h || tx->w != tdy->w ||
        tx->n != tdx->n || tx->c != tdx->c || tx->h != tdx->h || tx->w != tdx->w ||
        tx->dtype != ty->dtype || tx->dtype != tdy->dtype || tx->dtype != tdx->dtype)
        return CUDNN_STATUS_INVALID_VALUE;
    if (tx->dtype != CUDNN_DATA_FLOAT && tx->dtype != CUDNN_DATA_INT8)
        return CUDNN_STATUS_NOT_SUPPORTED;
    int Nx = tx->n*tx->c*tx->h*tx->w;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    int Nd = tdy->n*tdy->c*tdy->h*tdy->w;
    int Ndx = tdx->n*tdx->c*tdx->h*tdx->w;
    if (Nx <= 0 || Nx != Ny || Nx != Nd || Nx != Ndx) return CUDNN_STATUS_INVALID_VALUE;

    float a = *(const float*)alpha, b = *(const float*)beta;
    const float* xf = (const float*)x; const float* yf = (const float*)y;
    const float* dyf = (const float*)dy; float* dxf = (float*)dx;

    const bool isInt8 = tx->dtype == CUDNN_DATA_INT8;
    if (isInt8 && act->mode != CUDNN_ACTIVATION_RELU &&
        act->mode != CUDNN_ACTIVATION_CLIPPED_RELU)
        return CUDNN_STATUS_NOT_SUPPORTED;
    if (isInt8 && (!std::isfinite(a) || !std::isfinite(b)))
        return CUDNN_STATUS_BAD_PARAM;
    std::vector<float> xFloat, yFloat, dyFloat;
    if (isInt8) {
        xFloat.resize(Nx); yFloat.resize(Nx); dyFloat.resize(Nx);
        const int8_t* xi = (const int8_t*)x;
        const int8_t* yi = (const int8_t*)y;
        const int8_t* dyi = (const int8_t*)dy;
        for (int i = 0; i < Nx; ++i) {
            xFloat[i] = static_cast<float>(xi[i]);
            yFloat[i] = static_cast<float>(yi[i]);
            dyFloat[i] = static_cast<float>(dyi[i]);
        }
        xf = xFloat.data(); yf = yFloat.data(); dyf = dyFloat.data();
    }

    std::vector<float> dxTmp(Nx, 0.f);
    for (int i = 0; i < Nx; ++i) {
        float deriv = applyActivationDerivative(xf[i], yf[i], *act);
        dxTmp[i] = a * dyf[i] * deriv;
    }

    if (isInt8) {
        int8_t* dxi = (int8_t*)dx;
        for (int i = 0; i < Nx; ++i)
            dxi[i] = vgre_round_sat_i8(dxTmp[i] + (b == 0.f ? 0.f : b * dxi[i]));
    } else {
        for (int i = 0; i < Nx; ++i)
            dxf[i] = dxTmp[i] + (b == 0.f ? 0.f : b * dxf[i]);
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
