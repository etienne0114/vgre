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
    for (int i = 0; i < Nx; ++i)
        yf[i] = a * applyActivation(xf[i], *act) + b * yf[i];
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

    for (int i = 0; i < Nx; ++i) {
        float deriv = applyActivationDerivative(xf[i], yf[i], *act);
        dxf[i] = a * dyf[i] * deriv + b * dxf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
