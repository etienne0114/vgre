// cuDNN dropout forward + backward + reserve space query

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"

extern "C" {

cudnnStatus_t cudnnGetDropoutReserveSpaceSize(
    cudnnTensorDescriptor_t xDesc,
    size_t* sizeInBytes)
{
    if (!xDesc || !sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    auto* t = (TensorDesc*)xDesc;
    *sizeInBytes = t->n * t->c * t->h * t->w * sizeof(uint8_t);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward(
    cudnnHandle_t,
    cudnnDropoutDescriptor_t desc,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    cudnnTensorDescriptor_t yDesc,
    void* y,
    void* reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    if (!desc || !xDesc || !yDesc || !x || !y || !reserveSpace)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* d = (DropoutDesc*)desc;
    auto* t = (TensorDesc*)xDesc;
    int N = t->n * t->c * t->h * t->w;
    size_t expectedSize = N * sizeof(uint8_t);
    if (reserveSpaceSizeInBytes < expectedSize) return CUDNN_STATUS_INVALID_VALUE;

    float scale = 1.0f / (1.0f - d->dropout);
    const float* xf = (const float*)x;
    float* yf = (float*)y;
    uint8_t* mask = (uint8_t*)reserveSpace;

    #ifdef _OPENMP
    #pragma omp parallel for if (N > 1024)
    #endif
    for (int i = 0; i < N; ++i) {
        // Deterministic per-element RNG: seed + i
        std::mt19937_64 elemRng(d->seed + static_cast<uint64_t>(i));
        std::uniform_real_distribution<float> elemDist(0.0f, 1.0f);
        float r = elemDist(elemRng);
        if (r < d->dropout) {
            mask[i] = 0;
            yf[i] = 0.0f;
        } else {
            mask[i] = 1;
            yf[i] = xf[i] * scale;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward(
    cudnnHandle_t,
    cudnnDropoutDescriptor_t desc,
    cudnnTensorDescriptor_t dyDesc,
    const void* dy,
    cudnnTensorDescriptor_t dxDesc,
    void* dx,
    void* reserveSpace,
    size_t reserveSpaceSizeInBytes)
{
    if (!desc || !dyDesc || !dxDesc || !dy || !dx || !reserveSpace)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* d = (DropoutDesc*)desc;
    auto* td = (TensorDesc*)dyDesc;
    int N = td->n * td->c * td->h * td->w;
    size_t expectedSize = N * sizeof(uint8_t);
    if (reserveSpaceSizeInBytes < expectedSize) return CUDNN_STATUS_INVALID_VALUE;

    float scale = 1.0f / (1.0f - d->dropout);
    const float* dyf = (const float*)dy;
    float* dxf = (float*)dx;
    const uint8_t* mask = (const uint8_t*)reserveSpace;

    #ifdef _OPENMP
    #pragma omp parallel for if (N > 1024)
    #endif
    for (int i = 0; i < N; ++i) {
        dxf[i] = mask[i] ? (dyf[i] * scale) : 0.0f;
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
