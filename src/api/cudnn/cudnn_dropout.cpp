// cuDNN dropout forward + backward + reserve space query

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"

#include <cstdint>

// SplitMix64 avalanche hash: maps a 64-bit counter to a uniform 64-bit hash.
// Used as a counter-based RNG: hash(seed ^ index) → uniform mask.
// O(1) per element, no state initialization overhead, trivially parallelizable.
// Constants from Sebastiano Vigna "Further scramblings of Marsaglia's xorshift
// generators" (2018).
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Map 64-bit hash to float in [0, 1).
static inline float hashToFloat(uint64_t h) {
    // Top 23 mantissa bits → IEEE-754 float in [1,2); subtract 1 → [0,1).
    uint32_t m = static_cast<uint32_t>(h >> 41) | 0x3f800000u;
    float f; __builtin_memcpy(&f, &m, sizeof(f));
    return f - 1.0f;
}

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
    uint64_t seed = d->seed;

    // Counter-based RNG: each element i gets a unique hash splitmix64(seed ^ i).
    // No per-element state allocation — O(1) arithmetic per element, fully
    // data-parallel, identical mask sequence regardless of thread count.
    #ifdef _OPENMP
    #pragma omp parallel for if (N > 1024)
    #endif
    for (int i = 0; i < N; ++i) {
        float r = hashToFloat(splitmix64(seed ^ static_cast<uint64_t>(i)));
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
