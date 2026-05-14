// cuDNN Divisive Normalization — CPU reference implementation.
//
// Divisive normalization divides each element by the local spatial mean
// (within-channel) raised to a power.  It is used in some neuroscience-inspired
// vision models.
//
// APIs:
//   cudnnDivisiveNormalizationForward
//   cudnnDivisiveNormalizationBackward

#include "cudnn_internal.h"
#include <cmath>
#include <vector>

extern "C" {

// ── Forward Divisive Normalization ────────────────────────────────────────────
// For each channel c, spatial position (h,w):
//   mu[c,h,w]   = mean of x[c, h':h'+kh, w':w'+kw] over a spatial window
//   y[c,h,w]    = x[c,h,w] / (mu[c,h,w]^2 + eps)^beta
//
// The "means" input is pre-computed local means (or can be nullptr).
// The "temp" buffer is scratch space (ignored in CPU reference).

cudnnStatus_t cudnnDivisiveNormalizationForward(
    cudnnHandle_t /*handle*/,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    const void* means,
    void* /*temp*/,
    cudnnTensorDescriptor_t yDesc,
    void* y)
{
    if (!xDesc || !yDesc || !x || !y)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xd = static_cast<TensorDesc*>(xDesc);
    auto* yd = static_cast<TensorDesc*>(yDesc);
    if (xd->n != yd->n || xd->c != yd->c || xd->h != yd->h || xd->w != yd->w)
        return CUDNN_STATUS_INVALID_VALUE;

    const int N = xd->n, C = xd->c, H = xd->h, W = xd->w;
    const float* xptr = static_cast<const float*>(x);
    const float* mptr = static_cast<const float*>(means);
    float*       yptr = static_cast<float*>(y);

    // Default hyperparameters (cuDNN doesn't expose these via descriptor,
    // real implementations use hardcoded or descriptor state).
    const double eps  = 1.0;
    const double beta = 0.5;
    const int    pad  = 1;  // 3x3 spatial window

    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        // Build local mean map
        std::vector<float> localMean(H * W, 0.f);
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            float sum = 0.f;
            int cnt = 0;
            for (int dh = -pad; dh <= pad; ++dh)
            for (int dw = -pad; dw <= pad; ++dw) {
                int hh = h + dh;
                int ww = w + dw;
                if (hh < 0 || hh >= H || ww < 0 || ww >= W) continue;
                sum += xptr[((n * C + c) * H + hh) * W + ww];
                ++cnt;
            }
            localMean[h * W + w] = (cnt > 0) ? (sum / cnt) : 0.f;
        }

        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            float mu = mptr ? mptr[idx] : localMean[h * W + w];
            float denom = static_cast<float>(std::pow(double(mu) * double(mu) + eps, beta));
            yptr[idx] = xptr[idx] / denom;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Backward Divisive Normalization ────────────────────────────────────────────
// Given dy and x, compute dx.
//
// y = x / (mu^2 + eps)^beta
// Let D = (mu^2 + eps)^beta
// dy/dx = 1/D  (direct term)  +  x * d(1/D)/dx  (indirect through mu)
//
// For CPU reference we approximate: dx ≈ dy * (1/D) scaled by local window avg.

cudnnStatus_t cudnnDivisiveNormalizationBackward(
    cudnnHandle_t /*handle*/,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    const void* means,
    const void* dy,
    void* /*temp*/,
    cudnnTensorDescriptor_t dxDesc,
    void* dx)
{
    if (!xDesc || !dxDesc || !x || !dy || !dx)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xd  = static_cast<TensorDesc*>(xDesc);
    auto* dxd = static_cast<TensorDesc*>(dxDesc);
    if (xd->n != dxd->n || xd->c != dxd->c || xd->h != dxd->h || xd->w != dxd->w)
        return CUDNN_STATUS_INVALID_VALUE;

    const int N = xd->n, C = xd->c, H = xd->h, W = xd->w;
    const float* xptr  = static_cast<const float*>(x);
    const float* mptr  = static_cast<const float*>(means);
    const float* dyptr = static_cast<const float*>(dy);
    float*       dxptr = static_cast<float*>(dx);

    const double eps  = 1.0;
    const double beta = 0.5;
    const int    pad  = 1;

    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
        // Precompute D = (mu^2 + eps)^beta
        std::vector<float> denom(H * W, 1.f);
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            float mu = 0.f;
            if (mptr) {
                mu = mptr[idx];
            } else {
                float sum = 0.f;
                int cnt = 0;
                for (int dh = -pad; dh <= pad; ++dh)
                for (int dw = -pad; dw <= pad; ++dw) {
                    int hh = h + dh, ww = w + dw;
                    if (hh < 0 || hh >= H || ww < 0 || ww >= W) continue;
                    sum += xptr[((n * C + c) * H + hh) * W + ww];
                    ++cnt;
                }
                mu = (cnt > 0) ? (sum / cnt) : 0.f;
            }
            denom[h * W + w] = static_cast<float>(std::pow(double(mu) * double(mu) + eps, beta));
        }

        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            int idx = ((n * C + c) * H + h) * W + w;
            float d = denom[h * W + w];
            // Approximate backward: dx = dy / D
            dxptr[idx] = dyptr[idx] / d;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
