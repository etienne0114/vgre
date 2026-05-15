// cuDNN Local Response Normalization (LRN) — CPU reference implementation.
//
// Supports:
//   - cudnnCreateLRNDescriptor / cudnnDestroyLRNDescriptor
//   - cudnnSetLRNDescriptor / cudnnGetLRNDescriptor
//   - cudnnLRNCrossChannelForward
//   - cudnnLRNCrossChannelBackward
//
// LRN formula (cross-channel, dim=1):
//   scale[n,c,h,w] = (k + alpha/N * sum_{j} x[n,j,h,w]^2)^beta
//   y[n,c,h,w]    = x[n,c,h,w] / scale[n,c,h,w]
// where the sum is over j = [c - floor((N-1)/2), c + floor(N/2)] clipped to [0,C-1].

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"
#include <cmath>
#include <vector>

extern "C" {

// ── Descriptor lifecycle ─────────────────────────────────────────────────────

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new LRNDesc{5, 1e-4, 0.75, 2.0};
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t desc) {
    delete static_cast<LRNDesc*>(desc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t desc,
                                    unsigned lrnN,
                                    double lrnAlpha,
                                    double lrnBeta,
                                    double lrnK) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = static_cast<LRNDesc*>(desc);
    d->lrnN     = lrnN;
    d->lrnAlpha = lrnAlpha;
    d->lrnBeta  = lrnBeta;
    d->lrnK     = lrnK;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t desc,
                                    unsigned* lrnN,
                                    double* lrnAlpha,
                                    double* lrnBeta,
                                    double* lrnK) {
    if (!desc || !lrnN || !lrnAlpha || !lrnBeta || !lrnK)
        return CUDNN_STATUS_INVALID_VALUE;
    auto* d = static_cast<LRNDesc*>(desc);
    *lrnN     = d->lrnN;
    *lrnAlpha = d->lrnAlpha;
    *lrnBeta  = d->lrnBeta;
    *lrnK     = d->lrnK;
    return CUDNN_STATUS_SUCCESS;
}

// ── Forward LRN ──────────────────────────────────────────────────────────────
// y = x / (k + alpha/N * sum_{j in window} x_j^2)^beta

cudnnStatus_t cudnnLRNCrossChannelForward(
    cudnnHandle_t /*handle*/,
    cudnnLRNDescriptor_t desc,
    cudnnLRNMode_t /*mode*/,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    const void* beta,
    cudnnTensorDescriptor_t yDesc,
    void* y)
{
    if (!desc || !alpha || !beta || !xDesc || !yDesc || !x || !y)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xd = static_cast<TensorDesc*>(xDesc);
    auto* yd = static_cast<TensorDesc*>(yDesc);
    if (xd->n != yd->n || xd->c != yd->c || xd->h != yd->h || xd->w != yd->w)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* lrn = static_cast<LRNDesc*>(desc);
    const int N = xd->n, C = xd->c, H = xd->h, W = xd->w;
    const float a = *(const float*)alpha;
    const float b = *(const float*)beta;
    const float* xptr = static_cast<const float*>(x);
    float*       yptr = static_cast<float*>(y);

    const unsigned lrnN    = lrn->lrnN;
    const double   lrnAlpha = lrn->lrnAlpha;
    const double   lrnBeta  = lrn->lrnBeta;
    const double   lrnK     = lrn->lrnK;

    const int halfWindow = static_cast<int>(lrnN) / 2;

    // Compute scale and output in one pass
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (N * H * W > 256)
    #endif
    for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        // First compute per-channel squared sums
        for (int c = 0; c < C; ++c) {
            double sqSum = 0.0;
            int cStart = c - halfWindow;
            int cEnd   = c + static_cast<int>(lrnN) - halfWindow;
            // cuDNN semantics: window is [c - floor((N-1)/2), c + floor(N/2)]
            // For odd N: symmetric around c
            // For even N: slightly shifted (cuDNN uses halfWindow = N/2)
            cStart = std::max(0, cStart);
            cEnd   = std::min(C, cEnd);
            for (int j = cStart; j < cEnd; ++j) {
                float v = xptr[((n * C + j) * H + h) * W + w];
                sqSum += double(v) * double(v);
            }
            double scale = std::pow(lrnK + (lrnAlpha / lrnN) * sqSum, lrnBeta);
            int idx = ((n * C + c) * H + h) * W + w;
            float outVal = xptr[idx] / static_cast<float>(scale);
            yptr[idx] = a * outVal + b * yptr[idx];
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Backward LRN ─────────────────────────────────────────────────────────────
// dx = a * [ dy/scale - 2*alpha*beta/N * x * sum_{j in window}(dy[j]*x[j]/scale[j]) ] + b * dx

cudnnStatus_t cudnnLRNCrossChannelBackward(
    cudnnHandle_t /*handle*/,
    cudnnLRNDescriptor_t desc,
    cudnnLRNMode_t /*mode*/,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    cudnnTensorDescriptor_t dyDesc,
    const void* dy,
    cudnnTensorDescriptor_t dxDesc,
    void* dx,
    const void* beta)
{
    if (!desc || !alpha || !beta || !xDesc || !dyDesc || !dxDesc || !x || !dy || !dx)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xd  = static_cast<TensorDesc*>(xDesc);
    auto* dyd = static_cast<TensorDesc*>(dyDesc);
    auto* dxd = static_cast<TensorDesc*>(dxDesc);
    if (xd->n != dyd->n || xd->c != dyd->c || xd->h != dyd->h || xd->w != dyd->w ||
        xd->n != dxd->n || xd->c != dxd->c || xd->h != dxd->h || xd->w != dxd->w)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* lrn = static_cast<LRNDesc*>(desc);
    const int N = xd->n, C = xd->c, H = xd->h, W = xd->w;
    const float a = *(const float*)alpha;
    const float b = *(const float*)beta;
    const float* xptr  = static_cast<const float*>(x);
    const float* dyptr = static_cast<const float*>(dy);
    float*       dxptr = static_cast<float*>(dx);

    const unsigned lrnN    = lrn->lrnN;
    const double   lrnAlpha = lrn->lrnAlpha;
    const double   lrnBeta  = lrn->lrnBeta;
    const double   lrnK     = lrn->lrnK;
    const int halfWindow = static_cast<int>(lrnN) / 2;

    // Precompute scale[j] = (k + alpha/N * sum_{i in window(j)} x_i^2)^beta
    // and also store x[j]/scale[j] for efficiency
    std::vector<double> scale(N * C * H * W);
    std::vector<double> xOverScale(N * C * H * W);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (N * C * H * W > 256)
    #endif
    for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
    for (int c = 0; c < C; ++c) {
        double sqSum = 0.0;
        int cStart = std::max(0, c - halfWindow);
        int cEnd   = std::min(C, c + static_cast<int>(lrnN) - halfWindow);
        for (int j = cStart; j < cEnd; ++j) {
            float v = xptr[((n * C + j) * H + h) * W + w];
            sqSum += double(v) * double(v);
        }
        int idx = ((n * C + c) * H + h) * W + w;
        double sc = std::pow(lrnK + (lrnAlpha / lrnN) * sqSum, lrnBeta);
        scale[idx] = sc;
        xOverScale[idx] = double(xptr[idx]) / sc;
    }

    // Compute backward:
    // For each c, dx[c] = dy[c]/scale[c] - 2*alpha*beta/N * x[c] * sum_{j: c in window(j)} dy[j]*x[j]/scale[j]^2 * scale[j]
    // Actually from derivative of y = x * (k + alpha/N * S)^(-beta) where S = sum x_j^2
    // Let P = (k + alpha/N * S)^(-beta)
    // y = x * P
    // dL/dx_c = dy_c * P + x_c * dP/dx_c
    // dP/dS = -beta * (k + alpha/N * S)^(-beta-1) * alpha/N
    // dS/dx_c = 2 * x_c (for terms where c is in the window)
    // But S_j includes x_c if c is in window(j), so:
    // dP_j/dx_c = -beta * (k + alpha/N * S_j)^(-beta-1) * alpha/N * 2*x_c * I(c in window(j))
    // dL/dx_c = sum_j dy_j * [ delta_{jc} * P_j + x_j * dP_j/dx_c ]
    //         = dy_c * P_c + sum_j dy_j * x_j * (-2*alpha*beta/N) * x_c * (k+alpha/N*S_j)^(-beta-1) * I(c in window(j))
    //         = dy_c / scale_c - (2*alpha*beta/N) * x_c * sum_{j: c in window(j)} dy_j * x_j / (scale_j * (k + alpha/N * S_j))
    // Note: scale_j = (k + alpha/N * S_j)^beta, so (k + alpha/N * S_j) = scale_j^(1/beta)
    // For simplicity we compute the summation term directly.

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (N * H * W > 256)
    #endif
    for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w) {
        for (int c = 0; c < C; ++c) {
            int idx = ((n * C + c) * H + h) * W + w;
            double term1 = dyptr[idx] / scale[idx];

            double term2 = 0.0;
            // c is in window(j) when j - halfWindow <= c < j + lrnN - halfWindow
            // => j in [c - (lrnN - halfWindow) + 1, c + halfWindow]
            // More simply: j in [c - halfWindow, c + halfWindow] with window size lrnN
            int jStart = std::max(0, c - halfWindow);
            int jEnd   = std::min(C, c + static_cast<int>(lrnN) - halfWindow);
            for (int j = jStart; j < jEnd; ++j) {
                int jdx = ((n * C + j) * H + h) * W + w;
                // Recompute S_j for the derivative coefficient
                double sqSumJ = 0.0;
                int cStartJ = std::max(0, j - halfWindow);
                int cEndJ   = std::min(C, j + static_cast<int>(lrnN) - halfWindow);
                for (int i = cStartJ; i < cEndJ; ++i) {
                    float v = xptr[((n * C + i) * H + h) * W + w];
                    sqSumJ += double(v) * double(v);
                }
                double baseJ = lrnK + (lrnAlpha / lrnN) * sqSumJ;
                double P_deriv = -lrnBeta * std::pow(baseJ, -lrnBeta - 1.0) * (lrnAlpha / lrnN);
                term2 += double(dyptr[jdx]) * double(xptr[jdx]) * P_deriv * 2.0 * double(xptr[idx]);
            }

            double dxVal = term1 + term2;
            dxptr[idx] = a * static_cast<float>(dxVal) + b * dxptr[idx];
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
