// cuDNN Spatial Transformer Network (STN) — CPU reference implementation.
//
// Implements the differentiable bilinear sampler of Jaderberg et al. 2015.
// Grid generator: affine theta (N×2×3) → sampling grid (N×H×W×2)
// Sampler forward:  bilinear interpolation of input at grid coordinates
// Sampler backward: scatter-add gradients to dx and dgrid

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"
#include <algorithm>
#include <cmath>

extern "C" {

// ── Descriptor ───────────────────────────────────────────────────────────────

struct SpatialTfDesc {
    int samplerType = 0;  // 0 = bilinear
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int nbDims = 4;
    int dimA[4] = {1, 1, 1, 1};  // N, C, H, W of output
};
typedef SpatialTfDesc* cudnnSpatialTransformerDescriptor_t;

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t* desc)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new SpatialTfDesc();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t desc)
{
    delete desc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t desc,
    int samplerType, cudnnDataType_t dataType,
    int nbDims, const int dimA[])
{
    if (!desc || !dimA || nbDims < 1) return CUDNN_STATUS_INVALID_VALUE;
    desc->samplerType = samplerType;
    desc->dataType    = dataType;
    desc->nbDims      = nbDims;
    for (int i = 0; i < nbDims && i < 4; ++i) desc->dimA[i] = dimA[i];
    return CUDNN_STATUS_SUCCESS;
}

// ── Grid generator: theta (N,2,3) → grid (N,H,W,2) ─────────────────────────
// grid[n,h,w,0] = theta[n,0,:] · [x_norm, y_norm, 1]
// grid[n,h,w,1] = theta[n,1,:] · [x_norm, y_norm, 1]
// where x_norm = 2w/(W-1)-1, y_norm = 2h/(H-1)-1

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t /*handle*/,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void* theta,
    void* grid)
{
    if (!stDesc || !theta || !grid) return CUDNN_STATUS_INVALID_VALUE;

    const int N    = stDesc->dimA[0];
    const int H    = stDesc->dimA[2];
    const int W    = stDesc->dimA[3];
    const float* th = static_cast<const float*>(theta);
    float*       gr = static_cast<float*>(grid);

    const float Wm1inv = (W > 1) ? 2.0f / (W - 1) : 0.0f;
    const float Hm1inv = (H > 1) ? 2.0f / (H - 1) : 0.0f;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (N * H * W > 1024)
    #endif
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            const float* th_n = th + n * 6;  // 2×3 matrix, row-major
            float y_norm = h * Hm1inv - 1.0f;
            for (int w = 0; w < W; ++w) {
                float x_norm = w * Wm1inv - 1.0f;
                // gx = th[0]*x + th[1]*y + th[2]
                // gy = th[3]*x + th[4]*y + th[5]
                float gx = th_n[0] * x_norm + th_n[1] * y_norm + th_n[2];
                float gy = th_n[3] * x_norm + th_n[4] * y_norm + th_n[5];
                int base = ((n * H + h) * W + w) * 2;
                gr[base + 0] = gx;
                gr[base + 1] = gy;
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Sampler forward: bilinear interpolation ───────────────────────────────────
// y[n,c,h_out,w_out] = alpha * bilinear(x[n,c,:,:], grid[n,h_out,w_out]) + beta * y

cudnnStatus_t cudnnSpatialTfSamplerForward(
    cudnnHandle_t /*handle*/,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void* alpha_ptr,
    cudnnTensorDescriptor_t xDesc,
    const void* x,
    const void* grid,
    const void* beta_ptr,
    cudnnTensorDescriptor_t yDesc,
    void* y)
{
    if (!stDesc || !xDesc || !yDesc || !x || !y || !grid) return CUDNN_STATUS_INVALID_VALUE;

    auto* xd = static_cast<TensorDesc*>(xDesc);
    auto* yd = static_cast<TensorDesc*>(yDesc);
    const float alpha = alpha_ptr ? *static_cast<const float*>(alpha_ptr) : 1.0f;
    const float beta  = beta_ptr  ? *static_cast<const float*>(beta_ptr)  : 0.0f;

    const int N    = xd->n, C = xd->c, H_in = xd->h, W_in = xd->w;
    const int H_out = yd->h, W_out = yd->w;
    const float* xf  = static_cast<const float*>(x);
    const float* gr  = static_cast<const float*>(grid);
    float*       yf  = static_cast<float*>(y);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (N * C > 4)
    #endif
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* xnc = xf + (n * C + c) * H_in * W_in;
            float*       ync = yf + (n * C + c) * H_out * W_out;
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int gbase = ((n * H_out + h) * W_out + w) * 2;
                    float gx  = gr[gbase + 0];
                    float gy  = gr[gbase + 1];
                    // Map from [-1,1] to pixel coords
                    float px = (gx + 1.0f) * 0.5f * (W_in - 1);
                    float py = (gy + 1.0f) * 0.5f * (H_in - 1);
                    // Clamp to valid range
                    px = std::max(0.0f, std::min(static_cast<float>(W_in - 1), px));
                    py = std::max(0.0f, std::min(static_cast<float>(H_in - 1), py));
                    int x0 = static_cast<int>(px),  x1 = std::min(x0 + 1, W_in - 1);
                    int y0 = static_cast<int>(py),  y1 = std::min(y0 + 1, H_in - 1);
                    float wx = px - x0, wy = py - y0;
                    float v = xnc[y0 * W_in + x0] * (1 - wy) * (1 - wx)
                            + xnc[y0 * W_in + x1] * (1 - wy) * wx
                            + xnc[y1 * W_in + x0] * wy       * (1 - wx)
                            + xnc[y1 * W_in + x1] * wy       * wx;
                    int oi = h * W_out + w;
                    ync[oi] = alpha * v + beta * ync[oi];
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Sampler backward ──────────────────────────────────────────────────────────
// Computes dx (input gradient) and dgrid (grid coordinate gradient).
// dx scatter-adds bilinear-weighted dy.
// dgrid accumulates dy * d(bilinear)/d(gx) and d(bilinear)/d(gy).

cudnnStatus_t cudnnSpatialTfSamplerBackward(
    cudnnHandle_t /*handle*/,
    cudnnSpatialTransformerDescriptor_t stDesc,
    const void* alpha_ptr,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta_ptr,
    cudnnTensorDescriptor_t dxDesc, void* dx,
    const void* alphaDgrid_ptr,
    const void* grid,
    const void* betaDgrid_ptr,
    void* dgrid,
    cudnnTensorDescriptor_t dyDesc, const void* dy)
{
    if (!stDesc || !xDesc || !dxDesc || !dyDesc || !x || !dx || !dy) return CUDNN_STATUS_INVALID_VALUE;

    auto* xd  = static_cast<TensorDesc*>(xDesc);
    auto* dxd = static_cast<TensorDesc*>(dxDesc);
    auto* dyd = static_cast<TensorDesc*>(dyDesc);
    const float alpha      = alpha_ptr      ? *static_cast<const float*>(alpha_ptr)      : 1.0f;
    const float beta       = beta_ptr       ? *static_cast<const float*>(beta_ptr)       : 0.0f;
    const float alphaDgrid = alphaDgrid_ptr ? *static_cast<const float*>(alphaDgrid_ptr) : 1.0f;
    const float betaDgrid  = betaDgrid_ptr  ? *static_cast<const float*>(betaDgrid_ptr)  : 0.0f;

    const int N    = xd->n, C = xd->c, H_in = xd->h, W_in = xd->w;
    const int H_out = dyd->h, W_out = dyd->w;
    const float* xf  = static_cast<const float*>(x);
    const float* dyf = static_cast<const float*>(dy);
    const float* gr  = grid  ? static_cast<const float*>(grid)  : nullptr;
    float*       dxf = static_cast<float*>(dx);
    float*       dgr = dgrid ? static_cast<float*>(dgrid) : nullptr;
    (void)xf;  // x not needed for bilinear backward (only grid and dy)

    // Scale dx by beta, scale dgrid by betaDgrid
    int dxSize = N * C * H_in * W_in;
    for (int i = 0; i < dxSize; ++i) dxf[i] *= beta;
    if (dgr) {
        int dgrSize = N * H_out * W_out * 2;
        for (int i = 0; i < dgrSize; ++i) dgr[i] *= betaDgrid;
    }

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                float gx = 0.f, gy = 0.f;
                if (gr) {
                    int gbase = ((n * H_out + h) * W_out + w) * 2;
                    gx = gr[gbase + 0];
                    gy = gr[gbase + 1];
                }
                float px = (gx + 1.0f) * 0.5f * (W_in - 1);
                float py = (gy + 1.0f) * 0.5f * (H_in - 1);
                px = std::max(0.0f, std::min(static_cast<float>(W_in - 1), px));
                py = std::max(0.0f, std::min(static_cast<float>(H_in - 1), py));
                int x0 = static_cast<int>(px), x1 = std::min(x0 + 1, W_in - 1);
                int y0 = static_cast<int>(py), y1 = std::min(y0 + 1, H_in - 1);
                float wx = px - x0, wy = py - y0;

                float dgx_sum = 0.f, dgy_sum = 0.f;

                for (int c = 0; c < C; ++c) {
                    float* dxnc = dxf + (n * C + c) * H_in * W_in;
                    float  dy_val = dyf[((n * C + c) * H_out + h) * W_out + w] * alpha;

                    // Scatter dx
                    dxnc[y0 * W_in + x0] += dy_val * (1 - wy) * (1 - wx);
                    dxnc[y0 * W_in + x1] += dy_val * (1 - wy) * wx;
                    dxnc[y1 * W_in + x0] += dy_val * wy       * (1 - wx);
                    dxnc[y1 * W_in + x1] += dy_val * wy       * wx;

                    // dgrid accumulation
                    if (dgr) {
                        const float* xnc = xf + (n * C + c) * H_in * W_in;
                        // d(bilinear)/d(px): sum over four corners
                        float dpx = dy_val * (
                            xnc[y0 * W_in + x1] * (1 - wy) - xnc[y0 * W_in + x0] * (1 - wy)
                          + xnc[y1 * W_in + x1] * wy       - xnc[y1 * W_in + x0] * wy);
                        float dpy = dy_val * (
                            xnc[y1 * W_in + x0] * (1 - wx) + xnc[y1 * W_in + x1] * wx
                          - xnc[y0 * W_in + x0] * (1 - wx) - xnc[y0 * W_in + x1] * wx);
                        // Chain rule: d(gx)/d(px) = 0.5*(W_in-1), d(gy)/d(py) = 0.5*(H_in-1)
                        dgx_sum += dpx * 0.5f * (W_in - 1);
                        dgy_sum += dpy * 0.5f * (H_in - 1);
                    }
                }

                if (dgr) {
                    int gbase = ((n * H_out + h) * W_out + w) * 2;
                    dgr[gbase + 0] += alphaDgrid * dgx_sum;
                    dgr[gbase + 1] += alphaDgrid * dgy_sum;
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
