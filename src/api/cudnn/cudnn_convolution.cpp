// cuDNN convolution: forward, backward-data, backward-filter

#include "cudnn_internal.h"

#include "vgre/common/openmp_helper.h"
#include "vgre/runtime/vector_engine.h"

// ── INT8 native helpers ────────────────────────────────────────────────────

// Transpose int8 weight matrix from K_out×K_in → K_in×K_out so it can be
// passed as the K×N "B" argument of VectorEngine::matMulInt8.
static void transpose_w_int8(const int8_t* w, int8_t* wT, int K_out, int K_in) {
    for (int k = 0; k < K_out; ++k)
        for (int c = 0; c < K_in; ++c)
            wT[c * K_out + k] = w[k * K_in + c];
}

// im2col for INT8 input: converts (N,C,H,W) input into a
// (N*outH*outW) × (C*KH*KW) column matrix using zero-padding.
static void im2col_int8(const int8_t* src, int8_t* col,
                        int N, int C, int H, int W,
                        int KH, int KW,
                        int padH, int padW, int strH, int strW,
                        int outH, int outW) {
    const int K_inner = C * KH * KW;
    for (int n = 0; n < N; ++n)
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        int rowIdx = (n * outH + oh) * outW + ow;
        for (int c = 0; c < C; ++c)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            int colIdx = (c * KH + kh) * KW + kw;
            int ih = oh * strH + kh - padH;
            int iw = ow * strW + kw - padW;
            int8_t val = 0;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                val = src[((n * C + c) * H + ih) * W + iw];
            col[rowIdx * K_inner + colIdx] = val;
        }
    }
}

// Algorithm selection
static cudnnConvolutionFwdAlgo_t selectConvFwdAlgo(
    const TensorDesc* x, const FilterDesc* f, const ConvDesc* cv)
{
    if (!x || !f || !cv) return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    if (f->r == 1 && f->s == 1 && cv->str_h == 1 && cv->str_w == 1)
        return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    bool winograd = (f->r == 3 && f->s == 3)
                 && (cv->str_h == 1 && cv->str_w == 1)
                 && (cv->dil_h == 1 && cv->dil_w == 1)
                 && (cv->pad_h <= 1 && cv->pad_w <= 1)
                 && (x->h % 2 == 0 && x->w % 2 == 0)
                 && (x->c % 8 == 0 && f->k % 8 == 0);
    if (winograd) return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
}

extern "C" {

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc, int algo, size_t* size)
{
    if (size) *size = 0;
    (void)xDesc; (void)wDesc; (void)convDesc; (void)yDesc; (void)algo;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    int requestedCount, int* returnedCount,
    cudnnConvolutionFwdAlgoPerf_t* results)
{
    if (!returnedCount || !results) return CUDNN_STATUS_INVALID_VALUE;
    auto* x = (TensorDesc*)xDesc; auto* f = (FilterDesc*)wDesc;
    auto* cv = (ConvDesc*)convDesc;
    cudnnConvolutionFwdAlgo_t best = selectConvFwdAlgo(x, f, cv);
    *returnedCount = 1;
    results[0] = {best, CUDNN_STATUS_SUCCESS, 0.0f, 0, 0, {0,0,0}};
    (void)handle; (void)yDesc; (void)requestedCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    int requestedCount, int* returnedCount,
    cudnnConvolutionFwdAlgoPerf_t* results)
{
    return cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc,
        convDesc, yDesc, requestedCount, returnedCount, results);
}

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    int algo,
    void* /*workspace*/, size_t /*wsSize*/,
    const void* beta,
    cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !wDesc || !convDesc || !yDesc || !x || !w || !y || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xt=(TensorDesc*)xDesc; auto* ft=(FilterDesc*)wDesc; auto* cv=(ConvDesc*)convDesc;
    auto* yt=(TensorDesc*)yDesc;
    // Timer lives until function returns — destructor records AEE + profiler event.
    size_t _flops = 2ULL * static_cast<size_t>(xt->n) * ft->k * xt->c *
                    static_cast<size_t>(yt->h) * yt->w * ft->r * ft->s;
    size_t _bytes = sizeof(float) * (static_cast<size_t>(xt->n) * xt->c * xt->h * xt->w +
                                     static_cast<size_t>(ft->k) * ft->c * ft->r * ft->s +
                                     static_cast<size_t>(yt->n) * yt->c * yt->h * yt->w);
    VgreDnnTimed _convTimer("cudnn::conv_fwd", _flops, _bytes);
    float a = *(const float*)alpha, b = *(const float*)beta;

    bool isInt8 = (xt->dtype == CUDNN_DATA_INT8 || xt->dtype == CUDNN_DATA_INT8x4 ||
                   xt->dtype == CUDNN_DATA_INT8x32);
    int ySize = yt->n * yt->c * yt->h * yt->w;

    // ── Native INT8 path: AVX-VNNI / AMX accelerated (no FP32 round-trip) ────
    // Replaces the old dequant→FP32→requant path for both 1×1 and general convs.
    if (isInt8) {
        const int8_t* xi = static_cast<const int8_t*>(x);
        const int8_t* wi = static_cast<const int8_t*>(w);
        int8_t*       yi = static_cast<int8_t*>(y);

        const float xScale  = (a > 1e-8f) ? a : (1.f / 128.f);
        const float wScale  = (b > 1e-8f) ? b : (1.f / 128.f);
        // Output: real_value = int32_result * xScale * wScale; requantize with outScale=xScale
        const float combScale = wScale;  // xScale / outScale cancels (outScale == xScale)

        const int M       = xt->n * yt->h * yt->w;   // output pixels
        const int N_out   = ft->k;                    // output channels
        const int K_inner = xt->c * ft->r * ft->s;   // inner dimension

        // Transpose weights: K_out×K_inner → K_inner×K_out (required by matMulInt8 B shape)
        std::vector<int8_t> wT(static_cast<size_t>(K_inner) * N_out);
        transpose_w_int8(wi, wT.data(), N_out, K_inner);

        std::vector<int32_t> out_i32(static_cast<size_t>(M) * N_out, 0);

        if (ft->r == 1 && ft->s == 1) {
            // 1×1 conv: input is already in (N*H*W)×C layout when read row-by-row
            // x[n][c][h][w] → reshaped as x_mat[(n*H+h)*W+w][c] = M×C_in
            // For CHW layout this means reading C_in elements apart (stride C) — need contiguous copy
            std::vector<int8_t> xMat(static_cast<size_t>(M) * xt->c);
            for (int n = 0; n < xt->n; ++n)
            for (int h = 0; h < yt->h; ++h)
            for (int w_ = 0; w_ < yt->w; ++w_) {
                int row = (n * yt->h + h) * yt->w + w_;
                int ih = h * cv->str_h - cv->pad_h;
                int iw = w_ * cv->str_w - cv->pad_w;
                if (ih >= 0 && ih < xt->h && iw >= 0 && iw < xt->w) {
                    for (int c = 0; c < xt->c; ++c)
                        xMat[row * xt->c + c] = xi[((n * xt->c + c) * xt->h + ih) * xt->w + iw];
                } // else: stays 0 (already zeroed)
            }
            vgre::runtime::VectorEngine::instance().matMulInt8(
                xMat.data(), wT.data(), out_i32.data(), M, N_out, xt->c);
        } else {
            // General conv: im2col then INT8 GEMM
            std::vector<int8_t> col(static_cast<size_t>(M) * K_inner, 0);
            im2col_int8(xi, col.data(),
                        xt->n, xt->c, xt->h, xt->w,
                        ft->r, ft->s,
                        cv->pad_h, cv->pad_w, cv->str_h, cv->str_w,
                        yt->h, yt->w);
            vgre::runtime::VectorEngine::instance().matMulInt8(
                col.data(), wT.data(), out_i32.data(), M, N_out, K_inner);
        }

        // Scale int32 → int8 output: out[n][k][h][w] = clamp(round(acc * combScale))
        for (int n = 0; n < xt->n; ++n)
        for (int k = 0; k < N_out; ++k)
        for (int h = 0; h < yt->h; ++h)
        for (int w_ = 0; w_ < yt->w; ++w_) {
            int srcIdx = ((n * yt->h + h) * yt->w + w_) * N_out + k;
            int dstIdx = ((n * N_out + k) * yt->h + h) * yt->w + w_;
            yi[dstIdx] = vgre_quant_f32_to_i8(
                static_cast<float>(out_i32[srcIdx]) * combScale, 1.f);
        }
        return CUDNN_STATUS_SUCCESS;
    }

    // ── FP32 / FP16 path (unchanged) ──────────────────────────────────────────
    const float* xPtr = static_cast<const float*>(x);
    const float* wPtr = static_cast<const float*>(w);

    float* yf = static_cast<float*>(y);
    if (b != 0.0f) {
        for (int i = 0; i < ySize; ++i) yf[i] *= b;
    } else {
        memset(yf, 0, ySize * sizeof(float));
    }

    std::vector<float> tmp(ySize, 0.f);

    if ((algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM
         || algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
        && ft->r == 1 && ft->s == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) schedule(static) if (xt->n * ft->k * yt->h * yt->w > 1024)
        #endif
        for (int n = 0; n < xt->n; ++n)
        for (int k = 0; k < ft->k; ++k)
        for (int oh = 0; oh < yt->h; ++oh)
        for (int ow = 0; ow < yt->w; ++ow) {
            float acc = 0.f;
            int ih = oh * cv->str_h - cv->pad_h;
            int iw = ow * cv->str_w - cv->pad_w;
            if (ih >= 0 && ih < xt->h && iw >= 0 && iw < xt->w) {
                for (int c = 0; c < xt->c; ++c) {
                    acc += xPtr[((n * xt->c + c) * xt->h + ih) * xt->w + iw]
                         * wPtr[(k * ft->c + c) * ft->r * ft->s];
                }
            }
            tmp[((n * yt->c + k) * yt->h + oh) * yt->w + ow] = acc;
        }
    } else {
        cpuConv2d(xt->n, xt->c, xt->h, xt->w,
                  ft->k, ft->r, ft->s,
                  cv->pad_h, cv->pad_w, cv->str_h, cv->str_w, cv->dil_h, cv->dil_w,
                  xPtr, wPtr, tmp.data(), cv->groupCount);
    }
    for (int i = 0; i < ySize; ++i) yf[i] += a * tmp[i];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t,
    cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
    cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t,
    cudnnConvolutionBwdDataAlgo_t, size_t* size)
{
    if (size) *size = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle,
    cudnnFilterDescriptor_t wDesc,
    cudnnTensorDescriptor_t dyDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t dxDesc,
    int requestedCount, int* returnedCount,
    cudnnConvolutionBwdDataAlgoPerf_t* results)
{
    if (!returnedCount || !results) return CUDNN_STATUS_INVALID_VALUE;
    *returnedCount = 1;
    results[0] = {CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, CUDNN_STATUS_SUCCESS, 0.0f, 0, 0, {0,0,0}};
    (void)handle; (void)wDesc; (void)dyDesc; (void)convDesc; (void)dxDesc; (void)requestedCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t,
    const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t /*algo*/,
    void* /*workspace*/, size_t /*wsSize*/,
    const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx)
{
    if (!wDesc || !dyDesc || !convDesc || !dxDesc || !w || !dy || !dx || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* ft=(FilterDesc*)wDesc; auto* dyt=(TensorDesc*)dyDesc;
    auto* cv=(ConvDesc*)convDesc; auto* dxt=(TensorDesc*)dxDesc;
    float a = *(const float*)alpha, b = *(const float*)beta;

    int dxSize = dxt->n * dxt->c * dxt->h * dxt->w;
    float* dxf = (float*)dx;
    if (b != 0.0f) {
        for (int i = 0; i < dxSize; ++i) dxf[i] *= b;
    } else {
        memset(dxf, 0, dxSize * sizeof(float));
    }

    std::vector<float> tmp(dxSize, 0.f);

    if (ft->r == 1 && ft->s == 1 && cv->str_h == 1 && cv->str_w == 1
        && cv->pad_h == 0 && cv->pad_w == 0 && cv->dil_h == 1 && cv->dil_w == 1) {
        const float* dyf = (const float*)dy;
        const float* wf = (const float*)w;
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) schedule(static) if (dxt->n * dxt->c * dxt->h * dxt->w > 1024)
        #endif
        for (int n = 0; n < dxt->n; ++n)
        for (int c = 0; c < dxt->c; ++c)
        for (int h = 0; h < dxt->h; ++h)
        for (int w_ = 0; w_ < dxt->w; ++w_) {
            float acc = 0.f;
            for (int k = 0; k < ft->k; ++k)
                acc += dyf[((n * dyt->c + k) * dyt->h + h) * dyt->w + w_]
                     * wf[(k * ft->c + c) * ft->r * ft->s];
            tmp[((n * dxt->c + c) * dxt->h + h) * dxt->w + w_] = acc;
        }
    } else {
        cpuConv2dBackwardData(dxt->n, dxt->c, dxt->h, dxt->w,
                              ft->k, ft->r, ft->s,
                              cv->pad_h, cv->pad_w, cv->str_h, cv->str_w,
                              cv->dil_h, cv->dil_w,
                              (const float*)dy, (const float*)w, tmp.data(),
                              cv->groupCount);
    }

    for (int i = 0; i < dxSize; ++i) dxf[i] += a * tmp[i];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t,
    cudnnTensorDescriptor_t, cudnnTensorDescriptor_t,
    cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t,
    cudnnConvolutionBwdFilterAlgo_t, size_t* size)
{
    if (size) *size = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t dyDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t dwDesc,
    int requestedCount, int* returnedCount,
    cudnnConvolutionBwdFilterAlgoPerf_t* results)
{
    if (!returnedCount || !results) return CUDNN_STATUS_INVALID_VALUE;
    *returnedCount = 1;
    results[0] = {CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, CUDNN_STATUS_SUCCESS, 0.0f, 0, 0, {0,0,0}};
    (void)handle; (void)xDesc; (void)dyDesc; (void)convDesc; (void)dwDesc; (void)requestedCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t /*algo*/,
    void* /*workspace*/, size_t /*wsSize*/,
    const void* beta,
    cudnnFilterDescriptor_t dwDesc, void* dw)
{
    if (!xDesc || !dyDesc || !convDesc || !dwDesc || !x || !dy || !dw || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xt=(TensorDesc*)xDesc; auto* dyt=(TensorDesc*)dyDesc;
    auto* cv=(ConvDesc*)convDesc; auto* dwt=(FilterDesc*)dwDesc;
    float a = *(const float*)alpha, b = *(const float*)beta;

    int dwSize = dwt->k * dwt->c * dwt->r * dwt->s;
    float* dwf = (float*)dw;
    if (b != 0.0f) {
        for (int i = 0; i < dwSize; ++i) dwf[i] *= b;
    } else {
        memset(dwf, 0, dwSize * sizeof(float));
    }

    std::vector<float> tmp(dwSize, 0.f);

    if (dwt->r == 1 && dwt->s == 1 && cv->str_h == 1 && cv->str_w == 1
        && cv->pad_h == 0 && cv->pad_w == 0 && cv->dil_h == 1 && cv->dil_w == 1) {
        const float* xf = (const float*)x;
        const float* dyf = (const float*)dy;
        for (int k = 0; k < dwt->k; ++k)
        for (int c = 0; c < dwt->c; ++c) {
            float acc = 0.f;
            for (int n = 0; n < xt->n; ++n)
            for (int h = 0; h < xt->h; ++h)
            for (int w_ = 0; w_ < xt->w; ++w_) {
                acc += xf[((n*xt->c + c)*xt->h + h)*xt->w + w_]
                     * dyf[((n*dyt->c + k)*dyt->h + h)*dyt->w + w_];
            }
            tmp[(k*dwt->c + c)*dwt->r*dwt->s] = acc;
        }
    } else {
        cpuConv2dBackwardFilter(xt->n, xt->c, xt->h, xt->w,
                                dwt->k, dwt->r, dwt->s,
                                cv->pad_h, cv->pad_w, cv->str_h, cv->str_w,
                                cv->dil_h, cv->dil_w,
                                (const float*)x, (const float*)dy, tmp.data(),
                                cv->groupCount);
    }

    for (int i = 0; i < dwSize; ++i) dwf[i] += a * tmp[i];
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnConvolutionBiasActivationForward ─────────────────────────────────────
// Fused: y = activation(alpha1 * conv(x, w) + alpha2 * z + bias)
// z is a residual tensor (same shape as y); bias is [1,k,1,1].
cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t /*handle*/,
    const void* alpha1,
    cudnnTensorDescriptor_t xDesc,  const void* x,
    cudnnFilterDescriptor_t  wDesc,  const void* w,
    cudnnConvolutionDescriptor_t convDesc, int algo,
    void* workspace, size_t workspaceSize,
    const void* alpha2,
    cudnnTensorDescriptor_t zDesc,    const void* z,
    cudnnTensorDescriptor_t biasDesc, const void* bias,
    cudnnActivationDescriptor_t activationDesc,
    cudnnTensorDescriptor_t yDesc,    void* y)
{
    if (!xDesc || !wDesc || !convDesc || !yDesc || !x || !w || !y || !alpha1 || !alpha2)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xt  = (TensorDesc*)xDesc;
    auto* ft  = (FilterDesc*)wDesc;
    auto* cv  = (ConvDesc*)convDesc;
    auto* yt  = (TensorDesc*)yDesc;
    auto* act = activationDesc ? (ActDesc*)activationDesc : nullptr;

    // Step 1: compute conv(x, w) → tmp
    int ySize = yt->n * yt->c * yt->h * yt->w;
    std::vector<float> tmp(ySize, 0.f);

    if ((algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM
         || algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
        && ft->r == 1 && ft->s == 1)
    {
        const float* xf = (const float*)x;
        const float* wf = (const float*)w;
        #ifdef _OPENMP
        #pragma omp parallel for OMP_COLLAPSE(2) schedule(static) if (xt->n * ft->k * yt->h * yt->w > 1024)
        #endif
        for (int n = 0; n < xt->n; ++n)
        for (int k = 0; k < ft->k; ++k)
        for (int oh = 0; oh < yt->h; ++oh)
        for (int ow = 0; ow < yt->w; ++ow) {
            float acc = 0.f;
            int ih = oh * cv->str_h - cv->pad_h;
            int iw = ow * cv->str_w - cv->pad_w;
            if (ih >= 0 && ih < xt->h && iw >= 0 && iw < xt->w) {
                for (int c = 0; c < xt->c; ++c)
                    acc += xf[((n*xt->c + c)*xt->h + ih)*xt->w + iw]
                         * wf[(k*ft->c + c)*ft->r*ft->s];
            }
            tmp[((n*yt->c + k)*yt->h + oh)*yt->w + ow] = acc;
        }
    } else {
        cpuConv2d(xt->n, xt->c, xt->h, xt->w,
                  ft->k, ft->r, ft->s,
                  cv->pad_h, cv->pad_w, cv->str_h, cv->str_w, cv->dil_h, cv->dil_w,
                  (const float*)x, (const float*)w, tmp.data(), cv->groupCount);
    }

    float a1 = *(const float*)alpha1;
    float a2 = *(const float*)alpha2;
    float* yf = (float*)y;

    // Step 2: y = a1*conv + a2*z + bias, then activation
    const float* zf    = z    ? (const float*)z    : nullptr;
    const float* biasf = bias ? (const float*)bias : nullptr;

    for (int n = 0; n < yt->n; ++n)
    for (int k = 0; k < yt->c; ++k)
    for (int oh = 0; oh < yt->h; ++oh)
    for (int ow = 0; ow < yt->w; ++ow) {
        int idx = ((n*yt->c + k)*yt->h + oh)*yt->w + ow;
        float val = a1 * tmp[idx];
        if (zf)    val += a2 * zf[idx];
        if (biasf) val += biasf[k];
        yf[idx] = act ? applyActivation(val, *act) : val;
    }
    (void)workspace; (void)workspaceSize; (void)zDesc; (void)biasDesc;
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnConvolutionBackwardBias ───────────────────────────────────────────────
// Gradient of the bias: db[k] = sum over (n, h, w) of dy[n, k, h, w]

cudnnStatus_t cudnnConvolutionBackwardBias(
    cudnnHandle_t /*handle*/,
    const void* alpha,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    const void* beta,
    cudnnTensorDescriptor_t dbDesc, void* db)
{
    if (!dyDesc || !dbDesc || !dy || !db || !alpha || !beta)
        return CUDNN_STATUS_BAD_PARAM;

    auto* dyt = (TensorDesc*)dyDesc;
    auto* dbt = (TensorDesc*)dbDesc;
    if (!dyt || !dbt) return CUDNN_STATUS_BAD_PARAM;

    int n = dyt->n, k = dyt->c, h = dyt->h, w = dyt->w;
    float a = *(const float*)alpha;
    float b = *(const float*)beta;

    float* dbf = (float*)db;
    const float* dyf = (const float*)dy;

    // db[k] = beta*db[k] + alpha * sum_{n,h,w} dy[n,k,h,w]
    for (int ck = 0; ck < dbt->c; ++ck)
        dbf[ck] = b * dbf[ck];

    for (int ni = 0; ni < n; ++ni)
        for (int ck = 0; ck < k; ++ck)
            for (int hi = 0; hi < h; ++hi)
                for (int wi = 0; wi < w; ++wi)
                    dbf[ck] += a * dyf[((ni*k + ck)*h + hi)*w + wi];

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
