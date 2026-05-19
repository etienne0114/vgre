// cuDNN convolution: forward, backward-data, backward-filter

#include "cudnn_internal.h"

#include "vgre/common/openmp_helper.h"

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
    int xElem = xt->n * xt->c * xt->h * xt->w;
    int wElem = ft->k * ft->c * ft->r * ft->s;
    std::vector<float> xFloat, wFloat;
    const float* xPtr = (const float*)x;
    const float* wPtr = (const float*)w;
    if (isInt8) {
        float xScale = (a > 1e-8f) ? a : (1.f / 128.f);
        float wScale = (b > 1e-8f) ? b : (1.f / 128.f);
        xFloat.resize(xElem);
        wFloat.resize(wElem);
        const int8_t* xi = (const int8_t*)x;
        const int8_t* wi = (const int8_t*)w;
        for (int i = 0; i < xElem; ++i) xFloat[i] = vgre_dequant_i8(xi[i], xScale);
        for (int i = 0; i < wElem; ++i) wFloat[i] = vgre_dequant_i8(wi[i], wScale);
        xPtr = xFloat.data();
        wPtr = wFloat.data();
        a = 1.f; b = 0.f;
    }

    int ySize = yt->n * yt->c * yt->h * yt->w;
    float* yf = (float*)y;
    if (!isInt8) {
        if (b != 0.0f) {
            for (int i = 0; i < ySize; ++i) yf[i] *= b;
        } else {
            std::memset(yf, 0, ySize * sizeof(float));
        }
    }

    std::vector<float> tmp(ySize, 0.f);

    if ((algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM
         || algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
        && ft->r == 1 && ft->s == 1) {
        const float* xf = xPtr;
        const float* wf = wPtr;
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
                    acc += xf[((n * xt->c + c) * xt->h + ih) * xt->w + iw]
                         * wf[(k * ft->c + c) * ft->r * ft->s];
                }
            }
            tmp[((n * yt->c + k) * yt->h + oh) * yt->w + ow] = acc;
        }
    } else {
        cpuConv2d(xt->n, xt->c, xt->h, xt->w,
                  ft->k, ft->r, ft->s,
                  cv->pad_h, cv->pad_w, cv->str_h, cv->str_w, cv->dil_h, cv->dil_w,
                  xPtr, wPtr, tmp.data());
    }
    if (isInt8) {
        float outScale = (a > 1e-8f) ? a : (1.f / 128.f);
        float invOutScale = 1.f / outScale;
        int8_t* yi = (int8_t*)y;
        for (int i = 0; i < ySize; ++i)
            yi[i] = vgre_quant_f32_to_i8(tmp[i], invOutScale);
    } else {
        for (int i = 0; i < ySize; ++i) yf[i] += a * tmp[i];
    }
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
        std::memset(dxf, 0, dxSize * sizeof(float));
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
                              (const float*)dy, (const float*)w, tmp.data());
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
        std::memset(dwf, 0, dwSize * sizeof(float));
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
                                (const float*)x, (const float*)dy, tmp.data());
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
                  (const float*)x, (const float*)w, tmp.data());
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
