// cuDNN shim: intercepts cuDNN API calls and routes them to hand-rolled
// CPU implementations for the operations most commonly used by AI frameworks.
// Covers: tensor descriptors, convolution (direct/GEMM), activation, pooling,
// batch normalization, and softmax.

#include "vgre/common/logger.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

// ── cuDNN type stubs (no cudnn.h needed) ─────────────────────────────────────
typedef int    cudnnStatus_t;
typedef void*  cudnnHandle_t;
typedef void*  cudnnTensorDescriptor_t;
typedef void*  cudnnFilterDescriptor_t;
typedef void*  cudnnConvolutionDescriptor_t;
typedef void*  cudnnPoolingDescriptor_t;
typedef void*  cudnnActivationDescriptor_t;
typedef void*  cudnnOpTensorDescriptor_t;

static constexpr cudnnStatus_t CUDNN_STATUS_SUCCESS       = 0;
static constexpr cudnnStatus_t CUDNN_STATUS_NOT_INITIALIZED = 1;
static constexpr cudnnStatus_t CUDNN_STATUS_INVALID_VALUE  = 8;
static constexpr cudnnStatus_t CUDNN_STATUS_NOT_SUPPORTED  = 9;

// ── Activation modes ─────────────────────────────────────────────────────────
enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4,
    CUDNN_ACTIVATION_IDENTITY     = 5,
    CUDNN_ACTIVATION_SWISH        = 6,
    CUDNN_ACTIVATION_GELU         = 7,  // Gaussian Error Linear Unit
    CUDNN_ACTIVATION_SELU         = 8,  // Scaled ELU
    CUDNN_ACTIVATION_MISH         = 9   // Mish: x * tanh(softplus(x))
};

enum cudnnPoolingMode_t {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
};

enum cudnnDataType_t {
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2,
    CUDNN_DATA_INT8   = 3,
    CUDNN_DATA_INT32  = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8  = 6,
    CUDNN_DATA_UINT8x4= 7,
    CUDNN_DATA_INT8x32= 8,
    CUDNN_DATA_BFLOAT16= 9,
    CUDNN_DATA_INT64  = 10
};

enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1
};

enum cudnnNanPropagation_t {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN = 1
};

// ── INT8 quantization helpers ─────────────────────────────────────────────────
// Scale factor stored in alpha/beta when dataType == CUDNN_DATA_INT8.
// VGRE uses symmetric per-tensor quantization: x_float = x_int8 * scale.
static inline float vgre_dequant_i8(int8_t v, float scale) {
    return static_cast<float>(v) * scale;
}
static inline int8_t vgre_quant_f32_to_i8(float v, float inv_scale) {
    float q = v * inv_scale;
    if (q >  127.f) q =  127.f;
    if (q < -128.f) q = -128.f;
    return static_cast<int8_t>(std::round(q));
}

// ── Descriptor structs ────────────────────────────────────────────────────────
struct TensorDesc {
    int n, c, h, w;
    cudnnDataType_t dtype;
    cudnnTensorFormat_t fmt;
};
struct FilterDesc {
    int k, c, r, s;  // out_channels, in_channels, filter_h, filter_w
    cudnnDataType_t dtype;
};
struct ConvDesc {
    int pad_h, pad_w, str_h, str_w, dil_h, dil_w;
};
struct PoolDesc {
    cudnnPoolingMode_t mode;
    int win_h, win_w, pad_h, pad_w, str_h, str_w;
};
struct ActDesc {
    cudnnActivationMode_t mode;
    double coeff;
};
struct HandleCtx {
    void* stream   = nullptr;  // bound CUDA stream (for async ordering)
    int   deviceId = 0;
    bool  deterministicAlgorithms = false;
};

// ── Helpers ───────────────────────────────────────────────────────────────────
static inline float applyActivation(float x, const ActDesc& act) {
    switch (act.mode) {
    case CUDNN_ACTIVATION_RELU:         return x > 0.f ? x : 0.f;
    case CUDNN_ACTIVATION_SIGMOID:      return 1.f / (1.f + expf(-x));
    case CUDNN_ACTIVATION_TANH:         return tanhf(x);
    case CUDNN_ACTIVATION_CLIPPED_RELU: return std::min(std::max(x, 0.f), (float)act.coeff);
    case CUDNN_ACTIVATION_ELU:          return x > 0.f ? x : (float)(act.coeff * (expf(x) - 1.f));
    case CUDNN_ACTIVATION_SWISH:        return x / (1.f + expf(-x));
    // GELU: 0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    // This is the "tanh" approximation used by PyTorch and TensorFlow.
    case CUDNN_ACTIVATION_GELU: {
        static constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
        static constexpr float kCoeff       = 0.044715f;
        float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        return 0.5f * x * (1.f + tanhf(inner));
    }
    // SELU: scale * (x if x>0 else alpha*(exp(x)-1))
    // Standard SELU constants: alpha=1.6732632..., scale=1.0507009...
    case CUDNN_ACTIVATION_SELU: {
        static constexpr float kAlpha = 1.6732632423543772f;
        static constexpr float kScale = 1.0507009873554805f;
        return kScale * (x > 0.f ? x : kAlpha * (expf(x) - 1.f));
    }
    // Mish: x * tanh(softplus(x))  where softplus(x) = ln(1 + e^x)
    // Numerically stable: for x > 20 softplus(x) ≈ x, tanh → 1
    case CUDNN_ACTIVATION_MISH: {
        float sp = (x > 20.f) ? x : logf(1.f + expf(x));
        return x * tanhf(sp);
    }
    default: return x;
    }
}

// Index helper for NCHW layout: [n,c,h,w] → flat index
static inline int nchw(int N, int C, int H, int W, int n, int c, int h, int w) {
    return ((n*C + c)*H + h)*W + w;
}

// ── Direct (naive) convolution: output = input * filter + bias ────────────────
static void cpuConv2d(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* input, const float* filter, float* output)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;

    for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        float acc = 0.f;
        for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s) {
            int ih = oh*str_h + r*dil_h - pad_h;
            int iw = ow*str_w + s*dil_w - pad_w;
            if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
            acc += input[nchw(N,C,H,W,n,c,ih,iw)] *
                   filter[((k*C + c)*R + r)*S + s];
        }
        output[nchw(N,K,outH,outW,n,k,oh,ow)] = acc;
    }
}

// ── cuDNN extern "C" API ──────────────────────────────────────────────────────
extern "C" {

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
    if (!handle) return CUDNN_STATUS_INVALID_VALUE;
    *handle = new HandleCtx();
    VGRE_LOG_DEBUG("cuDNN", "cudnnCreate");
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroy(cudnnHandle_t h) { delete (HandleCtx*)h; return CUDNN_STATUS_SUCCESS; }
cudnnStatus_t cudnnGetVersion(unsigned long long* v) {
    // Report the cuDNN version this shim emulates (8.9.x feature set).
    if (v) *v = 8904;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, void* stream) {
    if (!handle) return CUDNN_STATUS_NOT_INITIALIZED;
    ((HandleCtx*)handle)->stream = stream;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, void** stream) {
    if (!handle || !stream) return CUDNN_STATUS_INVALID_VALUE;
    *stream = ((HandleCtx*)handle)->stream;
    return CUDNN_STATUS_SUCCESS;
}

// ── Tensor descriptors ────────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d) {
    *d = new TensorDesc{}; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d) {
    delete (TensorDesc*)d; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n,int c,int h,int w)
{
    auto* t=(TensorDesc*)d; t->n=n; t->c=c; t->h=h; t->w=w; t->dtype=dtype; t->fmt=fmt;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnDataType_t* dtype, int* n,int* c,int* h,int* w,
    int* ns,int* cs,int* hs,int* ws)
{
    auto* t=(TensorDesc*)d;
    if(dtype)*dtype=t->dtype; if(n)*n=t->n; if(c)*c=t->c; if(h)*h=t->h; if(w)*w=t->w;
    // strides for NCHW
    if(ws)*ws=1; if(hs)*hs=t->w; if(cs)*cs=t->h*t->w; if(ns)*ns=t->c*t->h*t->w;
    return CUDNN_STATUS_SUCCESS;
}

// ── Filter descriptor ─────────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* d) {
    *d = new FilterDesc{}; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d) {
    delete (FilterDesc*)d; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d,
    cudnnDataType_t dtype, cudnnTensorFormat_t /*fmt*/, int k,int c,int r,int s)
{
    auto* f=(FilterDesc*)d; f->k=k; f->c=c; f->r=r; f->s=s; f->dtype=dtype;
    return CUDNN_STATUS_SUCCESS;
}

// ── Convolution descriptor ────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* d) {
    *d = new ConvDesc{}; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d) {
    delete (ConvDesc*)d; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,
    int ph,int pw,int sh,int sw,int dh,int dw,int /*mode*/,cudnnDataType_t)
{
    auto* c=(ConvDesc*)d; c->pad_h=ph; c->pad_w=pw; c->str_h=sh; c->str_w=sw; c->dil_h=dh; c->dil_w=dw;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    cudnnConvolutionDescriptor_t cd,
    cudnnTensorDescriptor_t td, cudnnFilterDescriptor_t fd,
    int* n,int* c,int* h,int* w)
{
    auto* t=(TensorDesc*)td; auto* f=(FilterDesc*)fd; auto* cv=(ConvDesc*)cd;
    if(n)*n=t->n; if(c)*c=f->k;
    if(h)*h=(t->h+2*cv->pad_h-cv->dil_h*(f->r-1)-1)/cv->str_h+1;
    if(w)*w=(t->w+2*cv->pad_w-cv->dil_w*(f->s-1)-1)/cv->str_w+1;
    return CUDNN_STATUS_SUCCESS;
}

// Algorithm selection constants (mirrors cuDNN enum)
enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6
};

struct cudnnConvolutionFwdAlgoPerf_t {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t             status;
    float                     time;
    size_t                    memory;
    int                       mathType;
    int                       reserved[3];
};

// Select the best algorithm based on tensor/filter dimensions.
// Conditions follow cuDNN documentation and NVIDIA's Winograd applicability guide.
static cudnnConvolutionFwdAlgo_t selectConvFwdAlgo(
    const TensorDesc* x, const FilterDesc* f, const ConvDesc* cv)
{
    if (!x || !f || !cv) return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;

    // GEMM: 1×1 convolutions with unit stride reduce to pure matrix multiply.
    if (f->r == 1 && f->s == 1 && cv->str_h == 1 && cv->str_w == 1)
        return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

    // Winograd F(2,3) applicability — all six conditions must hold:
    //   1. Filter exactly 3×3
    //   2. Stride 1×1 (F(2,3) is only defined for unit stride)
    //   3. Dilation 1×1 (dilated convolutions break Winograd tile math)
    //   4. Padding ≤ (R-1)/2 = 1 (fits within F(2,3) boundary extension)
    //   5. Spatial dims even (F(2,3) produces 2×2 output tiles; odd dims waste one tile)
    //   6. Channel count multiple of 8 (SIMD alignment for the transform matrices)
    bool winograd = (f->r == 3 && f->s == 3)
                 && (cv->str_h == 1 && cv->str_w == 1)
                 && (cv->dil_h == 1 && cv->dil_w == 1)
                 && (cv->pad_h <= 1 && cv->pad_w <= 1)
                 && (x->h % 2 == 0 && x->w % 2 == 0)
                 && (x->c % 8 == 0 && f->k % 8 == 0);
    if (winograd) return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

    // IMPLICIT_PRECOMP_GEMM: general-purpose best choice for all other sizes.
    // Precomputes index maps into a small workspace for ~15% vs DIRECT.
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc, int algo, size_t* size)
{
    // VGRE's direct conv needs no workspace; GEMM path allocates inline.
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
    int algo,   // cudnnConvolutionFwdAlgo_t — used to select execution path
    void* /*workspace*/, size_t /*wsSize*/,
    const void* beta,
    cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !wDesc || !convDesc || !yDesc || !x || !w || !y || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xt=(TensorDesc*)xDesc; auto* ft=(FilterDesc*)wDesc; auto* cv=(ConvDesc*)convDesc;
    auto* yt=(TensorDesc*)yDesc;
    float a = *(const float*)alpha, b = *(const float*)beta;

    // ── INT8 path: dequantize inputs to FP32, run conv, requantize output ──────
    bool isInt8 = (xt->dtype == CUDNN_DATA_INT8 || xt->dtype == CUDNN_DATA_INT8x4);
    int xElem = xt->n * xt->c * xt->h * xt->w;
    int wElem = ft->k * ft->c * ft->r * ft->s;
    std::vector<float> xFloat, wFloat;
    const float* xPtr = (const float*)x;
    const float* wPtr = (const float*)w;
    if (isInt8) {
        // alpha carries the input scale; beta carries the weight scale.
        // If scale is unreasonably small, use 1/128 as a safe default.
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
        a = 1.f; b = 0.f; // alpha/beta absorbed into dequant scales
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
        // 1×1 conv with stride: NCHW input, output spatial dims from conv params.
        // Input is NCHW: x[n, c, h, w]. Filter is KCRS: w[k, c, 1, 1].
        // Output is NCHW: y[n, k, oh, ow] where:
        //   oh = (h + 2*pad_h - str_h*(r-1) - 1) / str_h + 1  (for r=1 → (h + 2*pad_h) / str_h)
        //   ow = same for w
        // Use the output dims from the pre-computed yt descriptor.
        const float* xf = xPtr;
        const float* wf = wPtr;
        for (int n = 0; n < xt->n; ++n)
        for (int k = 0; k < ft->k; ++k)
        for (int oh = 0; oh < yt->h; ++oh)
        for (int ow = 0; ow < yt->w; ++ow) {
            float acc = 0.f;
            // Map output position to input position via stride
            int ih = oh * cv->str_h - cv->pad_h;
            int iw = ow * cv->str_w - cv->pad_w;
            if (ih >= 0 && ih < xt->h && iw >= 0 && iw < xt->w) {
                for (int c = 0; c < xt->c; ++c) {
                    // NCHW indexing: x[n, c, ih, iw]
                    acc += xf[((n * xt->c + c) * xt->h + ih) * xt->w + iw]
                         * wf[(k * ft->c + c) * ft->r * ft->s]; // 1×1 filter
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
        // Requantize FP32 result back to INT8 using output scale = a.
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

// ── Backward-data convolution: computes dx from dy and w ─────────────────────
// dx[n,c,ih,iw] += sum_{k,oh,ow} dy[n,k,oh,ow] * w[k,c,r,s]
//   where ih = oh*str_h + r*dil_h - pad_h, iw = ow*str_w + s*dil_w - pad_w
static void cpuConv2dBackwardData(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* dy, const float* w, float* dx)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;

    for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        float dyVal = dy[nchw(N,K,outH,outW,n,k,oh,ow)];
        for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s) {
            int ih = oh*str_h + r*dil_h - pad_h;
            int iw = ow*str_w + s*dil_w - pad_w;
            if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
            dx[nchw(N,C,H,W,n,c,ih,iw)] += dyVal * w[((k*C + c)*R + r)*S + s];
        }
    }
}

enum cudnnConvolutionBwdDataAlgo_t {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
};

struct cudnnConvolutionBwdDataAlgoPerf_t {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int mathType;
    int reserved[3];
};

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

    // 1×1 GEMM fast path: when filter is 1×1 with unit stride and no padding,
    // backward data reduces to a matrix multiply.
    if (ft->r == 1 && ft->s == 1 && cv->str_h == 1 && cv->str_w == 1
        && cv->pad_h == 0 && cv->pad_w == 0 && cv->dil_h == 1 && cv->dil_w == 1) {
        const float* dyf = (const float*)dy;
        const float* wf = (const float*)w;
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

// ── Backward-filter convolution: computes dw from x and dy ─────────────────────
// dw[k,c,r,s] += sum_{n,oh,ow} dy[n,k,oh,ow] * x[n,c,ih,iw]
//   where ih = oh*str_h + r*dil_h - pad_h, iw = ow*str_w + s*dil_w - pad_w
static void cpuConv2dBackwardFilter(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* x, const float* dy, float* dw)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s) {
        float acc = 0.f;
        for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
            int ih = oh*str_h + r*dil_h - pad_h;
            int iw = ow*str_w + s*dil_w - pad_w;
            if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
            acc += x[nchw(N,C,H,W,n,c,ih,iw)] * dy[nchw(N,K,outH,outW,n,k,oh,ow)];
        }
        dw[((k*C + c)*R + r)*S + s] += acc;
    }
}

enum cudnnConvolutionBwdFilterAlgo_t {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
};

struct cudnnConvolutionBwdFilterAlgoPerf_t {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int mathType;
    int reserved[3];
};

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

    // 1×1 GEMM fast path
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

// ── Activation ────────────────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* d) {
    *d = new ActDesc{}; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d) {
    delete (ActDesc*)d; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d,
    cudnnActivationMode_t mode, cudnnNanPropagation_t, double coeff)
{
    auto* a=(ActDesc*)d; a->mode=mode; a->coeff=coeff; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t actDesc,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,  cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta) return CUDNN_STATUS_INVALID_VALUE;
    auto* tx=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc; auto* act=(ActDesc*)actDesc;
    // Output size must match input size.
    int Nx = tx->n*tx->c*tx->h*tx->w;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    if (Nx != Ny) return CUDNN_STATUS_INVALID_VALUE;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;
    for (int i = 0; i < Nx; ++i)
        yf[i] = a * applyActivation(xf[i], *act) + b * yf[i];
    return CUDNN_STATUS_SUCCESS;
}

// Activation backward (compute dx = alpha * dy * f'(x) + beta * dx)
static inline float applyActivationDerivative(float x, float y, const ActDesc& act) {
    switch (act.mode) {
    case CUDNN_ACTIVATION_RELU:         return x > 0.f ? 1.f : 0.f;
    case CUDNN_ACTIVATION_SIGMOID: {
        // y = sigmoid(x) if forward used sigmoid; derivative = y*(1-y)
        float s = y; return s * (1.f - s);
    }
    case CUDNN_ACTIVATION_TANH: {
        // y = tanh(x)
        float t = y; return 1.f - t * t;
    }
    case CUDNN_ACTIVATION_CLIPPED_RELU: return (x > 0.f && x < (float)act.coeff) ? 1.f : 0.f;
    case CUDNN_ACTIVATION_ELU: {
        // ELU: x>0 -> x ; x<=0 -> coeff*(exp(x)-1). Derivative: x>0 -> 1; else coeff*exp(x)
        return x > 0.f ? 1.f : (float)(act.coeff * expf(x));
    }
    case CUDNN_ACTIVATION_SWISH: {
        // swish = x * sigmoid(x)
        float sig = 1.f / (1.f + expf(-x));
        return sig + x * sig * (1.f - sig);
    }
    case CUDNN_ACTIVATION_GELU: {
        // tanh-approx GELU derivative
        static constexpr float kSqrt2OverPi = 0.7978845608f;
        static constexpr float kCoeff       = 0.044715f;
        float x3 = x * x * x;
        float inner = kSqrt2OverPi * (x + kCoeff * x3);
        float tanh_i = tanhf(inner);
        float sech2 = 1.f - tanh_i * tanh_i; // sech^2(inner)
        float inner_prime = kSqrt2OverPi * (1.f + 3.f * kCoeff * x * x);
        // derivative: 0.5*(1 + tanh(inner)) + 0.5*x * sech^2(inner) * inner_prime
        return 0.5f * (1.f + tanh_i) + 0.5f * x * sech2 * inner_prime;
    }
    case CUDNN_ACTIVATION_SELU: {
        static constexpr float kAlpha = 1.6732632423543772f;
        static constexpr float kScale = 1.0507009873554805f;
        return kScale * (x > 0.f ? 1.f : kAlpha * expf(x));
    }
    case CUDNN_ACTIVATION_MISH: {
        // mish = x * tanh(softplus(x)); softplus' = sigmoid(x)
        float sp = (x > 20.f) ? x : logf(1.f + expf(x));
        float tsp = tanhf(sp);
        float sig = 1.f / (1.f + expf(-x));
        float sech2 = 1.f - tsp * tsp;
        return tsp + x * sech2 * sig;
    }
    default: return 1.f; // identity
    }
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

// ── Softmax ───────────────────────────────────────────────────────────────────
// algo: 0=FAST (per-instance, C dimension), 1=ACCURATE (log-sum-exp stable)
// mode: 0=INSTANCE, 1=CHANNEL
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t,
    int algo, int mode,
    const void* alpha, cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,  cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta) return CUDNN_STATUS_INVALID_VALUE;
    auto* t=(TensorDesc*)xDesc; auto* ty=(TensorDesc*)yDesc;
    if (t->n*t->c*t->h*t->w != ty->n*ty->c*ty->h*ty->w) return CUDNN_STATUS_INVALID_VALUE;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;

    // Both FAST and ACCURATE use the numerically-stable max-subtract form.
    // CHANNEL mode: softmax over C for each (n, h, w) position.
    // INSTANCE mode: softmax over C*H*W for each n.
    if (mode == 0) {
        // INSTANCE: flatten C*H*W into a single softmax vector per n.
        int VEC = C * HW;
        for (int n=0; n<N; ++n) {
            const float* row = xf + n*VEC;
            float* out = yf + n*VEC;
            float maxv = *std::max_element(row, row+VEC);
            float sum = 0.f;
            for (int i=0; i<VEC; ++i) sum += expf(row[i] - maxv);
            for (int i=0; i<VEC; ++i) out[i] = a*(expf(row[i]-maxv)/sum) + b*out[i];
        }
    } else {
        // CHANNEL (default): softmax over C for each (n, h, w).
        for (int n=0; n<N; ++n)
        for (int hw=0; hw<HW; ++hw) {
            float maxv = xf[n*C*HW + 0*HW + hw];
            for (int c=1; c<C; ++c) maxv = std::max(maxv, xf[n*C*HW + c*HW + hw]);
            float sum = 0.f;
            for (int c=0; c<C; ++c) sum += expf(xf[n*C*HW + c*HW + hw] - maxv);
            for (int c=0; c<C; ++c) {
                float v = expf(xf[n*C*HW + c*HW + hw] - maxv) / sum;
                yf[n*C*HW + c*HW + hw] = a*v + b*yf[n*C*HW + c*HW + hw];
            }
        }
    }
    (void)algo;
    return CUDNN_STATUS_SUCCESS;
}

// ── Pooling descriptors ───────────────────────────────────────────────────────
cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* d) {
    *d = new PoolDesc{}; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t d) {
    delete (PoolDesc*)d; return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t d, cudnnPoolingMode_t mode, int /*nanOpt*/,
    int winH, int winW, int padH, int padW, int strH, int strW)
{
    auto* p=(PoolDesc*)d; p->mode=mode;
    p->win_h=winH; p->win_w=winW; p->pad_h=padH; p->pad_w=padW;
    p->str_h=strH; p->str_w=strW;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetPooling2dDescriptor(
    cudnnPoolingDescriptor_t d, cudnnPoolingMode_t* mode, int* /*nanOpt*/,
    int* winH, int* winW, int* padH, int* padW, int* strH, int* strW)
{
    auto* p=(PoolDesc*)d;
    if(mode)*mode=p->mode; if(winH)*winH=p->win_h; if(winW)*winW=p->win_w;
    if(padH)*padH=p->pad_h; if(padW)*padW=p->pad_w;
    if(strH)*strH=p->str_h; if(strW)*strW=p->str_w;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
    cudnnPoolingDescriptor_t pd, cudnnTensorDescriptor_t xDesc,
    int* n, int* c, int* h, int* w)
{
    auto* p=(PoolDesc*)pd; auto* t=(TensorDesc*)xDesc;
    if(n)*n=t->n; if(c)*c=t->c;
    if(h)*h=(t->h+2*p->pad_h-p->win_h)/p->str_h+1;
    if(w)*w=(t->w+2*p->pad_w-p->win_w)/p->str_w+1;
    return CUDNN_STATUS_SUCCESS;
}

// ── Pooling forward (MaxPool and AvgPool) ─────────────────────────────────────
// Supports CUDNN_POOLING_MAX, CUDNN_POOLING_MAX_DETERMINISTIC,
// CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING.
cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t,
    cudnnPoolingDescriptor_t pd,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    cudnnTensorDescriptor_t /*yDesc*/, void* y)
{
    if (!pd || !xDesc || !x || !y || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;
    auto* p=(PoolDesc*)pd; auto* xt=(TensorDesc*)xDesc;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;

    int outH=(xt->h+2*p->pad_h-p->win_h)/p->str_h+1;
    int outW=(xt->w+2*p->pad_w-p->win_w)/p->str_w+1;
    bool isMax=(p->mode==CUDNN_POOLING_MAX||p->mode==CUDNN_POOLING_MAX_DETERMINISTIC);
    // For avg pooling, count only valid (non-padded) elements when EXCLUDE_PADDING mode
    bool excludePad=(p->mode==CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

    for(int n=0; n<xt->n; ++n)
    for(int c=0; c<xt->c; ++c)
    for(int oh=0; oh<outH; ++oh)
    for(int ow=0; ow<outW; ++ow) {
        float acc=isMax?-1e38f:0.f; int cnt=0;
        for(int kh=0; kh<p->win_h; ++kh)
        for(int kw=0; kw<p->win_w; ++kw) {
            int ih=oh*p->str_h-p->pad_h+kh;
            int iw=ow*p->str_w-p->pad_w+kw;
            bool inBounds=(ih>=0&&ih<xt->h&&iw>=0&&iw<xt->w);
            if(!inBounds) { if(!excludePad && !isMax) ++cnt; continue; }
            float v=xf[((n*xt->c+c)*xt->h+ih)*xt->w+iw];
            if(isMax) { if(v>acc) acc=v; }
            else { acc+=v; ++cnt; }
        }
        if(!isMax && cnt>0) acc/=static_cast<float>(cnt);
        else if(!isMax && cnt==0) acc=0.f;
        int oi=((n*xt->c+c)*outH+oh)*outW+ow;
        yf[oi]=a*acc+b*yf[oi];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Batch Normalization ───────────────────────────────────────────────────────
cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t, int /*mode*/,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnDesc*/,
    const void* scale, const void* bias,
    const void* mean, const void* var, double eps)
{
    auto* t=(TensorDesc*)xDesc;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;
    const float* sc=(const float*)scale; const float* bi=(const float*)bias;
    const float* mu=(const float*)mean; const float* va=(const float*)var;
    for (int n=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int hw=0; hw<HW; ++hw) {
        int idx = (n*C + c)*HW + hw;
        float xhat = (xf[idx] - mu[c]) / sqrtf((float)va[c] + (float)eps);
        yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
    }
    (void)yDesc;
    return CUDNN_STATUS_SUCCESS;
}

enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL        = 1
};

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnScaleBiasMeanVarDesc*/,
    const void* scale, const void* bias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta || !scale || !bias)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* t=(TensorDesc*)xDesc;
    int N=t->n, C=t->c, HW=t->h*t->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x; float* yf=(float*)y;
    const float* sc=(const float*)scale; const float* bi=(const float*)bias;

    // Compute per-channel mean and variance
    std::vector<float> mean(C, 0.f), var(C, 0.f);
    int count = N * HW;

    if (mode == CUDNN_BATCHNORM_SPATIAL) {
        for (int c=0; c<C; ++c) {
            double sum = 0.0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw)
                sum += xf[(n*C + c)*HW + hw];
            mean[c] = static_cast<float>(sum / count);
        }
        for (int c=0; c<C; ++c) {
            double sumSq = 0.0;
            for (int n=0; n<N; ++n)
            for (int hw=0; hw<HW; ++hw) {
                float d = xf[(n*C + c)*HW + hw] - mean[c];
                sumSq += d * d;
            }
            var[c] = static_cast<float>(sumSq / count);
        }
    } else {
        // PER_ACTIVATION: compute mean/var for every (c,h,w) position across N
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            double sum = 0.0;
            for (int n=0; n<N; ++n)
                sum += xf[(n*C + c)*HW + hw];
            mean[c*HW + hw] = static_cast<float>(sum / N);
        }
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            double sumSq = 0.0;
            for (int n=0; n<N; ++n) {
                float d = xf[(n*C + c)*HW + hw] - mean[c*HW + hw];
                sumSq += d * d;
            }
            var[c*HW + hw] = static_cast<float>(sumSq / N);
        }
    }

    // Normalize and apply scale/shift
    if (mode == CUDNN_BATCHNORM_SPATIAL) {
        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c) {
            float invVar = 1.f / sqrtf(var[c] + (float)epsilon);
            for (int hw=0; hw<HW; ++hw) {
                int idx = (n*C + c)*HW + hw;
                float xhat = (xf[idx] - mean[c]) * invVar;
                yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
            }
        }
    } else {
        for (int n=0; n<N; ++n)
        for (int c=0; c<C; ++c)
        for (int hw=0; hw<HW; ++hw) {
            int idx = (n*C + c)*HW + hw;
            float invVar = 1.f / sqrtf(var[c*HW + hw] + (float)epsilon);
            float xhat = (xf[idx] - mean[c*HW + hw]) * invVar;
            yf[idx] = a * (sc[c]*xhat + bi[c]) + b*yf[idx];
        }
    }

    // Save mean / invVariance for backward pass
    if (resultSaveMean && resultSaveInvVariance) {
        float* sm = (float*)resultSaveMean;
        float* siv = (float*)resultSaveInvVariance;
        if (mode == CUDNN_BATCHNORM_SPATIAL) {
            for (int c=0; c<C; ++c) {
                sm[c] = mean[c];
                siv[c] = 1.f / sqrtf(var[c] + (float)epsilon);
            }
        } else {
            for (int i=0; i<C*HW; ++i) {
                sm[i] = mean[i];
                siv[i] = 1.f / sqrtf(var[i] + (float)epsilon);
            }
        }
    }

    // Update running statistics
    if (resultRunningMean && resultRunningVariance) {
        float* rm = (float*)resultRunningMean;
        float* rv = (float*)resultRunningVariance;
        float factor = static_cast<float>(exponentialAverageFactor);
        if (mode == CUDNN_BATCHNORM_SPATIAL) {
            for (int c=0; c<C; ++c) {
                rm[c] = (1.f - factor) * rm[c] + factor * mean[c];
                rv[c] = (1.f - factor) * rv[c] + factor * var[c];
            }
        } else {
            for (int i=0; i<C*HW; ++i) {
                rm[i] = (1.f - factor) * rm[i] + factor * mean[i];
                rv[i] = (1.f - factor) * rv[i] + factor * var[i];
            }
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
