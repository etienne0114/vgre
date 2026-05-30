// cuDNN internal shared types, enums, structs, and helpers.
// Included by all cudnn_*.cpp split files.

#pragma once

#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

// Thin RAII timer for cuDNN operations — records AEE and profiler events.
struct VgreDnnTimed {
    const char *name;
    size_t flops;
    size_t bytes;
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    VgreDnnTimed(const char *n, size_t f, size_t b) : name(n), flops(f), bytes(b) {}
    ~VgreDnnTimed() {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms <= 0.0) return;
        vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
            name, 1, 1, ms, bytes, flops);
        auto &prof = vgre::advanced::RuntimeProfiler::instance();
        if (prof.isEnabled()) {
            vgre::advanced::ProfileEvent pev;
            pev.kernelName   = name;
            pev.durationMs   = ms;
            pev.memoryBytes  = bytes;
            pev.flops        = flops;
            pev.throughputGBps = (bytes > 0) ? (bytes / 1e9) / (ms / 1000.0) : 0.0;
            pev.gflops         = (flops > 0) ? (flops / 1e9) / (ms / 1000.0) : 0.0;
            pev.timestamp    = std::chrono::steady_clock::now();
            prof.recordEvent(pev);
        }
    }
};

// ── cuDNN type stubs (no cudnn.h needed) ─────────────────────────────────────
typedef int    cudnnStatus_t;
typedef void*  cudnnHandle_t;
typedef void*  cudnnTensorDescriptor_t;
typedef void*  cudnnFilterDescriptor_t;
typedef void*  cudnnConvolutionDescriptor_t;
typedef void*  cudnnPoolingDescriptor_t;
typedef void*  cudnnActivationDescriptor_t;
typedef void*  cudnnDropoutDescriptor_t;
typedef void*  cudnnRNNDescriptor_t;
typedef void*  cudnnOpTensorDescriptor_t;
typedef void*  cudnnReduceTensorDescriptor_t;
typedef void*  cudnnMultiHeadAttnDescriptor_t;
typedef void*  cudnnLRNDescriptor_t;

static constexpr cudnnStatus_t CUDNN_STATUS_SUCCESS       = 0;
static constexpr cudnnStatus_t CUDNN_STATUS_NOT_INITIALIZED = 1;
static constexpr cudnnStatus_t CUDNN_STATUS_ALLOC_FAILED  = 2;
static constexpr cudnnStatus_t CUDNN_STATUS_BAD_PARAM      = 3;
static constexpr cudnnStatus_t CUDNN_STATUS_INTERNAL_ERROR = 4;
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
    CUDNN_ACTIVATION_GELU         = 7,
    CUDNN_ACTIVATION_SELU         = 8,
    CUDNN_ACTIVATION_MISH         = 9
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
    int k, c, r, s;
    cudnnDataType_t dtype;
    cudnnTensorFormat_t fmt = CUDNN_TENSOR_NCHW;
};
struct ConvDesc {
    int pad_h = 0, pad_w = 0, str_h = 1, str_w = 1, dil_h = 1, dil_w = 1;
    int groupCount = 1;
    int mathType   = 0; // 0 = CUDNN_DEFAULT_MATH
    int mode       = 1; // 1 = CUDNN_CROSS_CORRELATION
};
struct PoolDesc {
    cudnnPoolingMode_t mode;
    int win_h, win_w, pad_h, pad_w, str_h, str_w;
};
struct ActDesc {
    cudnnActivationMode_t mode;
    double coeff;
};
struct DropoutDesc {
    float dropout;
    unsigned long long seed;
    void* states = nullptr;
    size_t statesSize = 0;
};
struct HandleCtx {
    void* stream   = nullptr;
    int   deviceId = 0;
    bool  deterministicAlgorithms = false;
};

// ── LRN descriptor ───────────────────────────────────────────────────────────
enum cudnnLRNMode_t {
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
};
struct LRNDesc {
    unsigned lrnN;
    double   lrnAlpha;
    double   lrnBeta;
    double   lrnK;
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
    case CUDNN_ACTIVATION_GELU: {
        static constexpr float kSqrt2OverPi = 0.7978845608f;
        static constexpr float kCoeff       = 0.044715f;
        float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
        return 0.5f * x * (1.f + tanhf(inner));
    }
    case CUDNN_ACTIVATION_SELU: {
        static constexpr float kAlpha = 1.6732632423543772f;
        static constexpr float kScale = 1.0507009873554805f;
        return kScale * (x > 0.f ? x : kAlpha * (expf(x) - 1.f));
    }
    case CUDNN_ACTIVATION_MISH: {
        float sp = (x > 20.f) ? x : logf(1.f + expf(x));
        return x * tanhf(sp);
    }
    default: return x;
    }
}

// Index helper for NCHW layout: [n,c,h,w] -> flat index
static inline int nchw(int N, int C, int H, int W, int n, int c, int h, int w) {
    return ((n*C + c)*H + h)*W + w;
}

// ── Direct (naive) convolution: output = input * filter + bias ────────────────
inline void cpuConv2d(
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

// ── Backward-data convolution: computes dx from dy and w ─────────────────────
inline void cpuConv2dBackwardData(
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

// ── Backward-filter convolution: computes dw from x and dy ─────────────────────
inline void cpuConv2dBackwardFilter(
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

// Activation backward derivative helper
static inline float applyActivationDerivative(float x, float y, const ActDesc& act) {
    switch (act.mode) {
    case CUDNN_ACTIVATION_RELU:         return x > 0.f ? 1.f : 0.f;
    case CUDNN_ACTIVATION_SIGMOID: {
        float s = y; return s * (1.f - s);
    }
    case CUDNN_ACTIVATION_TANH: {
        float t = y; return 1.f - t * t;
    }
    case CUDNN_ACTIVATION_CLIPPED_RELU: return (x > 0.f && x < (float)act.coeff) ? 1.f : 0.f;
    case CUDNN_ACTIVATION_ELU: {
        return x > 0.f ? 1.f : (float)(act.coeff * expf(x));
    }
    case CUDNN_ACTIVATION_SWISH: {
        float sig = 1.f / (1.f + expf(-x));
        return sig + x * sig * (1.f - sig);
    }
    case CUDNN_ACTIVATION_GELU: {
        static constexpr float kSqrt2OverPi = 0.7978845608f;
        static constexpr float kCoeff       = 0.044715f;
        float x3 = x * x * x;
        float inner = kSqrt2OverPi * (x + kCoeff * x3);
        float tanh_i = tanhf(inner);
        float sech2 = 1.f - tanh_i * tanh_i;
        float inner_prime = kSqrt2OverPi * (1.f + 3.f * kCoeff * x * x);
        return 0.5f * (1.f + tanh_i) + 0.5f * x * sech2 * inner_prime;
    }
    case CUDNN_ACTIVATION_SELU: {
        static constexpr float kAlpha = 1.6732632423543772f;
        static constexpr float kScale = 1.0507009873554805f;
        return kScale * (x > 0.f ? 1.f : kAlpha * expf(x));
    }
    case CUDNN_ACTIVATION_MISH: {
        float sp = (x > 20.f) ? x : logf(1.f + expf(x));
        float tsp = tanhf(sp);
        float sig = 1.f / (1.f + expf(-x));
        float sech2 = 1.f - tsp * tsp;
        return tsp + x * sech2 * sig;
    }
    default: return 1.f;
    }
}

// ── Convolution algorithm enums ─────────────────────────────────────────────
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

// ── Batch Normalization mode ──────────────────────────────────────────────────
enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL        = 1
};

// ── Normalization (cuDNN 8.5+) ────────────────────────────────────────────────
enum cudnnNormMode_t {
    CUDNN_NORM_PER_ACTIVATION = 0,  // layer norm: per element scale/bias
    CUDNN_NORM_PER_CHANNEL    = 1   // batch norm: per channel (reuses BN path)
};
enum cudnnNormAlgo_t {
    CUDNN_NORM_ALGO_STANDARD   = 0,
    CUDNN_NORM_ALGO_PERSIST    = 1
};
enum cudnnNormOps_t {
    CUDNN_NORM_OPS_NORM        = 0,
    CUDNN_NORM_OPS_NORM_ACTIVATION = 1,
    CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = 2
};

// ── OpTensor ─────────────────────────────────────────────────────────────────
enum cudnnOpTensorOp_t {
    CUDNN_OP_TENSOR_ADD = 0,
    CUDNN_OP_TENSOR_MUL = 1,
    CUDNN_OP_TENSOR_MIN = 2,
    CUDNN_OP_TENSOR_MAX = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT = 5
};

struct OpTensorDesc {
    cudnnOpTensorOp_t op;
    cudnnDataType_t compType;
    cudnnNanPropagation_t nanOpt;
};

// ── ReduceTensor ─────────────────────────────────────────────────────────────
enum cudnnReduceTensorOp_t {
    CUDNN_REDUCE_TENSOR_ADD = 0,
    CUDNN_REDUCE_TENSOR_MUL = 1,
    CUDNN_REDUCE_TENSOR_MIN = 2,
    CUDNN_REDUCE_TENSOR_MAX = 3,
    CUDNN_REDUCE_TENSOR_AMAX = 4,
    CUDNN_REDUCE_TENSOR_AVG = 5,
    CUDNN_REDUCE_TENSOR_NORM1 = 6,
    CUDNN_REDUCE_TENSOR_NORM2 = 7
};

struct ReduceTensorDesc {
    cudnnReduceTensorOp_t op;
    cudnnDataType_t compType;
    cudnnNanPropagation_t nanOpt;
};

// ── RNN enums ────────────────────────────────────────────────────────────────
enum cudnnRNNInputMode_t {
    CUDNN_LINEAR_INPUT = 0,
    CUDNN_SKIP_INPUT = 1
};
enum cudnnDirectionMode_t {
    CUDNN_UNIDIRECTIONAL = 0,
    CUDNN_BIDIRECTIONAL = 1
};
enum cudnnRNNMode_t {
    CUDNN_RNN_RELU = 0,
    CUDNN_RNN_TANH = 1,
    CUDNN_LSTM = 2,
    CUDNN_GRU = 3
};

struct RNNDesc {
    int hiddenSize;
    int numLayers;
    cudnnRNNMode_t mode;
    cudnnDropoutDescriptor_t dropoutDesc;
    int inputSize;
};

// ── Attention enums ──────────────────────────────────────────────────────────
enum cudnnMultiHeadAttnMode_t {
    CUDNN_MHA_MODE_SCALE_DOT_PRODUCT = 0
};

struct MultiHeadAttnDesc {
    int numHeads;
    int headDim;
    int modelDim;
    int seqLen;
    int batchSize;
    float scaling;
};

// ── Pointwise operation modes (cudnnPointwiseMode_t) ─────────────────────────
// Used by CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR and
// CUDNN_BACKEND_POINTWISE_DESCRIPTOR (the mode sub-descriptor).
enum cudnnPointwiseMode_t {
    CUDNN_POINTWISE_ADD           = 0,   // binary: Y = alpha1*X + alpha2*B
    CUDNN_POINTWISE_MUL           = 1,   // binary: Y = alpha1*X * alpha2*B
    CUDNN_POINTWISE_MIN           = 2,
    CUDNN_POINTWISE_MAX           = 3,
    CUDNN_POINTWISE_SQRT          = 4,   // unary
    CUDNN_POINTWISE_RELU_FWD      = 5,
    CUDNN_POINTWISE_TANH_FWD      = 6,
    CUDNN_POINTWISE_SIGMOID_FWD   = 7,
    CUDNN_POINTWISE_ELU_FWD       = 8,
    CUDNN_POINTWISE_GELU_FWD      = 9,
    CUDNN_POINTWISE_SOFTPLUS_FWD  = 10,
    CUDNN_POINTWISE_SWISH_FWD     = 11,
    CUDNN_POINTWISE_RELU_BWD      = 12,
    CUDNN_POINTWISE_TANH_BWD      = 13,
    CUDNN_POINTWISE_SIGMOID_BWD   = 14,
    CUDNN_POINTWISE_ELU_BWD       = 15,
    CUDNN_POINTWISE_GELU_BWD      = 16,
    CUDNN_POINTWISE_SOFTPLUS_BWD  = 17,
    CUDNN_POINTWISE_SWISH_BWD     = 18,
    CUDNN_POINTWISE_DIV           = 19,
    CUDNN_POINTWISE_ADD_SQUARE    = 20,
    CUDNN_POINTWISE_EXP           = 21,
    CUDNN_POINTWISE_LOG           = 22,
    CUDNN_POINTWISE_NEG           = 23,
    CUDNN_POINTWISE_MOD           = 24,
    CUDNN_POINTWISE_POW           = 25,
    CUDNN_POINTWISE_ABS           = 26,
    CUDNN_POINTWISE_CEIL          = 27,
    CUDNN_POINTWISE_CMP_EQ        = 28,  // binary compare → bool
    CUDNN_POINTWISE_CMP_NEQ       = 29,
    CUDNN_POINTWISE_CMP_GT        = 30,
    CUDNN_POINTWISE_CMP_GE        = 31,
    CUDNN_POINTWISE_CMP_LT        = 32,
    CUDNN_POINTWISE_CMP_LE        = 33,
    CUDNN_POINTWISE_LOGICAL_AND   = 34,
    CUDNN_POINTWISE_LOGICAL_OR    = 35,
    CUDNN_POINTWISE_LOGICAL_NOT   = 36,
    CUDNN_POINTWISE_FLOOR         = 37,
    CUDNN_POINTWISE_IDENTITY      = 38,
    CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = 39,
    CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = 40,
    CUDNN_POINTWISE_RECIPROCAL    = 41,
};

// ── cudnnForwardMode_t (cuDNN v8.5+) ─────────────────────────────────────────
enum cudnnForwardMode_t {
    CUDNN_FWD_MODE_INFERENCE = 0,
    CUDNN_FWD_MODE_TRAINING  = 1
};

// ── cudnnRNNDataDescriptor (for RNNForwardTrainingEx) ─────────────────────────
enum cudnnRNNDataLayout_t {
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED   = 1,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2,
};

struct cudnnRNNData_st {
    cudnnDataType_t      dataType    = CUDNN_DATA_FLOAT;
    cudnnRNNDataLayout_t layout      = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    int                  maxSeqLength = 0;
    int                  batchSize    = 0;
    int                  vectorSize   = 0;
    std::vector<int>     seqLengthArray;
    float                paddingFill  = 0.0f;
};
typedef cudnnRNNData_st* cudnnRNNDataDescriptor_t;
