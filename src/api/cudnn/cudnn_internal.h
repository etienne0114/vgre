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
#ifdef __AVX2__
#include <immintrin.h>
#endif

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
        // Fast GELU via degree-7 Horner polynomial; Hendrycks & Gimpel 2016.
        // inner = √(2/π)·(x + 0.044715·x³); tanh(inner) ≈ Horner7 on |inner|<4.
        // Coefficients: Chebyshev minimax on [−4,4], not Taylor (Taylor diverges for
        // |inner|>π/2≈1.57). For |inner|≥4: tanh≈±1 → GELU→x or 0.
        static constexpr float kA  = 0.7978845608f;  // √(2/π)
        static constexpr float kB  = 0.044715f;
        // Chebyshev minimax degree-7 coefficients for tanh(u) on [−4,4]:
        static constexpr float kH1 =  0.924533f;
        static constexpr float kH3 = -0.152054f;
        static constexpr float kH5 =  0.012711f;
        static constexpr float kH7 = -0.000368f;
        float inner = kA * (x + kB * (x * x * x));
        float t;
        if (inner >= 4.0f)       t =  1.0f;
        else if (inner <= -4.0f) t = -1.0f;
        else {
            float u2 = inner * inner;
            // Horner: u·(kH1 + u²·(kH3 + u²·(kH5 + u²·kH7)))  O(log U) → O(1) ops
            t = inner * (kH1 + u2 * (kH3 + u2 * (kH5 + u2 * kH7)));
            t = (t >  1.0f) ?  1.0f : (t < -1.0f) ? -1.0f : t;
        }
        return 0.5f * x * (1.0f + t);
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

// ── Direct (naive) convolution — supports group, dilation, stride ─────────────
// Forward: Y[n,k,oh,ow] = Σ_{g,c_local,r,s} X[n, g*Cg+c_local, ih, iw]
//                                           × W[g*Kg+k_local, c_local, r, s]
//   where group g ∈ [0,G), Cg = C/G (input channels per group),
//   Kg = K/G (output channels per group), k_local = k - g*Kg.
//   Depthwise = group G==C==K (Cg=1, Kg=1).
// Filter layout: W[K, C/G, R, S] = W[(k * Cg + c_local) * R * S + r*S + s].
inline void cpuConv2d(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* input, const float* filter, float* output,
    int groupCount = 1)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C / groupCount; // input channels per group
    int Kg = K / groupCount; // output channels per group

    for (int n = 0; n < N; ++n)
    for (int g = 0; g < groupCount; ++g)
    for (int kg = 0; kg < Kg; ++kg) {
        int k = g * Kg + kg;
        for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
            float acc = 0.f;
            for (int cg = 0; cg < Cg; ++cg)
            for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s) {
                int ih = oh*str_h + r*dil_h - pad_h;
                int iw = ow*str_w + s*dil_w - pad_w;
                if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
                int c = g * Cg + cg;
                acc += input[nchw(N,C,H,W,n,c,ih,iw)] *
                       filter[((k * Cg + cg) * R + r) * S + s];
            }
            output[nchw(N,K,outH,outW,n,k,oh,ow)] = acc;
        }
    }
}

// ── Backward-data: ∂L/∂X = rot180(W) ⊛_full dY ──────────────────────────────
// For group g: dX[n, g*Cg+cg, ih, iw] += Σ_{kg,oh,ow} dY[n,g*Kg+kg,oh,ow]
//                                       × W[(g*Kg+kg)*Cg+cg, r, s]
// where ih = oh*str + r*dil - pad (only when in-bounds).
// Dilation inserts (dil-1) zeros between filter taps:
//   effective filter position at ih,iw: r = (ih + pad - oh*str) / dil (integer).
// Filter layout: W[K, C/G, R, S].
inline void cpuConv2dBackwardData(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* dy, const float* w, float* dx,
    int groupCount = 1)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C / groupCount;
    int Kg = K / groupCount;

    for (int n = 0; n < N; ++n)
    for (int g = 0; g < groupCount; ++g)
    for (int kg = 0; kg < Kg; ++kg) {
        int k = g * Kg + kg;
        for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
            float dyVal = dy[nchw(N,K,outH,outW,n,k,oh,ow)];
            for (int cg = 0; cg < Cg; ++cg)
            for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s) {
                int ih = oh*str_h + r*dil_h - pad_h;
                int iw = ow*str_w + s*dil_w - pad_w;
                if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
                int c = g * Cg + cg;
                dx[nchw(N,C,H,W,n,c,ih,iw)] +=
                    dyVal * w[((k * Cg + cg) * R + r) * S + s];
            }
        }
    }
}

// ── Backward-filter: ∂L/∂W = X ⊛ dY (cross-correlation) ─────────────────────
// For group g: dW[(g*Kg+kg)*Cg+cg, r, s] +=
//     Σ_{n,oh,ow} X[n, g*Cg+cg, oh*str+r*dil-pad, ow*str+s*dil-pad]
//               × dY[n, g*Kg+kg, oh, ow]
// Dilated filter: effective position (r*dil_h, s*dil_w) in the input.
// Filter layout: W[K, C/G, R, S].
inline void cpuConv2dBackwardFilter(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* x, const float* dy, float* dw,
    int groupCount = 1)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C / groupCount;
    int Kg = K / groupCount;

    for (int n = 0; n < N; ++n)
    for (int g = 0; g < groupCount; ++g)
    for (int kg = 0; kg < Kg; ++kg) {
        int k = g * Kg + kg;
        for (int cg = 0; cg < Cg; ++cg)
        for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s) {
            float acc = 0.f;
            for (int oh = 0; oh < outH; ++oh)
            for (int ow = 0; ow < outW; ++ow) {
                int ih = oh*str_h + r*dil_h - pad_h;
                int iw = ow*str_w + s*dil_w - pad_w;
                if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
                int c = g * Cg + cg;
                acc += x[nchw(N,C,H,W,n,c,ih,iw)] *
                       dy[nchw(N,K,outH,outW,n,k,oh,ow)];
            }
            dw[((k * Cg + cg) * R + r) * S + s] += acc;
        }
    }
}

// ── Winograd F(4×4, 3×3) convolution forward pass ──────────────────────────
// Reference: Lavin & Gray 2015, "Fast Algorithms for Convolutional Neural Networks"
// Math: Y_tile = A^T ⊙ [(G·g·G^T) ⊗ (B^T·d·B)] · A
//   where ⊗ denotes element-wise product, g=filter(3×3), d=input-tile(6×6)
//   G(6×3): filter transform; B^T(6×6): input transform; A^T(4×6): output inverse
// FLOP savings: element-wise multiply 36 ops/tile vs 9×16=144 for direct 4×4 output
//   → ~2.25× theoretical speedup for large channel counts (transform overhead amortised)
// Constraints: R=S=3, stride=1, dilation=1, groupCount≥1

// Winograd transform matrices for F(4,3): α = m+r-1 = 6 (tile side)
// B^T (6×6): maps 6×6 input patch to Winograd transform domain
static const float kWinoB[6][6] = {
    { 4.f,  0.f, -5.f,  0.f,  1.f,  0.f},
    { 0.f, -4.f, -4.f,  1.f,  1.f,  0.f},
    { 0.f,  4.f, -4.f, -1.f,  1.f,  0.f},
    { 0.f, -2.f, -1.f,  2.f,  1.f,  0.f},
    { 0.f,  2.f, -1.f, -2.f,  1.f,  0.f},
    { 0.f,  4.f,  0.f, -5.f,  0.f,  1.f}
};

// G (6×3): maps 3×3 filter to Winograd transform domain
static const float kWinoG[6][3] = {
    { 1.f/4.f,    0.f,       0.f      },
    {-1.f/6.f,  -1.f/6.f,  -1.f/6.f  },
    {-1.f/6.f,   1.f/6.f,  -1.f/6.f  },
    { 1.f/24.f,  1.f/12.f,  1.f/6.f  },
    { 1.f/24.f, -1.f/12.f,  1.f/6.f  },
    { 0.f,       0.f,       1.f      }
};

// A^T (4×6): inverse-transforms 6×6 product to 4×4 output tile
static const float kWinoA[4][6] = {
    {1.f, 1.f,  1.f, 1.f,  1.f, 0.f},
    {0.f, 1.f, -1.f, 2.f, -2.f, 0.f},
    {0.f, 1.f,  1.f, 4.f,  4.f, 0.f},
    {0.f, 1.f, -1.f, 8.f, -8.f, 1.f}
};

// g_hat[6×6] = G[6×3] × g[3×3] × G^T[3×6]
static inline void winoFilterTransform(const float g[3][3], float g_hat[6][6]) {
    float tmp[6][3];
    for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j) {
        float s = 0.f;
        for (int k = 0; k < 3; ++k) s += kWinoG[i][k] * g[k][j];
        tmp[i][j] = s;
    }
    // g_hat = tmp × G^T  (G^T[k][j] = kWinoG[j][k])
    for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) {
        float s = 0.f;
        for (int k = 0; k < 3; ++k) s += tmp[i][k] * kWinoG[j][k];
        g_hat[i][j] = s;
    }
}

// d_hat[6×6] = B^T[6×6] × d[6×6] × B[6×6]  (B = (B^T)^T)
static inline void winoInputTransform(const float d[6][6], float d_hat[6][6]) {
    float tmp[6][6];
    for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) {
        float s = 0.f;
        for (int k = 0; k < 6; ++k) s += kWinoB[i][k] * d[k][j];
        tmp[i][j] = s;
    }
    // d_hat = tmp × B  (B[k][j] = B^T[j][k] = kWinoB[j][k])
    for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) {
        float s = 0.f;
        for (int k = 0; k < 6; ++k) s += tmp[i][k] * kWinoB[j][k];
        d_hat[i][j] = s;
    }
}

// y[4×4] = A^T[4×6] × m_hat[6×6] × A[6×4]  (A = (A^T)^T)
static inline void winoOutputTransform(const float m_hat[6][6], float y[4][4]) {
    float tmp[4][6];
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 6; ++j) {
        float s = 0.f;
        for (int k = 0; k < 6; ++k) s += kWinoA[i][k] * m_hat[k][j];
        tmp[i][j] = s;
    }
    // y = tmp × A  (A[k][j] = A^T[j][k] = kWinoA[j][k])
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < 6; ++k) s += tmp[i][k] * kWinoA[j][k];
        y[i][j] = s;
    }
}

// cpuWinograd4x4_3x3: Winograd F(4×4,3×3) forward pass.
// Complexity: O(N·G·Kg·Cg·T·36) multiply-adds where T = ⌈outH/4⌉·⌈outW/4⌉
//   vs O(N·K·C·outH·outW·9) for direct — speedup ≈2.25× for large C,K.
// Preconditions: R=S=3, stride=1, dilation=1, groupCount divides C and K.
// Filter layout: W[K, C/G, 3, 3] (K outer, C/G inner).
// Output layout: NCHW; output zeroed by caller.
inline void cpuWinograd4x4_3x3(
    int N, int C, int H, int W,
    int K, int pad_h, int pad_w,
    const float* input, const float* filter, float* output,
    int groupCount = 1)
{
    int Cg   = C / groupCount;
    int Kg   = K / groupCount;
    int outH = H + 2*pad_h - 2;  // R=3: outH = H + 2·pad - (R-1)
    int outW = W + 2*pad_w - 2;
    int nt_h = (outH + 3) / 4;   // tiles needed in H direction
    int nt_w = (outW + 3) / 4;   // tiles needed in W direction

    // Pre-compute filter transforms: g_hat[k][cg] is a flattened 6×6 matrix
    // Layout: g_hat[(k·Cg + cg)·36 + i·6 + j]
    // Invariant: g_hat = G·W[k,cg,:,:]·G^T for each (k, cg)
    std::vector<float> g_hat(static_cast<size_t>(K) * Cg * 36, 0.f);
    for (int k = 0; k < K; ++k)
    for (int cg = 0; cg < Cg; ++cg) {
        float g[3][3];
        const float* fp = filter + (k * Cg + cg) * 9;
        for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            g[r][s] = fp[r*3 + s];
        float gh[6][6];
        winoFilterTransform(g, gh);
        float* dst = g_hat.data() + (k * Cg + cg) * 36;
        for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            dst[i*6+j] = gh[i][j];
    }

    // Per-tile work: transform, multiply, inverse-transform
    // d_hat[cg·36]: transformed input patches for all Cg channels of this tile
    // m_hat[kg·36]: accumulated element-wise products in transform domain
    std::vector<float> d_hat(static_cast<size_t>(Cg) * 36);
    std::vector<float> m_hat(static_cast<size_t>(Kg) * 36);

    for (int n = 0; n < N; ++n)
    for (int grp = 0; grp < groupCount; ++grp) {
        int cin_base  = grp * Cg;
        int cout_base = grp * Kg;
        for (int th = 0; th < nt_h; ++th)
        for (int tw = 0; tw < nt_w; ++tw) {
            int h0 = th * 4 - pad_h;  // top-left row in original input
            int w0 = tw * 4 - pad_w;  // top-left col in original input

            // Input transform: d_hat[cg] = B^T · patch[cg] · B
            for (int cg = 0; cg < Cg; ++cg) {
                int c = cin_base + cg;
                float patch[6][6];
                for (int pi = 0; pi < 6; ++pi)
                for (int pj = 0; pj < 6; ++pj) {
                    int ih = h0 + pi, iw = w0 + pj;
                    patch[pi][pj] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    ? input[nchw(N,C,H,W,n,c,ih,iw)] : 0.f;
                }
                float dh[6][6];
                winoInputTransform(patch, dh);
                float* dp = d_hat.data() + cg * 36;
                for (int i = 0; i < 36; ++i) dp[i] = (&dh[0][0])[i];
            }

            // Element-wise multiply in transform domain (the Winograd hotspot):
            // m_hat[kg][i] = Σ_{cg} g_hat[cout_base+kg][cg][i] · d_hat[cg][i]
            // AVX2: process 8 transform coefficients per cycle with FMA.
            std::fill(m_hat.begin(), m_hat.end(), 0.f);
            for (int kg = 0; kg < Kg; ++kg) {
                int k = cout_base + kg;
                float* mp = m_hat.data() + kg * 36;
                for (int cg = 0; cg < Cg; ++cg) {
                    const float* gp = g_hat.data() + (k * Cg + cg) * 36;
                    const float* dp = d_hat.data() + cg * 36;
#ifdef __AVX2__
                    // AVX2 FMA: 8 floats/iter × 4 iters = 32 elements
                    for (int i = 0; i < 32; i += 8) {
                        __m256 m = _mm256_loadu_ps(mp + i);
                        __m256 g = _mm256_loadu_ps(gp + i);
                        __m256 d = _mm256_loadu_ps(dp + i);
                        _mm256_storeu_ps(mp + i, _mm256_fmadd_ps(g, d, m));
                    }
                    for (int i = 32; i < 36; ++i) mp[i] += gp[i] * dp[i];
#else
                    for (int i = 0; i < 36; ++i) mp[i] += gp[i] * dp[i];
#endif
                }
            }

            // Output inverse transform: y[4×4] = A^T · m_hat[kg] · A
            for (int kg = 0; kg < Kg; ++kg) {
                int k = cout_base + kg;
                float mh[6][6];
                const float* mp = m_hat.data() + kg * 36;
                for (int i = 0; i < 36; ++i) (&mh[0][0])[i] = mp[i];
                float y_tile[4][4];
                winoOutputTransform(mh, y_tile);
                for (int ti = 0; ti < 4; ++ti)
                for (int tj = 0; tj < 4; ++tj) {
                    int oh = th*4 + ti, ow = tw*4 + tj;
                    if (oh < outH && ow < outW)
                        output[nchw(N,K,outH,outW,n,k,oh,ow)] += y_tile[ti][tj];
                }
            }
        }
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
        // Derivative must use ACTUAL polynomial derivative dp/du, NOT 1−t².
        // (1−t² is only exact for mathematical tanh; our t is a polynomial.)
        // For |inner|≥4: t=±1, polynomial derivative=0 → grad=1 or 0 correctly.
        static constexpr float kA  = 0.7978845608f;
        static constexpr float kB  = 0.044715f;
        static constexpr float kH1 =  0.924533f;
        static constexpr float kH3 = -0.152054f;
        static constexpr float kH5 =  0.012711f;
        static constexpr float kH7 = -0.000368f;
        float inner = kA * (x + kB * (x * x * x));
        float t;
        float dpdu;
        if (inner >= 4.0f) {
            t = 1.0f; dpdu = 0.0f;
        } else if (inner <= -4.0f) {
            t = -1.0f; dpdu = 0.0f;
        } else {
            float u2 = inner * inner;
            float t_raw = inner * (kH1 + u2 * (kH3 + u2 * (kH5 + u2 * kH7)));
            if (t_raw >= 1.0f) {
                t = 1.0f; dpdu = 0.0f;
            } else if (t_raw <= -1.0f) {
                t = -1.0f; dpdu = 0.0f;
            } else {
                t = t_raw;
                // p'(u) = kH1 + u²·(3·kH3 + u²·(5·kH5 + u²·7·kH7))  O(1) Horner
                dpdu = kH1 + u2 * (3.0f*kH3 + u2 * (5.0f*kH5 + u2 * 7.0f*kH7));
            }
        }
        float inner_prime = kA * (1.0f + 3.0f * kB * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * dpdu * inner_prime;
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
    CUDNN_NORM_PER_CHANNEL    = 1,  // batch norm: per channel (reuses BN path)
    CUDNN_NORM_RMS_LAYER      = 2   // RMSNorm: normalise by root-mean-square (LLaMA/Gemma)
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
    bool peephole = false;  // QUEUE-38: LSTM peephole connections
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
    const void* cachedWeights = nullptr;  // cached from most recent Forward call
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
