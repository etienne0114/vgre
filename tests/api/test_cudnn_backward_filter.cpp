/**
 * Test: cuDNN Convolution Backward Filter (P3.2)
 *
 * Verifies cudnnConvolutionBackwardFilter computes dw correctly
 * by comparing with a direct reference implementation.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnConvolutionDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* d);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d,
    cudnnDataType_t dtype, cudnnTensorFormat_t fmt, int k, int c, int r, int s);

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* d);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    int mode, cudnnDataType_t computeType);

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    int algo, void* workspace, size_t wsSize,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw);
}

static bool approx_eq(float a, float b, float eps = 5e-3f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

static inline int nchw(int N, int C, int H, int W, int n, int c, int h, int w) {
    return ((n*C + c)*H + h)*W + w;
}

// Reference backward-filter: dw[k,c,r,s] += sum_{n,oh,ow} dy[n,k,oh,ow] * x[n,c,ih,iw]
static void refConvBackwardFilter(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w,
    const float* x, const float* dy, float* dw)
{
    int outH = (H + 2*pad_h - R)/str_h + 1;
    int outW = (W + 2*pad_w - S)/str_w + 1;
    memset(dw, 0, K*C*R*S*sizeof(float));
    for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s) {
        float acc = 0;
        for (int oh = 0; oh < outH; ++oh)
        for (int ow = 0; ow < outW; ++ow) {
            int ih = oh*str_h + r - pad_h;
            int iw = ow*str_w + s - pad_w;
            if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;
            acc += x[nchw(N,C,H,W,n,c,ih,iw)] * dy[((n*K+k)*outH+oh)*outW+ow];
        }
        dw[((k*C+c)*R+r)*S+s] += acc;
    }
}

static bool runTest(
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w,
    cudnnHandle_t handle, const char* name, int& pass, int& total)
{
    ++total;
    int outH = (H + 2*pad_h - R)/str_h + 1;
    int outW = (W + 2*pad_w - S)/str_w + 1;

    std::vector<float> x(N*C*H*W);
    std::vector<float> dy(N*K*outH*outW);
    std::vector<float> dw(K*C*R*S);
    std::vector<float> ref_dw(K*C*R*S);

    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i % 5) - 2);
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = static_cast<float>((i % 4) - 1);

    cudnnTensorDescriptor_t xDesc, dyDesc;
    cudnnFilterDescriptor_t dwDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,C,H,W);
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N,K,outH,outW);
    cudnnCreateFilterDescriptor(&dwDesc);
    cudnnSetFilter4dDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K,C,R,S);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, pad_h,pad_w,str_h,str_w,1,1,0,CUDNN_DATA_FLOAT);

    float alpha = 1.0f, beta = 0.0f;
    auto r = cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x.data(),
                                             dyDesc, dy.data(), convDesc, 0, nullptr, 0,
                                             &beta, dwDesc, dw.data());

    bool ok = (r == CUDNN_STATUS_SUCCESS);
    if (ok) {
        refConvBackwardFilter(N,C,H,W,K,R,S,pad_h,pad_w,str_h,str_w,
                              x.data(), dy.data(), ref_dw.data());
        for (size_t i = 0; i < dw.size() && ok; ++i) {
            if (!approx_eq(dw[i], ref_dw[i], 1e-3f)) {
                std::cerr << "FAIL [" << name << "] mismatch at i=" << i
                          << " computed=" << dw[i] << " expected=" << ref_dw[i] << "\n";
                ok = false;
                break;
            }
        }
    } else {
        std::cerr << "FAIL [" << name << "] status=" << r << "\n";
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroyFilterDescriptor(dwDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    if (ok) { std::cout << "PASS [" << name << "]\n"; ++pass; }
    return ok;
}

int main() {
    std::cout << "=== Test: cuDNN Convolution Backward Filter (P3.2) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    // ── Test 1: 3×3 conv, stride 1, pad 1 ─────────────────────────────────
    runTest(1, 2, 4, 4, 3, 3, 3, 1, 1, 1, 1, handle, "3x3 s1 p1", pass, total);

    // ── Test 2: 1×1 conv (GEMM path) ──────────────────────────────────────
    runTest(1, 3, 3, 3, 2, 1, 1, 0, 0, 1, 1, handle, "1x1 s1 p0", pass, total);

    // ── Test 3: 3×3 conv, stride 2, pad 1 ─────────────────────────────────
    runTest(1, 2, 5, 5, 2, 3, 3, 1, 1, 2, 2, handle, "3x3 s2 p1", pass, total);

    // ── Test 4: Null pointer rejection ────────────────────────────────────
    ++total;
    {
        auto r = cudnnConvolutionBackwardFilter(handle, nullptr, nullptr, nullptr,
                                                 nullptr, nullptr, nullptr, 0, nullptr, 0,
                                                 nullptr, nullptr, nullptr);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
