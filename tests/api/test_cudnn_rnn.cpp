/**
 * Test: cuDNN RNN Forward/Backward (P3.9)
 *
 * Verifies cudnnRNNForwardInference/Training and cudnnRNNBackwardData/Weights
 * for a simple vanilla tanh RNN cell.
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
typedef void* cudnnRNNDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };
enum cudnnRNNMode_t { CUDNN_RNN_TANH = 1 };
enum cudnnRNNInputMode_t { CUDNN_LINEAR_INPUT = 0 };
enum cudnnDirectionMode_t { CUDNN_UNIDIRECTIONAL = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* d);
cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t d);
cudnnStatus_t cudnnSetRNNDescriptor(
    cudnnRNNDescriptor_t desc, int hiddenSize, int numLayers,
    void* dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnDataType_t dtype);

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    int seqLength, cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    int seqLength, cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);

cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void* workspace, size_t workSpaceSizeInBytes);

cudnnStatus_t cudnnRNNForwardTraining(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void* workspace, size_t workSpaceSizeInBytes,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);

cudnnStatus_t cudnnRNNBackwardData(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int seqLength,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    cudnnTensorDescriptor_t* dyDesc, const void* dy,
    cudnnTensorDescriptor_t dhyDesc, const void* dhy,
    cudnnTensorDescriptor_t dcyDesc, const void* dcy,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnTensorDescriptor_t* dxDesc, void* dx,
    cudnnTensorDescriptor_t dhxDesc, void* dhx,
    cudnnTensorDescriptor_t dcxDesc, void* dcx,
    void* workspace, size_t workSpaceSizeInBytes,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);

cudnnStatus_t cudnnRNNBackwardWeights(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    void* workspace, size_t workSpaceSizeInBytes,
    cudnnFilterDescriptor_t dwDesc, void* dw,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);
}

static bool approx_eq(float a, float b, float eps = 1e-3f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

static void refRNNForward(int seqLen, int batch, int inputSize, int hiddenSize,
    const float* x, const float* hx, const float* w, float* y)
{
    int wihSize = hiddenSize * inputSize;
    int whhSize = hiddenSize * hiddenSize;
    const float* W_ih = w;
    const float* W_hh = w + wihSize;
    const float* b = w + wihSize + whhSize;

    std::vector<float> hPrev(batch * hiddenSize, 0.f);
    if (hx) {
        for (int i = 0; i < batch * hiddenSize; ++i) hPrev[i] = hx[i];
    }

    for (int t = 0; t < seqLen; ++t) {
        const float* x_t = x + t * batch * inputSize;
        float* y_t = y + t * batch * hiddenSize;
        for (int n = 0; n < batch; ++n) {
            for (int j = 0; j < hiddenSize; ++j) {
                float sum = b[j];
                for (int k = 0; k < inputSize; ++k)
                    sum += W_ih[j * inputSize + k] * x_t[n * inputSize + k];
                for (int k = 0; k < hiddenSize; ++k)
                    sum += W_hh[j * hiddenSize + k] * hPrev[n * hiddenSize + k];
                y_t[n * hiddenSize + j] = tanhf(sum);
            }
        }
        for (int i = 0; i < batch * hiddenSize; ++i) hPrev[i] = y_t[i];
    }
}

int main() {
    std::cout << "=== Test: cuDNN RNN Forward/Backward (P3.9) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    int seqLen = 3, batch = 2, inputSize = 4, hiddenSize = 3;
    int wihSize = hiddenSize * inputSize;
    int whhSize = hiddenSize * hiddenSize;
    int totalWeights = wihSize + whhSize + hiddenSize;

    std::vector<float> x(seqLen * batch * inputSize);
    std::vector<float> hx(batch * hiddenSize, 0.f);
    std::vector<float> w(totalWeights);
    std::vector<float> y(seqLen * batch * hiddenSize, 0.f);
    std::vector<float> refY(seqLen * batch * hiddenSize, 0.f);

    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i % 5) - 2) * 0.1f;
    for (int i = 0; i < totalWeights; ++i) w[i] = static_cast<float>((i % 3) - 1) * 0.1f;

    // Build per-timestep descriptors
    std::vector<cudnnTensorDescriptor_t> xDesc(seqLen), yDesc(seqLen);
    for (int t = 0; t < seqLen; ++t) {
        cudnnCreateTensorDescriptor(&xDesc[t]);
        cudnnSetTensor4dDescriptor(xDesc[t], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, inputSize, 1, 1);
        cudnnCreateTensorDescriptor(&yDesc[t]);
        cudnnSetTensor4dDescriptor(yDesc[t], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, hiddenSize, 1, 1);
    }
    cudnnTensorDescriptor_t hxDesc;
    cudnnCreateTensorDescriptor(&hxDesc);
    cudnnSetTensor4dDescriptor(hxDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, hiddenSize, 1, 1);

    cudnnRNNDescriptor_t rnnDesc;
    cudnnCreateRNNDescriptor(&rnnDesc);
    cudnnSetRNNDescriptor(rnnDesc, hiddenSize, 1, nullptr, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_RNN_TANH, CUDNN_DATA_FLOAT);

    // ── Test 1: Workspace / ReserveSize queries ─────────────────────────
    ++total;
    {
        size_t ws = 99, rs = 99;
        auto r1 = cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLen, xDesc.data(), &ws);
        auto r2 = cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLen, xDesc.data(), &rs);
        if (r1 == CUDNN_STATUS_SUCCESS && r2 == CUDNN_STATUS_SUCCESS && ws == 0 && rs == 0) {
            std::cout << "PASS [Workspace/Reserve queries]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Workspace/Reserve] r1=" << r1 << " r2=" << r2 << " ws=" << ws << " rs=" << rs << "\n";
        }
    }

    // ── Test 2: Forward inference vs reference ────────────────────────────
    ++total;
    {
        auto r = cudnnRNNForwardInference(handle, rnnDesc, seqLen, xDesc.data(), x.data(),
            hxDesc, hx.data(), nullptr, nullptr, nullptr, w.data(),
            yDesc.data(), y.data(), nullptr, nullptr, nullptr, nullptr, 0, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);

        refRNNForward(seqLen, batch, inputSize, hiddenSize, x.data(), hx.data(), w.data(), refY.data());

        if (ok) {
            for (size_t i = 0; i < y.size() && ok; ++i) {
                if (!approx_eq(y[i], refY[i], 1e-3f)) {
                    std::cerr << "FAIL [Forward] y[" << i << "]=" << y[i] << " expected=" << refY[i] << "\n";
                    ok = false;
                }
            }
        } else {
            std::cerr << "FAIL [Forward] status=" << r << "\n";
        }
        if (ok) { std::cout << "PASS [Forward inference]\n"; ++pass; }
    }

    // ── Test 3: Forward training equals inference ─────────────────────────
    ++total;
    {
        std::vector<float> yTrain(seqLen * batch * hiddenSize, 0.f);
        auto r = cudnnRNNForwardTraining(handle, rnnDesc, seqLen, xDesc.data(), x.data(),
            hxDesc, hx.data(), nullptr, nullptr, nullptr, w.data(),
            yDesc.data(), yTrain.data(), nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            for (size_t i = 0; i < y.size() && ok; ++i) {
                if (!approx_eq(yTrain[i], refY[i], 1e-3f)) {
                    std::cerr << "FAIL [Training] y[" << i << "]=" << yTrain[i] << " expected=" << refY[i] << "\n";
                    ok = false;
                }
            }
        }
        if (ok) { std::cout << "PASS [Forward training]\n"; ++pass; }
    }

    // ── Test 4: Backward weights non-zero ───────────────────────────────
    ++total;
    {
        std::vector<float> dw(totalWeights, 0.f);
        auto r = cudnnRNNBackwardWeights(handle, rnnDesc, seqLen, xDesc.data(), x.data(),
            hxDesc, hx.data(), yDesc.data(), y.data(), nullptr, 0, nullptr, dw.data(), nullptr, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            bool anyNonZero = false;
            for (float v : dw) if (v != 0.f) { anyNonZero = true; break; }
            if (!anyNonZero) {
                std::cerr << "FAIL [BackwardWeights] all zeros\n"; ok = false;
            }
        }
        if (ok) { std::cout << "PASS [Backward weights non-zero]\n"; ++pass; }
    }

    // ── Test 5: Null pointer rejection ──────────────────────────────────
    ++total;
    {
        auto r = cudnnRNNForwardInference(handle, nullptr, 0, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    for (int t = 0; t < seqLen; ++t) {
        cudnnDestroyTensorDescriptor(xDesc[t]);
        cudnnDestroyTensorDescriptor(yDesc[t]);
    }
    cudnnDestroyTensorDescriptor(hxDesc);
    cudnnDestroyRNNDescriptor(rnnDesc);
    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
