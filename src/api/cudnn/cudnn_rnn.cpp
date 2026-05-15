// cuDNN RNN forward + backward

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"

extern "C" {

cudnnStatus_t cudnnGetRNNWorkspaceSize(
    cudnnHandle_t,
    cudnnRNNDescriptor_t,
    int /*seqLength*/,
    cudnnTensorDescriptor_t* /*xDesc*/,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t,
    cudnnRNNDescriptor_t,
    int /*seqLength*/,
    cudnnTensorDescriptor_t* /*xDesc*/,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    int seqLength,
    cudnnTensorDescriptor_t* xDesc,
    const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t /*cxDesc*/, const void* /*cx*/,
    cudnnFilterDescriptor_t /*wDesc*/, const void* w,
    cudnnTensorDescriptor_t* yDesc,
    void* y,
    cudnnTensorDescriptor_t /*hyDesc*/, void* /*hy*/,
    cudnnTensorDescriptor_t /*cyDesc*/, void* /*cy*/,
    void* /*workspace*/, size_t /*workSpaceSizeInBytes*/)
{
    if (!rnnDesc || !xDesc || !x || !yDesc || !y || !w) return CUDNN_STATUS_INVALID_VALUE;

    auto* rd = (RNNDesc*)rnnDesc;
    auto* xt0 = (TensorDesc*)xDesc[0];
    auto* yt0 = (TensorDesc*)yDesc[0];
    int batch = xt0->n;
    int inputSize = xt0->c * xt0->h * xt0->w;
    int hiddenSize = rd->hiddenSize;

    const float* wf = (const float*)w;
    const float* hxf = (const float*)hx;
    const float* xf = (const float*)x;
    float* yf = (float*)y;

    int wihSize = hiddenSize * inputSize;
    int whhSize = hiddenSize * hiddenSize;
    const float* W_ih = wf;
    const float* W_hh = wf + wihSize;
    const float* b = wf + wihSize + whhSize;

    std::vector<float> hPrev(batch * hiddenSize, 0.f);
    if (hxf) {
        for (int i = 0; i < batch * hiddenSize; ++i) hPrev[i] = hxf[i];
    }

    for (int t = 0; t < seqLength; ++t) {
        const float* x_t = xf + t * batch * inputSize;
        float* y_t = yf + t * batch * hiddenSize;
        #ifdef _OPENMP
        #pragma omp parallel for if (batch > 4)
        #endif
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

    (void)hxDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardTraining(
    cudnnHandle_t handle,
    cudnnRNNDescriptor_t rnnDesc,
    int seqLength,
    cudnnTensorDescriptor_t* xDesc,
    const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t* yDesc,
    void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void* workspace, size_t workSpaceSizeInBytes,
    void* reserveSpace, size_t reserveSpaceSizeInBytes)
{
    (void)reserveSpace; (void)reserveSpaceSizeInBytes;
    return cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x,
        hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y,
        hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);
}

cudnnStatus_t cudnnRNNBackwardData(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    int seqLength,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    cudnnTensorDescriptor_t* dyDesc, const void* dy,
    cudnnTensorDescriptor_t /*dhyDesc*/, const void* /*dhy*/,
    cudnnTensorDescriptor_t /*dcyDesc*/, const void* /*dcy*/,
    cudnnFilterDescriptor_t /*wDesc*/, const void* w,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t /*cxDesc*/, const void* /*cx*/,
    cudnnTensorDescriptor_t* dxDesc, void* dx,
    cudnnTensorDescriptor_t /*dhxDesc*/, void* /*dhx*/,
    cudnnTensorDescriptor_t /*dcxDesc*/, void* /*dcx*/,
    void* /*workspace*/, size_t /*workSpaceSizeInBytes*/,
    void* /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    if (!rnnDesc || !yDesc || !dy || !dxDesc || !dx || !w) return CUDNN_STATUS_INVALID_VALUE;

    auto* rd = (RNNDesc*)rnnDesc;
    auto* yt0 = (TensorDesc*)yDesc[0];
    auto* xt0 = (TensorDesc*)dxDesc[0];
    int batch = yt0->n;
    int hiddenSize = rd->hiddenSize;
    int inputSize = xt0->c * xt0->h * xt0->w;

    const float* wf = (const float*)w;
    const float* yf = (const float*)y;
    const float* dyf = (const float*)dy;
    const float* hxf = (const float*)hx;
    float* dxf = (float*)dx;

    int wihSize = hiddenSize * inputSize;
    int whhSize = hiddenSize * hiddenSize;
    const float* W_ih = wf;
    const float* W_hh = wf + wihSize;

    std::vector<std::vector<float>> h(seqLength + 1, std::vector<float>(batch * hiddenSize, 0.f));
    if (hxf) {
        for (int i = 0; i < batch * hiddenSize; ++i) h[0][i] = hxf[i];
    }

    for (int t = 0; t < seqLength; ++t) {
        for (int n = 0; n < batch; ++n)
            for (int j = 0; j < hiddenSize; ++j)
                h[t+1][n*hiddenSize+j] = yf[t*batch*hiddenSize + n*hiddenSize + j];
    }

    std::vector<float> dh(batch * hiddenSize, 0.f);
    for (int t = seqLength - 1; t >= 0; --t) {
        const float* dy_t = dyf + t * batch * hiddenSize;
        float* dx_t = dxf + t * batch * inputSize;

        #ifdef _OPENMP
        #pragma omp parallel for if (batch > 4)
        #endif
        for (int n = 0; n < batch; ++n) {
            for (int j = 0; j < hiddenSize; ++j) {
                int idx = n * hiddenSize + j;
                dh[idx] += dy_t[idx];
                float hval = h[t+1][idx];
                float grad = dh[idx] * (1.0f - hval * hval);
                dh[idx] = grad;

                for (int k = 0; k < inputSize; ++k)
                    dx_t[n * inputSize + k] += W_ih[j * inputSize + k] * grad;
            }
        }

        std::vector<float> dhNext(batch * hiddenSize, 0.f);
        #ifdef _OPENMP
        #pragma omp parallel for if (batch > 4)
        #endif
        for (int n = 0; n < batch; ++n) {
            for (int j = 0; j < hiddenSize; ++j) {
                float grad = dh[n * hiddenSize + j];
                for (int k = 0; k < hiddenSize; ++k)
                    dhNext[n * hiddenSize + k] += W_hh[j * hiddenSize + k] * grad;
            }
        }
        dh = std::move(dhNext);
    }

    (void)hxDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeights(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    void* /*workspace*/, size_t /*workSpaceSizeInBytes*/,
    cudnnFilterDescriptor_t /*dwDesc*/, void* dw,
    void* /*reserveSpace*/, size_t /*reserveSpaceSizeInBytes*/)
{
    if (!rnnDesc || !xDesc || !x || !y || !dw) return CUDNN_STATUS_INVALID_VALUE;

    auto* rd = (RNNDesc*)rnnDesc;
    auto* xt0 = (TensorDesc*)xDesc[0];
    auto* yt0 = (TensorDesc*)yDesc[0];
    int batch = xt0->n;
    int inputSize = xt0->c * xt0->h * xt0->w;
    int hiddenSize = rd->hiddenSize;

    const float* xf = (const float*)x;
    const float* yf = (const float*)y;
    const float* hxf = (const float*)hx;
    float* dwf = (float*)dw;

    int wihSize = hiddenSize * inputSize;
    int whhSize = hiddenSize * hiddenSize;
    int totalWeights = wihSize + whhSize + hiddenSize;
    for (int i = 0; i < totalWeights; ++i) dwf[i] = 0.f;

    float* dW_ih = dwf;
    float* dW_hh = dwf + wihSize;
    float* db = dwf + wihSize + whhSize;

    std::vector<float> hPrev(batch * hiddenSize, 0.f);
    if (hxf) {
        for (int i = 0; i < batch * hiddenSize; ++i) hPrev[i] = hxf[i];
    }

    for (int t = 0; t < seqLength; ++t) {
        const float* x_t = xf + t * batch * inputSize;
        const float* y_t = yf + t * batch * hiddenSize;

        for (int n = 0; n < batch; ++n) {
            for (int j = 0; j < hiddenSize; ++j) {
                float hval = y_t[n * hiddenSize + j];
                float grad = 1.0f - hval * hval;

                db[j] += grad;
                for (int k = 0; k < inputSize; ++k)
                    dW_ih[j * inputSize + k] += grad * x_t[n * inputSize + k];
                for (int k = 0; k < hiddenSize; ++k)
                    dW_hh[j * hiddenSize + k] += grad * hPrev[n * hiddenSize + k];
            }
        }

        for (int i = 0; i < batch * hiddenSize; ++i) hPrev[i] = y_t[i];
    }

    (void)hxDesc;
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
