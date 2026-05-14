// cuDNN OpTensor, ReduceTensor, TransformTensor

#include "cudnn_internal.h"

extern "C" {

cudnnStatus_t cudnnOpTensor(
    cudnnHandle_t,
    cudnnOpTensorDescriptor_t opDesc,
    const void* alpha1,
    cudnnTensorDescriptor_t aDesc, const void* A,
    const void* alpha2,
    cudnnTensorDescriptor_t bDesc, const void* B,
    const void* beta,
    cudnnTensorDescriptor_t cDesc, void* C)
{
    if (!opDesc || !aDesc || !bDesc || !cDesc || !A || !B || !C || !alpha1 || !alpha2 || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* od = (OpTensorDesc*)opDesc;
    auto* at = (TensorDesc*)aDesc;
    auto* bt = (TensorDesc*)bDesc;
    auto* ct = (TensorDesc*)cDesc;

    int nVal = at->n, cVal = at->c, hVal = at->h, wVal = at->w;
    int total = nVal * cVal * hVal * wVal;
    if (bt->n*bt->c*bt->h*bt->w != total || ct->n*ct->c*ct->h*ct->w != total)
        return CUDNN_STATUS_INVALID_VALUE;

    float a1 = *(const float*)alpha1;
    float a2 = *(const float*)alpha2;
    float b  = *(const float*)beta;
    const float* Af = (const float*)A;
    const float* Bf = (const float*)B;
    float* Cf = (float*)C;

    for (int i = 0; i < total; ++i) {
        float val = 0.f;
        switch (od->op) {
        case CUDNN_OP_TENSOR_ADD: val = a1*Af[i] + a2*Bf[i]; break;
        case CUDNN_OP_TENSOR_MUL: val = (a1*Af[i]) * (a2*Bf[i]); break;
        case CUDNN_OP_TENSOR_MIN: val = std::min(a1*Af[i], a2*Bf[i]); break;
        case CUDNN_OP_TENSOR_MAX: val = std::max(a1*Af[i], a2*Bf[i]); break;
        case CUDNN_OP_TENSOR_SQRT: val = a1*std::sqrt(std::abs(Af[i])); break;
        case CUDNN_OP_TENSOR_NOT:  val = a1*(Af[i] == 0.f ? 1.f : 0.f); break;
        default: val = 0.f;
        }
        Cf[i] = val + b*Cf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(
    cudnnHandle_t,
    cudnnReduceTensorDescriptor_t,
    cudnnTensorDescriptor_t,
    cudnnTensorDescriptor_t,
    size_t* sizeInBytes)
{
    if (!sizeInBytes) return CUDNN_STATUS_INVALID_VALUE;
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor(
    cudnnHandle_t,
    cudnnReduceTensorDescriptor_t reduceDesc,
    void* /*indices*/, size_t /*indicesSizeInBytes*/,
    void* /*workspace*/, size_t /*workspaceSizeInBytes*/,
    const void* alpha,
    cudnnTensorDescriptor_t aDesc, const void* A,
    const void* beta,
    cudnnTensorDescriptor_t cDesc, void* C)
{
    if (!reduceDesc || !aDesc || !cDesc || !A || !C || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* rd = (ReduceTensorDesc*)reduceDesc;
    auto* at = (TensorDesc*)aDesc;
    auto* ct = (TensorDesc*)cDesc;

    float a = *(const float*)alpha;
    float b = *(const float*)beta;
    const float* Af = (const float*)A;
    float* Cf = (float*)C;

    // Determine which dimensions are reduced (output dim == 1, input dim > 1)
    bool reduceN = (ct->n == 1 && at->n > 1);
    bool reduceC = (ct->c == 1 && at->c > 1);
    bool reduceH = (ct->h == 1 && at->h > 1);
    bool reduceW = (ct->w == 1 && at->w > 1);

    // Per-output-element reduction
    for (int on = 0; on < ct->n; ++on)
    for (int oc = 0; oc < ct->c; ++oc)
    for (int oh = 0; oh < ct->h; ++oh)
    for (int ow = 0; ow < ct->w; ++ow) {
        float result = 0.f;
        int count = 0;

        // Determine input ranges for this output element
        int nStart = reduceN ? 0 : on;
        int nEnd   = reduceN ? at->n : on + 1;
        int cStart = reduceC ? 0 : oc;
        int cEnd   = reduceC ? at->c : oc + 1;
        int hStart = reduceH ? 0 : oh;
        int hEnd   = reduceH ? at->h : oh + 1;
        int wStart = reduceW ? 0 : ow;
        int wEnd   = reduceW ? at->w : ow + 1;

        switch (rd->op) {
        case CUDNN_REDUCE_TENSOR_ADD: {
            double sum = 0;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                sum += Af[((n * at->c + c) * at->h + h) * at->w + w];
                ++count;
            }
            result = static_cast<float>(sum);
            break;
        }
        case CUDNN_REDUCE_TENSOR_MUL: {
            double prod = 1;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                prod *= Af[((n * at->c + c) * at->h + h) * at->w + w];
                ++count;
            }
            result = static_cast<float>(prod);
            break;
        }
        case CUDNN_REDUCE_TENSOR_MIN: {
            bool first = true;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                float v = Af[((n * at->c + c) * at->h + h) * at->w + w];
                if (first) { result = v; first = false; }
                else result = std::min(result, v);
                ++count;
            }
            break;
        }
        case CUDNN_REDUCE_TENSOR_MAX: {
            bool first = true;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                float v = Af[((n * at->c + c) * at->h + h) * at->w + w];
                if (first) { result = v; first = false; }
                else result = std::max(result, v);
                ++count;
            }
            break;
        }
        case CUDNN_REDUCE_TENSOR_AMAX: {
            bool first = true;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                float v = std::abs(Af[((n * at->c + c) * at->h + h) * at->w + w]);
                if (first) { result = v; first = false; }
                else result = std::max(result, v);
                ++count;
            }
            break;
        }
        case CUDNN_REDUCE_TENSOR_AVG: {
            double sum = 0;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                sum += Af[((n * at->c + c) * at->h + h) * at->w + w];
                ++count;
            }
            result = static_cast<float>(sum / count);
            break;
        }
        case CUDNN_REDUCE_TENSOR_NORM1: {
            double sum = 0;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                sum += std::abs(Af[((n * at->c + c) * at->h + h) * at->w + w]);
                ++count;
            }
            result = static_cast<float>(sum);
            break;
        }
        case CUDNN_REDUCE_TENSOR_NORM2: {
            double sum = 0;
            for (int n = nStart; n < nEnd; ++n)
            for (int c = cStart; c < cEnd; ++c)
            for (int h = hStart; h < hEnd; ++h)
            for (int w = wStart; w < wEnd; ++w) {
                float v = Af[((n * at->c + c) * at->h + h) * at->w + w];
                sum += v * v;
                ++count;
            }
            result = static_cast<float>(std::sqrt(sum));
            break;
        }
        default: result = 0.f;
        }

        int cIdx = ((on * ct->c + oc) * ct->h + oh) * ct->w + ow;
        Cf[cIdx] = a * result + b * Cf[cIdx];
    }

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor(
    cudnnHandle_t,
    const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    cudnnTensorDescriptor_t yDesc, void* y)
{
    if (!xDesc || !yDesc || !x || !y || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* xt = (TensorDesc*)xDesc;
    auto* yt = (TensorDesc*)yDesc;
    int xTotal = xt->n * xt->c * xt->h * xt->w;
    int yTotal = yt->n * yt->c * yt->h * yt->w;
    if (xTotal != yTotal) return CUDNN_STATUS_INVALID_VALUE;

    float a = *(const float*)alpha;
    float b = *(const float*)beta;
    const float* xf = (const float*)x;
    float* yf = (float*)y;

    for (int i = 0; i < xTotal; ++i)
        yf[i] = a * xf[i] + b * yf[i];

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
