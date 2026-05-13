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
    int aTotal = at->n * at->c * at->h * at->w;
    int cTotal = ct->n * ct->c * ct->h * ct->w;
    if (cTotal != 1) return CUDNN_STATUS_NOT_SUPPORTED;

    float a = *(const float*)alpha;
    float b = *(const float*)beta;
    const float* Af = (const float*)A;
    float* Cf = (float*)C;

    float result = 0.f;
    switch (rd->op) {
    case CUDNN_REDUCE_TENSOR_ADD: {
        double sum = 0;
        for (int i = 0; i < aTotal; ++i) sum += Af[i];
        result = static_cast<float>(sum);
        break;
    }
    case CUDNN_REDUCE_TENSOR_MUL: {
        double prod = 1;
        for (int i = 0; i < aTotal; ++i) prod *= Af[i];
        result = static_cast<float>(prod);
        break;
    }
    case CUDNN_REDUCE_TENSOR_MIN: {
        result = Af[0];
        for (int i = 1; i < aTotal; ++i) result = std::min(result, Af[i]);
        break;
    }
    case CUDNN_REDUCE_TENSOR_MAX: {
        result = Af[0];
        for (int i = 1; i < aTotal; ++i) result = std::max(result, Af[i]);
        break;
    }
    case CUDNN_REDUCE_TENSOR_AMAX: {
        result = std::abs(Af[0]);
        for (int i = 1; i < aTotal; ++i) result = std::max(result, std::abs(Af[i]));
        break;
    }
    case CUDNN_REDUCE_TENSOR_AVG: {
        double sum = 0;
        for (int i = 0; i < aTotal; ++i) sum += Af[i];
        result = static_cast<float>(sum / aTotal);
        break;
    }
    case CUDNN_REDUCE_TENSOR_NORM1: {
        double sum = 0;
        for (int i = 0; i < aTotal; ++i) sum += std::abs(Af[i]);
        result = static_cast<float>(sum);
        break;
    }
    case CUDNN_REDUCE_TENSOR_NORM2: {
        double sum = 0;
        for (int i = 0; i < aTotal; ++i) sum += Af[i] * Af[i];
        result = static_cast<float>(std::sqrt(sum));
        break;
    }
    default: result = 0.f;
    }

    Cf[0] = a * result + b * Cf[0];
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
