// cuDNN softmax forward + backward

#include "cudnn_internal.h"

#include "vgre/common/openmp_helper.h"

extern "C" {

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

    if (mode == 0) {
        int VEC = C * HW;
        #ifdef _OPENMP
        #pragma omp parallel for if (N > 4)
        #endif
        for (int n=0; n<N; ++n) {
            const float* row = xf + n*VEC;
            float* out = yf + n*VEC;
            float maxv = *std::max_element(row, row+VEC);
            float sum = 0.f;
            for (int i=0; i<VEC; ++i) sum += expf(row[i] - maxv);
            for (int i=0; i<VEC; ++i) out[i] = a*(expf(row[i]-maxv)/sum) + b*out[i];
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) if (N * HW > 1024)
        #endif
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

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t,
    int algo, int mode,
    const void* alpha,
    cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx)
{
    if (!yDesc || !dyDesc || !dxDesc || !y || !dy || !dx || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* ty=(TensorDesc*)yDesc; auto* tdy=(TensorDesc*)dyDesc; auto* tdx=(TensorDesc*)dxDesc;
    int Ny = ty->n*ty->c*ty->h*ty->w;
    int Nd = tdy->n*tdy->c*tdy->h*tdy->w;
    int Ndx = tdx->n*tdx->c*tdx->h*tdx->w;
    if (Ny != Nd || Ny != Ndx) return CUDNN_STATUS_INVALID_VALUE;

    int N=ty->n, C=ty->c, HW=ty->h*ty->w;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* yf=(const float*)y;
    const float* dyf=(const float*)dy;
    float* dxf=(float*)dx;

    if (mode == 0) {
        int VEC = C * HW;
        #ifdef _OPENMP
        #pragma omp parallel for if (N > 4)
        #endif
        for (int n=0; n<N; ++n) {
            float sum_dy_y = 0.f;
            for (int i=0; i<VEC; ++i)
                sum_dy_y += dyf[n*VEC+i] * yf[n*VEC+i];
            for (int i=0; i<VEC; ++i) {
                float v = yf[n*VEC+i] * (dyf[n*VEC+i] - sum_dy_y);
                dxf[n*VEC+i] = a*v + b*dxf[n*VEC+i];
            }
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) if (N * HW > 1024)
        #endif
        for (int n=0; n<N; ++n)
        for (int hw=0; hw<HW; ++hw) {
            float sum_dy_y = 0.f;
            for (int c=0; c<C; ++c) {
                int idx = n*C*HW + c*HW + hw;
                sum_dy_y += dyf[idx] * yf[idx];
            }
            for (int c=0; c<C; ++c) {
                int idx = n*C*HW + c*HW + hw;
                float v = yf[idx] * (dyf[idx] - sum_dy_y);
                dxf[idx] = a*v + b*dxf[idx];
            }
        }
    }
    (void)algo;
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
