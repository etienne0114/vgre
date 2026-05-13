// cuDNN pooling forward + backward

#include "cudnn_internal.h"

extern "C" {

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

cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t,
    cudnnPoolingDescriptor_t pd,
    const void* alpha,
    cudnnTensorDescriptor_t yDesc, const void* y,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx)
{
    if (!pd || !yDesc || !dyDesc || !xDesc || !dxDesc || !y || !dy || !x || !dx || !alpha || !beta)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* p=(PoolDesc*)pd;
    auto* xt=(TensorDesc*)xDesc; auto* yt=(TensorDesc*)yDesc;
    auto* dyt=(TensorDesc*)dyDesc; auto* dxt=(TensorDesc*)dxDesc;
    float a=*(const float*)alpha, b=*(const float*)beta;
    const float* xf=(const float*)x;
    const float* dyf=(const float*)dy;
    float* dxf=(float*)dx;

    int outH=(xt->h+2*p->pad_h-p->win_h)/p->str_h+1;
    int outW=(xt->w+2*p->pad_w-p->win_w)/p->str_w+1;
    bool isMax=(p->mode==CUDNN_POOLING_MAX||p->mode==CUDNN_POOLING_MAX_DETERMINISTIC);
    bool excludePad=(p->mode==CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

    if (yt->n != xt->n || yt->c != xt->c || yt->h != outH || yt->w != outW)
        return CUDNN_STATUS_INVALID_VALUE;
    if (dyt->n != xt->n || dyt->c != xt->c || dyt->h != outH || dyt->w != outW)
        return CUDNN_STATUS_INVALID_VALUE;
    if (dxt->n != xt->n || dxt->c != xt->c || dxt->h != xt->h || dxt->w != xt->w)
        return CUDNN_STATUS_INVALID_VALUE;

    int xSize = xt->n * xt->c * xt->h * xt->w;
    if (b != 0.0f) {
        for (int i=0; i<xSize; ++i) dxf[i] = b * dxf[i];
    } else {
        std::memset(dxf, 0, xSize*sizeof(float));
    }

    std::vector<float> tmp(xSize, 0.f);

    for(int n=0; n<xt->n; ++n)
    for(int c=0; c<xt->c; ++c)
    for(int oh=0; oh<outH; ++oh)
    for(int ow=0; ow<outW; ++ow) {
        int oi=((n*xt->c+c)*outH+oh)*outW+ow;
        float dyVal = dyf[oi];

        if (isMax) {
            float maxv = -1e38f;
            int maxIh = -1, maxIw = -1;
            for(int kh=0; kh<p->win_h; ++kh)
            for(int kw=0; kw<p->win_w; ++kw) {
                int ih=oh*p->str_h-p->pad_h+kh;
                int iw=ow*p->str_w-p->pad_w+kw;
                if (ih<0 || ih>=xt->h || iw<0 || iw>=xt->w) continue;
                int xi=((n*xt->c+c)*xt->h+ih)*xt->w+iw;
                if (xf[xi] > maxv) {
                    maxv = xf[xi];
                    maxIh = ih; maxIw = iw;
                }
            }
            if (maxIh >= 0) {
                int xi=((n*xt->c+c)*xt->h+maxIh)*xt->w+maxIw;
                tmp[xi] += a * dyVal;
            }
        } else {
            int cnt=0;
            for(int kh=0; kh<p->win_h; ++kh)
            for(int kw=0; kw<p->win_w; ++kw) {
                int ih=oh*p->str_h-p->pad_h+kh;
                int iw=ow*p->str_w-p->pad_w+kw;
                if (ih>=0 && ih<xt->h && iw>=0 && iw<xt->w) ++cnt;
            }
            float divisor = (cnt>0) ? static_cast<float>(cnt) : 1.f;
            for(int kh=0; kh<p->win_h; ++kh)
            for(int kw=0; kw<p->win_w; ++kw) {
                int ih=oh*p->str_h-p->pad_h+kh;
                int iw=ow*p->str_w-p->pad_w+kw;
                if (ih<0 || ih>=xt->h || iw<0 || iw>=xt->w) continue;
                int xi=((n*xt->c+c)*xt->h+ih)*xt->w+iw;
                tmp[xi] += a * dyVal / divisor;
            }
        }
    }

    for (int i=0; i<xSize; ++i) dxf[i] += tmp[i];
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
