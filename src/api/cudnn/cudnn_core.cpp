// cuDNN core: handle, tensor/filter/conv/pooling/activation descriptors

#include "cudnn_internal.h"

extern "C" {

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
    if (!handle) return CUDNN_STATUS_INVALID_VALUE;
    *handle = new HandleCtx();
    VGRE_LOG_DEBUG("cuDNN", "cudnnCreate");
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroy(cudnnHandle_t h) { delete (HandleCtx*)h; return CUDNN_STATUS_SUCCESS; }
cudnnStatus_t cudnnGetVersion(unsigned long long* v) {
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

// ── Activation descriptors ────────────────────────────────────────────────────
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

// ── Dropout descriptors ───────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new DropoutDesc{};
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t desc) {
    auto* d = (DropoutDesc*)desc;
    if (d && d->states) { std::free(d->states); }
    delete d;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetDropoutDescriptor(
    cudnnDropoutDescriptor_t desc,
    cudnnHandle_t,
    float dropout,
    void* states,
    size_t stateSizeInBytes,
    unsigned long long seed)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = (DropoutDesc*)desc;
    d->dropout = dropout;
    d->seed = seed;
    d->states = states;
    d->statesSize = stateSizeInBytes;
    return CUDNN_STATUS_SUCCESS;
}

// ── OpTensor descriptors ──────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new OpTensorDesc{};
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t desc) {
    delete (OpTensorDesc*)desc;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t desc,
    cudnnOpTensorOp_t op,
    cudnnDataType_t compType,
    cudnnNanPropagation_t nanOpt)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = (OpTensorDesc*)desc;
    d->op = op; d->compType = compType; d->nanOpt = nanOpt;
    return CUDNN_STATUS_SUCCESS;
}

// ── ReduceTensor descriptors ─────────────────────────────────────────────────
cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new ReduceTensorDesc{};
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t desc) {
    delete (ReduceTensorDesc*)desc;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t desc,
    cudnnReduceTensorOp_t op,
    cudnnDataType_t compType,
    cudnnNanPropagation_t nanOpt,
    cudnnReduceTensorOp_t /*idxCompType*/,
    cudnnDataType_t /*outType*/)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = (ReduceTensorDesc*)desc;
    d->op = op; d->compType = compType; d->nanOpt = nanOpt;
    return CUDNN_STATUS_SUCCESS;
}

// ── RNN descriptors ────────────────────────────────────────────────────────────
cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new RNNDesc{};
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t desc) {
    delete (RNNDesc*)desc;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetRNNDescriptor(
    cudnnRNNDescriptor_t desc,
    int hiddenSize,
    int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t /*inputMode*/,
    cudnnDirectionMode_t /*direction*/,
    cudnnRNNMode_t mode,
    cudnnDataType_t /*dataType*/)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = (RNNDesc*)desc;
    d->hiddenSize = hiddenSize;
    d->numLayers = numLayers;
    d->mode = mode;
    d->dropoutDesc = dropoutDesc;
    return CUDNN_STATUS_SUCCESS;
}

// QUEUE-38: Peephole LSTM — enable/disable diagonal c→gate connections.
// When enabled: i/f gates see c_{t-1} via p_i/p_f; o gate sees c_t via p_o.
// Peephole weights occupy 3*H floats appended after standard weights per layer.
cudnnStatus_t cudnnRNNSetPeephole(cudnnRNNDescriptor_t desc, int enable) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    static_cast<RNNDesc*>(desc)->peephole = (enable != 0);
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnRNNGetPeephole(cudnnRNNDescriptor_t desc, int* enable) {
    if (!desc || !enable) return CUDNN_STATUS_INVALID_VALUE;
    *enable = static_cast<RNNDesc*>(desc)->peephole ? 1 : 0;
    return CUDNN_STATUS_SUCCESS;
}

// ── Multi-Head Attention descriptors ─────────────────────────────────────────
cudnnStatus_t cudnnCreateMultiHeadAttnDescriptor(cudnnMultiHeadAttnDescriptor_t* desc) {
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    *desc = new MultiHeadAttnDesc{};
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnDestroyMultiHeadAttnDescriptor(cudnnMultiHeadAttnDescriptor_t desc) {
    delete (MultiHeadAttnDesc*)desc;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t cudnnSetMultiHeadAttnDescriptor(
    cudnnMultiHeadAttnDescriptor_t desc,
    cudnnMultiHeadAttnMode_t /*mode*/,
    int numHeads, int headDim, int modelDim,
    int /*seqLen*/, int /*batchSize*/,
    float scaling)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    auto* d = (MultiHeadAttnDesc*)desc;
    d->numHeads = numHeads;
    d->headDim = headDim;
    d->modelDim = modelDim;
    d->scaling = scaling;
    return CUDNN_STATUS_SUCCESS;
}

// ── Descriptor getter functions ───────────────────────────────────────────────
// These mirror the Set functions and allow callers to query descriptor state.

cudnnStatus_t cudnnGetFilter4dDescriptor(
    cudnnFilterDescriptor_t d,
    cudnnDataType_t *dtype, cudnnTensorFormat_t *fmt,
    int *k, int *c, int *r, int *s)
{
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    const auto* f = (const FilterDesc*)d;
    if (dtype) *dtype = f->dtype;
    if (fmt)   *fmt   = f->fmt;
    if (k)     *k     = f->k;
    if (c)     *c     = f->c;
    if (r)     *r     = f->r;
    if (s)     *s     = f->s;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t d,
    int *padH, int *padW, int *strH, int *strW,
    int *dilH, int *dilW, int *mode, cudnnDataType_t *computeType)
{
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    const auto* c = (const ConvDesc*)d;
    if (padH)        *padH        = c->pad_h;
    if (padW)        *padW        = c->pad_w;
    if (strH)        *strH        = c->str_h;
    if (strW)        *strW        = c->str_w;
    if (dilH)        *dilH        = c->dil_h;
    if (dilW)        *dilW        = c->dil_w;
    if (mode)        *mode        = c->mode;
    if (computeType) *computeType = CUDNN_DATA_FLOAT;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetActivationDescriptor(
    cudnnActivationDescriptor_t d,
    cudnnActivationMode_t *mode, cudnnNanPropagation_t *nanOpt, double *coeff)
{
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    const auto* a = (const ActDesc*)d;
    if (mode)   *mode   = a->mode;
    if (nanOpt) *nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    if (coeff)  *coeff  = a->coeff;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t desc,
    cudnnOpTensorOp_t *op, cudnnDataType_t *compType, cudnnNanPropagation_t *nanOpt)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    const auto* d = (const OpTensorDesc*)desc;
    if (op)       *op       = d->op;
    if (compType) *compType = d->compType;
    if (nanOpt)   *nanOpt   = d->nanOpt;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t desc,
    cudnnReduceTensorOp_t *op, cudnnDataType_t *compType,
    cudnnNanPropagation_t *nanOpt,
    cudnnReduceTensorOp_t *idxCompType, cudnnDataType_t *outType)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    const auto* d = (const ReduceTensorDesc*)desc;
    if (op)          *op          = d->op;
    if (compType)    *compType    = d->compType;
    if (nanOpt)      *nanOpt      = d->nanOpt;
    if (idxCompType) *idxCompType = d->op;    // mirror op for index type
    if (outType)     *outType     = d->compType;
    return CUDNN_STATUS_SUCCESS;
}

// ── Convolution group count + math type ──────────────────────────────────────

cudnnStatus_t cudnnSetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t d, int groupCount)
{
    if (!d || groupCount < 1) return CUDNN_STATUS_INVALID_VALUE;
    ((ConvDesc*)d)->groupCount = groupCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(
    cudnnConvolutionDescriptor_t d, int *groupCount)
{
    if (!d || !groupCount) return CUDNN_STATUS_INVALID_VALUE;
    *groupCount = ((ConvDesc*)d)->groupCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t d, int mathType)
{
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    ((ConvDesc*)d)->mathType = mathType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolutionMathType(
    cudnnConvolutionDescriptor_t d, int *mathType)
{
    if (!d || !mathType) return CUDNN_STATUS_INVALID_VALUE;
    *mathType = ((ConvDesc*)d)->mathType;
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnGetDropoutDescriptor ─────────────────────────────────────────────────

cudnnStatus_t cudnnGetDropoutDescriptor(
    cudnnDropoutDescriptor_t desc,
    cudnnHandle_t /*handle*/,
    float *dropout, void **states, unsigned long long *seed)
{
    if (!desc) return CUDNN_STATUS_INVALID_VALUE;
    const auto* d = (const DropoutDesc*)desc;
    if (dropout) *dropout = d->dropout;
    if (states)  *states  = d->states;
    if (seed)    *seed    = d->seed;
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
