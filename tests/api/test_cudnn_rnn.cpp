/**
 * Test: cuDNN RNN Forward/Backward (P0-1 fix)
 *
 * Numerically verifies LSTM, GRU, and vanilla RNN against reference implementations.
 * Tests forward inference, forward training, BPTT (BackwardData), and weight gradients.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnRNNDescriptor_t;
typedef void* cudnnDropoutDescriptor_t;

enum cudnnDataType_t    { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };
enum cudnnRNNMode_t     { CUDNN_RNN_RELU=0, CUDNN_RNN_TANH=1, CUDNN_LSTM=2, CUDNN_GRU=3 };
enum cudnnRNNInputMode_t { CUDNN_LINEAR_INPUT = 0 };
enum cudnnDirectionMode_t{ CUDNN_UNIDIRECTIONAL = 0 };

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
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnDataType_t dtype);
cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    int seqLength, cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    int seqLength, cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes, cudnnDataType_t);
cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t, cudnnRNNDescriptor_t, int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void* workspace, size_t workSpaceSizeInBytes);
cudnnStatus_t cudnnRNNForwardTraining(
    cudnnHandle_t, cudnnRNNDescriptor_t, int seqLength,
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
    cudnnHandle_t, cudnnRNNDescriptor_t, int seqLength,
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
    cudnnHandle_t, cudnnRNNDescriptor_t, int seqLength,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    void* workspace, size_t workSpaceSizeInBytes,
    cudnnFilterDescriptor_t dwDesc, void* dw,
    void* reserveSpace, size_t reserveSpaceSizeInBytes);
}

// ── Reference implementations ─────────────────────────────────────────────────

static inline float ref_sig(float x) { return 1.f / (1.f + expf(-x)); }

// Reference vanilla RNN forward
// Weight layout: [W_ih: H×I | W_hh: H×H | b: H]
static void ref_rnn(int T, int B, int I, int H, const float* x, const float* hx,
                    const float* w, float* y)
{
    const float* W_ih=w, *W_hh=w+H*I, *b=w+H*I+H*H;
    std::vector<float> hp(B*H,0.f);
    if (hx) memcpy(hp.data(),hx,B*H*sizeof(float));
    for (int t=0;t<T;++t) {
        const float* xt=x+t*B*I; float* yt=y+t*B*H;
        for (int n=0;n<B;++n)
            for (int j=0;j<H;++j) {
                float s=b[j];
                for(int k=0;k<I;++k) s+=W_ih[j*I+k]*xt[n*I+k];
                for(int k=0;k<H;++k) s+=W_hh[j*H+k]*hp[n*H+k];
                yt[n*H+j]=tanhf(s);
            }
        memcpy(hp.data(),yt,B*H*sizeof(float));
    }
}

// Reference LSTM forward
// Weight layout: [W_i_ih:H×I|W_f_ih:H×I|W_g_ih:H×I|W_o_ih:H×I |
//                 W_i_hh:H×H|...|W_o_hh:H×H | b_ih:4H | b_hh:4H]
static void ref_lstm(int T, int B, int I, int H, const float* x, const float* hx,
                     const float* cx, const float* w, float* y, float* cy_out)
{
    const float* W_ih=w, *W_hh=w+4*H*I, *b_ih=w+4*H*I+4*H*H, *b_hh=b_ih+4*H;
    std::vector<float> hp(B*H,0.f), cp(B*H,0.f);
    if (hx) memcpy(hp.data(),hx,B*H*sizeof(float));
    if (cx) memcpy(cp.data(),cx,B*H*sizeof(float));
    for (int t=0;t<T;++t) {
        const float* xt=x+t*B*I; float* yt=y+t*B*H;
        for (int n=0;n<B;++n)
            for (int j=0;j<H;++j) {
                float s0=b_ih[0*H+j]+b_hh[0*H+j];  // i pre-act
                float s1=b_ih[1*H+j]+b_hh[1*H+j];  // f
                float s2=b_ih[2*H+j]+b_hh[2*H+j];  // g
                float s3=b_ih[3*H+j]+b_hh[3*H+j];  // o
                for(int k=0;k<I;++k) {
                    s0+=W_ih[(0*H+j)*I+k]*xt[n*I+k];
                    s1+=W_ih[(1*H+j)*I+k]*xt[n*I+k];
                    s2+=W_ih[(2*H+j)*I+k]*xt[n*I+k];
                    s3+=W_ih[(3*H+j)*I+k]*xt[n*I+k];
                }
                for(int k=0;k<H;++k) {
                    s0+=W_hh[(0*H+j)*H+k]*hp[n*H+k];
                    s1+=W_hh[(1*H+j)*H+k]*hp[n*H+k];
                    s2+=W_hh[(2*H+j)*H+k]*hp[n*H+k];
                    s3+=W_hh[(3*H+j)*H+k]*hp[n*H+k];
                }
                float iv=ref_sig(s0), fv=ref_sig(s1), gv=tanhf(s2), ov=ref_sig(s3);
                float cn=fv*cp[n*H+j]+iv*gv;
                yt[n*H+j]=ov*tanhf(cn);
                cp[n*H+j]=cn;
            }
        memcpy(hp.data(),yt,B*H*sizeof(float));
    }
    if (cy_out) memcpy(cy_out,cp.data(),B*H*sizeof(float));
}

// Reference GRU forward
// Weight layout: [W_r_ih:H×I|W_z_ih:H×I|W_n_ih:H×I |
//                 W_r_hh:H×H|W_z_hh:H×H|W_n_hh:H×H | b_ih:3H | b_hh:3H]
static void ref_gru(int T, int B, int I, int H, const float* x, const float* hx,
                    const float* w, float* y)
{
    const float* W_ih=w, *W_hh=w+3*H*I, *b_ih=w+3*H*I+3*H*H, *b_hh=b_ih+3*H;
    std::vector<float> hp(B*H,0.f);
    if (hx) memcpy(hp.data(),hx,B*H*sizeof(float));
    for (int t=0;t<T;++t) {
        const float* xt=x+t*B*I; float* yt=y+t*B*H;
        for (int n=0;n<B;++n)
            for (int j=0;j<H;++j) {
                float sr=b_ih[0*H+j]+b_hh[0*H+j];
                float sz=b_ih[1*H+j]+b_hh[1*H+j];
                float snx=b_ih[2*H+j], snh=b_hh[2*H+j];
                for(int k=0;k<I;++k) {
                    sr+=W_ih[(0*H+j)*I+k]*xt[n*I+k];
                    sz+=W_ih[(1*H+j)*I+k]*xt[n*I+k];
                    snx+=W_ih[(2*H+j)*I+k]*xt[n*I+k];
                }
                for(int k=0;k<H;++k) {
                    sr+=W_hh[(0*H+j)*H+k]*hp[n*H+k];
                    sz+=W_hh[(1*H+j)*H+k]*hp[n*H+k];
                    snh+=W_hh[(2*H+j)*H+k]*hp[n*H+k];
                }
                float rv=ref_sig(sr), zv=ref_sig(sz), nv=tanhf(snx+rv*snh);
                yt[n*H+j]=(1.f-zv)*nv+zv*hp[n*H+j];
            }
        memcpy(hp.data(),yt,B*H*sizeof(float));
    }
}

static bool approx_eq(float a, float b, float eps=1e-3f) {
    if (std::isnan(a)||std::isnan(b)) return false;
    if (std::isinf(a)&&std::isinf(b)) return (a>0)==(b>0);
    return std::abs(a-b) <= eps+eps*std::abs(b);
}

// ── Test helpers ──────────────────────────────────────────────────────────────

struct RNNTest {
    cudnnHandle_t handle;
    int T, B, I, H, nL;
    cudnnRNNMode_t mode;

    // Built during setup
    std::vector<cudnnTensorDescriptor_t> xDesc, yDesc;
    cudnnTensorDescriptor_t hxDesc;
    cudnnRNNDescriptor_t rnnDesc;
    size_t rsvBytes, wsBytes;
    std::vector<float> x, hx, cx, w, y, hy, cy;
    size_t wCount;

    RNNTest(cudnnHandle_t h, int T, int B, int I, int H, int nL, cudnnRNNMode_t mode)
        : handle(h), T(T), B(B), I(I), H(H), nL(nL), mode(mode)
    {
        // Create descriptors
        xDesc.resize(T); yDesc.resize(T);
        for (int t=0;t<T;++t) {
            cudnnCreateTensorDescriptor(&xDesc[t]);
            cudnnSetTensor4dDescriptor(xDesc[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,I,1,1);
            cudnnCreateTensorDescriptor(&yDesc[t]);
            cudnnSetTensor4dDescriptor(yDesc[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,H,1,1);
        }
        cudnnCreateTensorDescriptor(&hxDesc);
        cudnnSetTensor4dDescriptor(hxDesc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,nL*B,H,1,1);
        cudnnCreateRNNDescriptor(&rnnDesc);
        cudnnSetRNNDescriptor(rnnDesc,H,nL,nullptr,CUDNN_LINEAR_INPUT,CUDNN_UNIDIRECTIONAL,mode,CUDNN_DATA_FLOAT);

        // Query sizes
        cudnnGetRNNTrainingReserveSize(handle,rnnDesc,T,xDesc.data(),&rsvBytes);
        cudnnGetRNNWorkspaceSize(handle,rnnDesc,T,xDesc.data(),&wsBytes);
        size_t wBytes=0;
        cudnnGetRNNParamsSize(handle,rnnDesc,xDesc[0],&wBytes,CUDNN_DATA_FLOAT);
        wCount=wBytes/sizeof(float);

        // Allocate buffers
        x.resize(T*B*I); hx.resize(nL*B*H,0.f); cx.resize(nL*B*H,0.f);
        w.resize(wCount); y.resize(T*B*H,0.f); hy.resize(nL*B*H,0.f); cy.resize(nL*B*H,0.f);

        // Fill with deterministic values (cast to int to avoid size_t underflow)
        for (size_t i=0;i<x.size();++i) x[i]=(float)((int)(i%7)-3)*0.1f;
        for (size_t i=0;i<w.size();++i) w[i]=(float)((int)(i%5)-2)*0.05f;
    }

    ~RNNTest() {
        for (int t=0;t<T;++t) {
            cudnnDestroyTensorDescriptor(xDesc[t]);
            cudnnDestroyTensorDescriptor(yDesc[t]);
        }
        cudnnDestroyTensorDescriptor(hxDesc);
        cudnnDestroyRNNDescriptor(rnnDesc);
    }
};

int main() {
    std::cout << "=== Test: cuDNN RNN Forward/Backward (P3.9) ===\n";
    int pass=0, total=0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n"; return 1;
    }

    // ── Test 1: Reserve/workspace size API ──────────────────────────────
    ++total;
    {
        int T=3, B=2, I=4, H=3;
        std::vector<cudnnTensorDescriptor_t> xD(T), yD(T);
        for (int t=0;t<T;++t) {
            cudnnCreateTensorDescriptor(&xD[t]);
            cudnnSetTensor4dDescriptor(xD[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,I,1,1);
            cudnnCreateTensorDescriptor(&yD[t]);
            cudnnSetTensor4dDescriptor(yD[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,H,1,1);
        }
        cudnnRNNDescriptor_t rd; cudnnCreateRNNDescriptor(&rd);
        cudnnSetRNNDescriptor(rd,H,1,nullptr,CUDNN_LINEAR_INPUT,CUDNN_UNIDIRECTIONAL,CUDNN_RNN_TANH,CUDNN_DATA_FLOAT);
        size_t ws=99, rs=99;
        auto r1=cudnnGetRNNWorkspaceSize(handle,rd,T,xD.data(),&ws);
        auto r2=cudnnGetRNNTrainingReserveSize(handle,rd,T,xD.data(),&rs);
        // ws must be 0 (we use reserve, not workspace); rs must be > 0 (real reserve for BPTT)
        bool ok=(r1==CUDNN_STATUS_SUCCESS && r2==CUDNN_STATUS_SUCCESS && ws==0 && rs>0);
        if (ok) { std::cout<<"PASS [Reserve/Workspace sizes]\n"; ++pass; }
        else std::cerr<<"FAIL [Workspace/Reserve] r1="<<r1<<" r2="<<r2<<" ws="<<ws<<" rs="<<rs<<"\n";
        for(int t=0;t<T;++t){cudnnDestroyTensorDescriptor(xD[t]);cudnnDestroyTensorDescriptor(yD[t]);}
        cudnnDestroyRNNDescriptor(rd);
    }

    // ── Test 2: Vanilla RNN forward inference vs reference ───────────────
    ++total;
    {
        int T=3, B=2, I=4, H=3;
        int wIH=H*I, wHH=H*H, wb=H, wTot=wIH+wHH+wb;
        std::vector<float> x(T*B*I), hx(B*H,0.f), w(wTot), y(T*B*H), refY(T*B*H);
        for (size_t i=0;i<x.size();++i) x[i]=(float)((int)i%5-2)*0.1f;
        for (int i=0;i<wTot;++i) w[i]=(float)(i%3-1)*0.1f;

        std::vector<cudnnTensorDescriptor_t> xD(T), yD(T);
        for (int t=0;t<T;++t) {
            cudnnCreateTensorDescriptor(&xD[t]);
            cudnnSetTensor4dDescriptor(xD[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,I,1,1);
            cudnnCreateTensorDescriptor(&yD[t]);
            cudnnSetTensor4dDescriptor(yD[t],CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,H,1,1);
        }
        cudnnTensorDescriptor_t hxD; cudnnCreateTensorDescriptor(&hxD);
        cudnnSetTensor4dDescriptor(hxD,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,H,1,1);
        cudnnRNNDescriptor_t rd; cudnnCreateRNNDescriptor(&rd);
        cudnnSetRNNDescriptor(rd,H,1,nullptr,CUDNN_LINEAR_INPUT,CUDNN_UNIDIRECTIONAL,CUDNN_RNN_TANH,CUDNN_DATA_FLOAT);

        auto r=cudnnRNNForwardInference(handle,rd,T,xD.data(),x.data(),
            hxD,hx.data(),nullptr,nullptr,nullptr,w.data(),yD.data(),y.data(),
            nullptr,nullptr,nullptr,nullptr,nullptr,0);
        ref_rnn(T,B,I,H,x.data(),hx.data(),w.data(),refY.data());

        bool ok=(r==CUDNN_STATUS_SUCCESS);
        for (size_t i=0;i<y.size()&&ok;++i)
            if (!approx_eq(y[i],refY[i])) {
                std::cerr<<"FAIL [RNN Fwd] y["<<i<<"]="<<y[i]<<" ref="<<refY[i]<<"\n"; ok=false;
            }
        if (ok) { std::cout<<"PASS [RNN forward inference]\n"; ++pass; }

        for(int t=0;t<T;++t){cudnnDestroyTensorDescriptor(xD[t]);cudnnDestroyTensorDescriptor(yD[t]);}
        cudnnDestroyTensorDescriptor(hxD); cudnnDestroyRNNDescriptor(rd);
    }

    // ── Test 3: LSTM forward vs reference (core P0-1 fix) ───────────────
    ++total;
    {
        int T=4, B=2, I=3, H=4;
        RNNTest rt(handle,T,B,I,H,1,CUDNN_LSTM);

        // Compute reference
        std::vector<float> refY(T*B*H,0.f), refCY(B*H,0.f);
        ref_lstm(T,B,I,H,rt.x.data(),rt.hx.data(),rt.cx.data(),rt.w.data(),refY.data(),refCY.data());

        // Run our implementation
        auto r=cudnnRNNForwardInference(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),rt.hxDesc,rt.cx.data(),
            nullptr,rt.w.data(),rt.yDesc.data(),rt.y.data(),
            nullptr,nullptr,nullptr,nullptr,nullptr,0);

        bool ok=(r==CUDNN_STATUS_SUCCESS);
        for (size_t i=0;i<rt.y.size()&&ok;++i)
            if (!approx_eq(rt.y[i],refY[i],1e-4f)) {
                std::cerr<<"FAIL [LSTM Fwd] y["<<i<<"]="<<rt.y[i]<<" ref="<<refY[i]<<"\n"; ok=false;
            }
        if (ok) { std::cout<<"PASS [LSTM forward (P0-1)]\n"; ++pass; }
    }

    // ── Test 4: GRU forward vs reference (core P0-1 fix) ────────────────
    ++total;
    {
        int T=4, B=2, I=3, H=4;
        RNNTest rt(handle,T,B,I,H,1,CUDNN_GRU);

        std::vector<float> refY(T*B*H,0.f);
        ref_gru(T,B,I,H,rt.x.data(),rt.hx.data(),rt.w.data(),refY.data());

        auto r=cudnnRNNForwardInference(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),nullptr,nullptr,
            nullptr,rt.w.data(),rt.yDesc.data(),rt.y.data(),
            nullptr,nullptr,nullptr,nullptr,nullptr,0);

        bool ok=(r==CUDNN_STATUS_SUCCESS);
        for (size_t i=0;i<rt.y.size()&&ok;++i)
            if (!approx_eq(rt.y[i],refY[i],1e-4f)) {
                std::cerr<<"FAIL [GRU Fwd] y["<<i<<"]="<<rt.y[i]<<" ref="<<refY[i]<<"\n"; ok=false;
            }
        if (ok) { std::cout<<"PASS [GRU forward (P0-1)]\n"; ++pass; }
    }

    // ── Test 5: ForwardTraining == ForwardInference (with reserve) ───────
    ++total;
    {
        int T=3, B=2, I=4, H=3;
        RNNTest rt(handle,T,B,I,H,1,CUDNN_RNN_TANH);
        std::vector<float> yTrain(T*B*H,0.f), rsv(rt.rsvBytes/sizeof(float)+1,0.f);

        // Inference
        cudnnRNNForwardInference(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),nullptr,nullptr,nullptr,rt.w.data(),
            rt.yDesc.data(),rt.y.data(),nullptr,nullptr,nullptr,nullptr,nullptr,0);
        // Training
        auto r=cudnnRNNForwardTraining(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),nullptr,nullptr,nullptr,rt.w.data(),
            rt.yDesc.data(),yTrain.data(),nullptr,nullptr,nullptr,nullptr,nullptr,0,
            rsv.data(),rt.rsvBytes);

        bool ok=(r==CUDNN_STATUS_SUCCESS);
        for (size_t i=0;i<rt.y.size()&&ok;++i)
            if (!approx_eq(yTrain[i],rt.y[i])) {
                std::cerr<<"FAIL [Training==Inference] i="<<i<<" train="<<yTrain[i]<<" infer="<<rt.y[i]<<"\n"; ok=false;
            }
        if (ok) { std::cout<<"PASS [ForwardTraining == ForwardInference]\n"; ++pass; }
    }

    // ── Test 6: Full LSTM backward cycle (gradient non-zero + no crash) ──
    ++total;
    {
        int T=3, B=2, I=3, H=4;
        RNNTest rt(handle,T,B,I,H,1,CUDNN_LSTM);
        std::vector<float> rsv(rt.rsvBytes/sizeof(float)+1,0.f);
        std::vector<float> dy(T*B*H,1.f);  // dL/dy = 1 everywhere
        std::vector<float> dx(T*B*I,0.f), dhx(B*H,0.f), dcx(B*H,0.f);
        std::vector<float> dw(rt.wCount,0.f);

        // Forward training (fills reserve)
        auto r1=cudnnRNNForwardTraining(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),rt.hxDesc,rt.cx.data(),nullptr,rt.w.data(),
            rt.yDesc.data(),rt.y.data(),nullptr,nullptr,nullptr,nullptr,nullptr,0,
            rsv.data(),rt.rsvBytes);

        // Backward data (fills reserve Part C with dpre)
        auto r2=cudnnRNNBackwardData(handle,rt.rnnDesc,T,
            rt.yDesc.data(),rt.y.data(),rt.yDesc.data(),dy.data(),
            nullptr,nullptr,nullptr,nullptr,nullptr,rt.w.data(),
            rt.hxDesc,rt.hx.data(),rt.hxDesc,rt.cx.data(),
            rt.xDesc.data(),dx.data(),rt.hxDesc,dhx.data(),rt.hxDesc,dcx.data(),
            nullptr,0,rsv.data(),rt.rsvBytes);

        // Backward weights (reads reserve Part A + Part C)
        auto r3=cudnnRNNBackwardWeights(handle,rt.rnnDesc,T,
            rt.xDesc.data(),rt.x.data(),rt.hxDesc,rt.hx.data(),
            rt.yDesc.data(),rt.y.data(),nullptr,0,nullptr,dw.data(),
            rsv.data(),rt.rsvBytes);

        bool ok=(r1==CUDNN_STATUS_SUCCESS && r2==CUDNN_STATUS_SUCCESS && r3==CUDNN_STATUS_SUCCESS);
        if (ok) {
            // dx should be non-zero (gradient w.r.t. input)
            float dxNorm=0.f; for (float v:dx) dxNorm+=v*v;
            if (dxNorm<1e-12f) { std::cerr<<"FAIL [LSTM Backward] dx all zero\n"; ok=false; }
        }
        if (ok) {
            float dwNorm=0.f; for (float v:dw) dwNorm+=v*v;
            if (dwNorm<1e-12f) { std::cerr<<"FAIL [LSTM Backward] dw all zero\n"; ok=false; }
        }
        if (!ok && r1!=CUDNN_STATUS_SUCCESS) std::cerr<<"FAIL [LSTM Backward] forward r="<<r1<<"\n";
        if (!ok && r2!=CUDNN_STATUS_SUCCESS) std::cerr<<"FAIL [LSTM Backward] bwdData r="<<r2<<"\n";
        if (!ok && r3!=CUDNN_STATUS_SUCCESS) std::cerr<<"FAIL [LSTM Backward] bwdWeights r="<<r3<<"\n";
        if (ok) { std::cout<<"PASS [LSTM backward (dx+dw non-zero)]\n"; ++pass; }
    }

    // ── Test 7: GRU backward cycle ───────────────────────────────────────
    ++total;
    {
        int T=3, B=2, I=3, H=4;
        RNNTest rt(handle,T,B,I,H,1,CUDNN_GRU);
        std::vector<float> rsv(rt.rsvBytes/sizeof(float)+1,0.f);
        std::vector<float> dy(T*B*H,1.f);
        std::vector<float> dx(T*B*I,0.f), dhx(B*H,0.f);
        std::vector<float> dw(rt.wCount,0.f);

        auto r1=cudnnRNNForwardTraining(handle,rt.rnnDesc,T,rt.xDesc.data(),rt.x.data(),
            rt.hxDesc,rt.hx.data(),nullptr,nullptr,nullptr,rt.w.data(),
            rt.yDesc.data(),rt.y.data(),nullptr,nullptr,nullptr,nullptr,nullptr,0,
            rsv.data(),rt.rsvBytes);
        auto r2=cudnnRNNBackwardData(handle,rt.rnnDesc,T,
            rt.yDesc.data(),rt.y.data(),rt.yDesc.data(),dy.data(),
            nullptr,nullptr,nullptr,nullptr,nullptr,rt.w.data(),
            rt.hxDesc,rt.hx.data(),nullptr,nullptr,
            rt.xDesc.data(),dx.data(),rt.hxDesc,dhx.data(),nullptr,nullptr,
            nullptr,0,rsv.data(),rt.rsvBytes);
        auto r3=cudnnRNNBackwardWeights(handle,rt.rnnDesc,T,
            rt.xDesc.data(),rt.x.data(),rt.hxDesc,rt.hx.data(),
            rt.yDesc.data(),rt.y.data(),nullptr,0,nullptr,dw.data(),
            rsv.data(),rt.rsvBytes);

        bool ok=(r1==CUDNN_STATUS_SUCCESS && r2==CUDNN_STATUS_SUCCESS && r3==CUDNN_STATUS_SUCCESS);
        if (ok) {
            float dxN=0.f; for(float v:dx) dxN+=v*v;
            float dwN=0.f; for(float v:dw) dwN+=v*v;
            if (dxN<1e-12f) { std::cerr<<"FAIL [GRU Backward] dx zero\n"; ok=false; }
            if (dwN<1e-12f) { std::cerr<<"FAIL [GRU Backward] dw zero\n"; ok=false; }
        }
        if (ok) { std::cout<<"PASS [GRU backward (dx+dw non-zero)]\n"; ++pass; }
    }

    // ── Test 8: Null pointer rejection ───────────────────────────────────
    ++total;
    {
        auto r=cudnnRNNForwardInference(handle,nullptr,0,nullptr,nullptr,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
            nullptr,nullptr,nullptr,nullptr,nullptr,0);
        if (r==CUDNN_STATUS_INVALID_VALUE) { std::cout<<"PASS [Null pointer rejected]\n"; ++pass; }
        else std::cerr<<"FAIL [Null pointer] status="<<r<<"\n";
    }

    cudnnDestroy(handle);
    std::cout<<"\n"<<pass<<"/"<<total<<" tests passed.\n";
    return (pass==total)?0:1;
}
