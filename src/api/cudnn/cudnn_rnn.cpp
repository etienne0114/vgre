// cuDNN RNN — real LSTM, GRU, RNN_TANH, RNN_RELU
// Implements: forward inference, forward training (with reserve), BPTT, weight gradients.
//
// Weight layout per layer (gate index: LSTM i=0,f=1,g=2,o=3 / GRU r=0,z=1,n=2 / RNN 0):
//   [W_ih: G*H×I | W_hh: G*H×H | b_ih: G*H | b_hh: G*H]   (b_hh absent for RNN_TANH/RELU)
// hx/cx/hy/cy shape: [numLayers × batch × H]
//
// Reserve space layout (ForwardTraining fills; BackwardData also writes Part C):
//   Part A: h_store [numL×T×B×H]                        — hidden output per layer/timestep
//   Part B: fwd_gates [numL×T×B×fwd_slots×H]            — post-activation gate values + c_t
//   Part C: bwd_dpre [numL×T×B×bwd_slots×H]             — pre-activation gradients for dW
// fwd_slots: LSTM=5 (i,f,g,o,c), GRU=3 (r,z,n), RNN=0
// bwd_slots: LSTM=4 (di,df,dg,do), GRU=4 (dr,dz,dn,dn*r), RNN=1 (dpre)

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"
#include <cstring>

extern "C" {

// ── Helpers ────────────────────────────────────────────────────────────────────
static inline float sig(float x) { return 1.f / (1.f + expf(-x)); }

static inline int gate_count(cudnnRNNMode_t m)
{ return m==CUDNN_LSTM ? 4 : (m==CUDNN_GRU ? 3 : 1); }

static inline int fwd_slots(cudnnRNNMode_t m)   // Part B depth per hidden unit
{ return m==CUDNN_LSTM ? 5 : (m==CUDNN_GRU ? 3 : 0); }

static inline int bwd_slots(cudnnRNNMode_t m)   // Part C depth per hidden unit
{ return m==CUDNN_LSTM ? 4 : (m==CUDNN_GRU ? 4 : 1); }

static size_t lpc(cudnnRNNMode_t m, int H, int I) {  // layer param count
    int G = gate_count(m);
    bool twoB = (m!=CUDNN_RNN_TANH && m!=CUDNN_RNN_RELU);
    return (size_t)G*(H*I + H*H + H) + (twoB ? (size_t)G*H : 0);
}

// Weight layout accessors
struct LW { const float *W_ih, *W_hh, *b_ih, *b_hh; };
struct LWm { float *W_ih, *W_hh, *b_ih, *b_hh; };
static LW get_lw(cudnnRNNMode_t m, int H, int I, const float* p) {
    int G = gate_count(m);
    bool twoB = (m!=CUDNN_RNN_TANH && m!=CUDNN_RNN_RELU);
    LW r; r.W_ih=p; r.W_hh=p+(size_t)G*H*I; r.b_ih=r.W_hh+(size_t)G*H*H;
    r.b_hh = twoB ? r.b_ih+(size_t)G*H : nullptr; return r;
}
static LWm get_lwm(cudnnRNNMode_t m, int H, int I, float* p) {
    int G = gate_count(m);
    bool twoB = (m!=CUDNN_RNN_TANH && m!=CUDNN_RNN_RELU);
    LWm r; r.W_ih=p; r.W_hh=p+(size_t)G*H*I; r.b_ih=r.W_hh+(size_t)G*H*H;
    r.b_hh = twoB ? r.b_ih+(size_t)G*H : nullptr; return r;
}
static size_t lw_off(cudnnRNNMode_t m, int H, int I, int l) {
    if (l==0) return 0;
    size_t off = lpc(m,H,I);
    for (int i=1;i<l;++i) off+=lpc(m,H,H);
    return off;
}

// Reserve space pointers: a=Part A, b=Part B, c=Part C
static float*       rA(float* r) { return r; }
static const float* rAc(const float* r) { return r; }
static float*       rB(float* r, int nL,int T,int B,int H)
    { return r+(size_t)nL*T*B*H; }
static const float* rBc(const float* r, int nL,int T,int B,int H)
    { return r+(size_t)nL*T*B*H; }
static float*       rC(float* r, int nL,int T,int B,int H,int fs)
    { return r+(size_t)nL*T*B*H+(size_t)nL*T*B*fs*H; }
static const float* rCc(const float* r,int nL,int T,int B,int H,int fs)
    { return r+(size_t)nL*T*B*H+(size_t)nL*T*B*fs*H; }

// h_store[l,t,n] → H floats
static inline float* hAt(float* a,int l,int t,int n,int T,int B,int H)
    { return a+((size_t)l*T*B+t*B+n)*H; }
static inline const float* hAtc(const float* a,int l,int t,int n,int T,int B,int H)
    { return a+((size_t)l*T*B+t*B+n)*H; }
// gate_store[l,t,n,s] → H floats  (for Part B or C with their respective slot counts)
static inline float* gAt(float* b,int l,int t,int n,int s,int sl,int T,int B,int H)
    { return b+(((size_t)l*T*B+t*B+n)*sl+s)*H; }
static inline const float* gAtc(const float* b,int l,int t,int n,int s,int sl,int T,int B,int H)
    { return b+(((size_t)l*T*B+t*B+n)*sl+s)*H; }

// ── Forward step functions ─────────────────────────────────────────────────────

// Vanilla RNN: h_t = act(W_ih*x_t + b + W_hh*h_prev)
// fwd_out[n*5*H..] or fwd_out null (gates_out not used for RNN; slot 0 = h_t in Part A)
static void rnn_fwd(const LW& lw, int B, int H, int I, bool relu,
                    const float* x_t, const float* hp, float* h_t)
{
    #ifdef _OPENMP
    #pragma omp parallel for if(B>4)
    #endif
    for (int n=0;n<B;++n)
        for (int j=0;j<H;++j) {
            float s=lw.b_ih[j];
            for (int k=0;k<I;++k) s+=lw.W_ih[j*I+k]*x_t[n*I+k];
            if (hp) for (int k=0;k<H;++k) s+=lw.W_hh[j*H+k]*hp[n*H+k];
            h_t[n*H+j]= relu ? (s>0.f?s:0.f) : tanhf(s);
        }
}

// LSTM: h_t = o*tanh(c_t),  c_t = f*c_prev + i*g
// fwd_out layout [n*5*H + s*H + j]: s=0→i,1→f,2→g,3→o,4→c_t  (may be null)
static void lstm_fwd(const LW& lw, int B, int H, int I,
                     const float* x_t, const float* hp, const float* cp,
                     float* h_t, float* c_t, float* fwd_out)
{
    #ifdef _OPENMP
    #pragma omp parallel for if(B>4)
    #endif
    for (int n=0;n<B;++n) {
        for (int j=0;j<H;++j) {
            float bh0=lw.b_hh?lw.b_hh[0*H+j]:0.f, bh1=lw.b_hh?lw.b_hh[1*H+j]:0.f;
            float bh2=lw.b_hh?lw.b_hh[2*H+j]:0.f, bh3=lw.b_hh?lw.b_hh[3*H+j]:0.f;
            float s0=lw.b_ih[0*H+j]+bh0, s1=lw.b_ih[1*H+j]+bh1;
            float s2=lw.b_ih[2*H+j]+bh2, s3=lw.b_ih[3*H+j]+bh3;
            for (int k=0;k<I;++k) {
                float xk=x_t[n*I+k];
                s0+=lw.W_ih[(0*H+j)*I+k]*xk; s1+=lw.W_ih[(1*H+j)*I+k]*xk;
                s2+=lw.W_ih[(2*H+j)*I+k]*xk; s3+=lw.W_ih[(3*H+j)*I+k]*xk;
            }
            if (hp) for (int k=0;k<H;++k) {
                float hk=hp[n*H+k];
                s0+=lw.W_hh[(0*H+j)*H+k]*hk; s1+=lw.W_hh[(1*H+j)*H+k]*hk;
                s2+=lw.W_hh[(2*H+j)*H+k]*hk; s3+=lw.W_hh[(3*H+j)*H+k]*hk;
            }
            float iv=sig(s0), fv=sig(s1), gv=tanhf(s2), ov=sig(s3);
            float cn= fv*(cp?cp[n*H+j]:0.f) + iv*gv;
            float hn= ov * std::tanh(cn);
            c_t[n*H+j] = cn;
            h_t[n*H+j] = hn;
            if (fwd_out) {
                fwd_out[(n*5+0)*H+j] = iv;
                fwd_out[(n*5+1)*H+j] = fv;
                fwd_out[(n*5+2)*H+j] = gv;
                fwd_out[(n*5+3)*H+j] = ov;
                fwd_out[(n*5+4)*H+j] = cn;
            }
        }
    }
}

// GRU: r=sig(...), z=sig(...), n=tanh(sn_x + r*sn_h), h=(1-z)*n + z*h_prev
// fwd_out [n*3*H + s*H + j]: s=0→r,1→z,2→n  (may be null)
static void gru_fwd(const LW& lw, int B, int H, int I,
                    const float* x_t, const float* hp,
                    float* h_t, float* fwd_out)
{
    #ifdef _OPENMP
    #pragma omp parallel for if(B>4)
    #endif
    for (int n=0;n<B;++n) {
        for (int j=0;j<H;++j) {
            float sr=lw.b_ih[0*H+j]+(lw.b_hh?lw.b_hh[0*H+j]:0.f);
            float sz=lw.b_ih[1*H+j]+(lw.b_hh?lw.b_hh[1*H+j]:0.f);
            float snx=lw.b_ih[2*H+j], snh=lw.b_hh?lw.b_hh[2*H+j]:0.f;
            for (int k=0;k<I;++k) {
                float xk=x_t[n*I+k];
                sr+=lw.W_ih[(0*H+j)*I+k]*xk;
                sz+=lw.W_ih[(1*H+j)*I+k]*xk;
                snx+=lw.W_ih[(2*H+j)*I+k]*xk;
            }
            if (hp) for (int k=0;k<H;++k) {
                float hk=hp[n*H+k];
                sr+=lw.W_hh[(0*H+j)*H+k]*hk;
                sz+=lw.W_hh[(1*H+j)*H+k]*hk;
                snh+=lw.W_hh[(2*H+j)*H+k]*hk;
            }
            float rv=sig(sr), zv=sig(sz), nv=tanhf(snx+rv*snh);
            float hp_j=hp?hp[n*H+j]:0.f;
            h_t[n*H+j]=(1.f-zv)*nv+zv*hp_j;
            if (fwd_out) {
                fwd_out[(n*3+0)*H+j]=rv;
                fwd_out[(n*3+1)*H+j]=zv;
                fwd_out[(n*3+2)*H+j]=nv;
            }
        }
    }
}

// ── Forward implementation (shared by Inference and Training) ──────────────────
static cudnnStatus_t rnn_forward(
    cudnnRNNMode_t mode, int numL, int H,
    int T, int B, int I,
    const float* w, const float* hxf, const float* cxf,
    const float* xf, float* yf, float* hyf, float* cyf,
    float* rsv)  // null for inference
{
    bool isLSTM=(mode==CUDNN_LSTM), isGRU=(mode==CUDNN_GRU);
    bool relu=(mode==CUDNN_RNN_RELU);
    int fs=fwd_slots(mode);

    // Part A pointer
    float* pa = rsv ? rA(rsv) : nullptr;
    float* pb = rsv && fs>0 ? rB(rsv,numL,T,B,H) : nullptr;

    // Per-layer current hidden/cell states (carry across timesteps)
    std::vector<float> hprev((size_t)numL*B*H, 0.f);
    std::vector<float> cprev((size_t)numL*B*H, 0.f);  // LSTM only
    if (hxf) memcpy(hprev.data(), hxf, (size_t)numL*B*H*sizeof(float));
    if (isLSTM && cxf) memcpy(cprev.data(), cxf, (size_t)numL*B*H*sizeof(float));

    // Intermediate layer outputs (needed for multilayer forward + reserve Part A)
    // For layer l, timestep t: stored in Part A if rsv, else in tmp_inter.
    std::vector<float> tmp_inter;
    if (!rsv && numL>1) tmp_inter.assign((size_t)(numL-1)*T*B*H, 0.f);

    auto inter_ptr = [&](int l, int t) -> float* {
        if (rsv) return hAt(pa, l, t, 0, T, B, H);
        if (l==numL-1) return yf + (size_t)t*B*H;
        return tmp_inter.data() + ((size_t)l*T+t)*B*H;
    };

    std::vector<float> ctmp(B*H, 0.f);  // scratch for lstm c_t before placing in cprev

    for (int t=0;t<T;++t) {
        for (int l=0;l<numL;++l) {
            int li=(l==0)?I:H;
            LW lw=get_lw(mode,H,li, w+lw_off(mode,H,I,l));
            float* hp=hprev.data()+l*B*H;
            float* cp=isLSTM ? (cprev.data()+l*B*H) : nullptr;
            const float* x_t=(l==0)?(xf+(size_t)t*B*I):(inter_ptr(l-1,t));
            float* h_t=inter_ptr(l,t);
            float* fwd_out=nullptr;
            if (pb && fs>0) fwd_out=gAt(pb,l,t,0,0,fs,T,B,H);

            if (isLSTM) {
                lstm_fwd(lw,B,H,li, x_t,hp,cp, h_t,ctmp.data(),fwd_out);
                memcpy(cp, ctmp.data(), (size_t)B*H*sizeof(float));
            } else if (isGRU) {
                gru_fwd(lw,B,H,li, x_t,hp, h_t,fwd_out);
            } else {
                rnn_fwd(lw,B,H,li,relu, x_t,hp,h_t);
            }
            // Update hprev for next timestep
            memcpy(hp, h_t, (size_t)B*H*sizeof(float));
        }
    }

    // Copy final layer output to y (when using reserve, inter_ptr(numL-1,t) is in Part A)
    if (rsv) {
        for (int t=0;t<T;++t)
            memcpy(yf+(size_t)t*B*H, hAt(pa,numL-1,t,0,T,B,H), (size_t)B*H*sizeof(float));
    }

    if (hyf) memcpy(hyf, hprev.data(), (size_t)numL*B*H*sizeof(float));
    if (isLSTM && cyf) memcpy(cyf, cprev.data(), (size_t)numL*B*H*sizeof(float));
    return CUDNN_STATUS_SUCCESS;
}

// ── Public forward API ─────────────────────────────────────────────────────────

cudnnStatus_t cudnnGetRNNWorkspaceSize(
    cudnnHandle_t, cudnnRNNDescriptor_t, int, cudnnTensorDescriptor_t*, size_t* s)
{ if (!s) return CUDNN_STATUS_INVALID_VALUE; *s=0; return CUDNN_STATUS_SUCCESS; }

cudnnStatus_t cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int T,
    cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes)
{
    if (!sizeInBytes||!rnnDesc||!xDesc) return CUDNN_STATUS_INVALID_VALUE;
    auto* rd=(RNNDesc*)rnnDesc;
    int H=rd->hiddenSize, B=((TensorDesc*)xDesc[0])->n, nL=rd->numLayers;
    int fs=fwd_slots(rd->mode), bs=bwd_slots(rd->mode);
    *sizeInBytes=(size_t)nL*T*B*H*(1+fs+bs)*sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
    cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes, cudnnDataType_t)
{
    if (!sizeInBytes||!rnnDesc||!xDesc) return CUDNN_STATUS_INVALID_VALUE;
    auto* rd=(RNNDesc*)rnnDesc; auto* td=(TensorDesc*)xDesc;
    int H=rd->hiddenSize, I=td->c*td->h*td->w, nL=rd->numLayers;
    size_t tot=lpc(rd->mode,H,I);
    for (int l=1;l<nL;++l) tot+=lpc(rd->mode,H,H);
    *sizeInBytes=tot*sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int T,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t, const void* cx,
    cudnnFilterDescriptor_t, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t, void* hy,
    cudnnTensorDescriptor_t, void* cy,
    void*, size_t)
{
    if (!rnnDesc||!xDesc||!x||!yDesc||!y||!w) return CUDNN_STATUS_INVALID_VALUE;
    (void)hxDesc;
    auto* rd=(RNNDesc*)rnnDesc;
    auto* xt=(TensorDesc*)xDesc[0];
    int B=xt->n, I=xt->c*xt->h*xt->w;
    return rnn_forward(rd->mode,rd->numLayers,rd->hiddenSize,T,B,I,
        (const float*)w,(const float*)hx,(const float*)cx,
        (const float*)x,(float*)y,(float*)hy,(float*)cy,nullptr);
}

cudnnStatus_t cudnnRNNForwardTraining(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int T,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t, const void* hx,
    cudnnTensorDescriptor_t, const void* cx,
    cudnnFilterDescriptor_t, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t, void* hy,
    cudnnTensorDescriptor_t, void* cy,
    void*, size_t,
    void* rsv, size_t rsvSz)
{
    if (!rnnDesc||!xDesc||!x||!yDesc||!y||!w) return CUDNN_STATUS_INVALID_VALUE;
    auto* rd=(RNNDesc*)rnnDesc;
    auto* xt=(TensorDesc*)xDesc[0];
    int B=xt->n, I=xt->c*xt->h*xt->w;
    if (rsv) memset(rsv,0,rsvSz);
    return rnn_forward(rd->mode,rd->numLayers,rd->hiddenSize,T,B,I,
        (const float*)w,(const float*)hx,(const float*)cx,
        (const float*)x,(float*)y,(float*)hy,(float*)cy,(float*)rsv);
}

// ── Backward data (BPTT) ───────────────────────────────────────────────────────
cudnnStatus_t cudnnRNNBackwardData(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int T,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    cudnnTensorDescriptor_t* dyDesc, const void* dy,
    cudnnTensorDescriptor_t, const void* dhy,
    cudnnTensorDescriptor_t, const void* dcy,
    cudnnFilterDescriptor_t, const void* w,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t, const void* cx,
    cudnnTensorDescriptor_t* dxDesc, void* dx,
    cudnnTensorDescriptor_t, void* dhx,
    cudnnTensorDescriptor_t, void* dcx,
    void*, size_t, void* rsv, size_t)
{
    if (!rnnDesc||!dy||!dx||!w||!rsv) return CUDNN_STATUS_INVALID_VALUE;
    (void)hxDesc; (void)y; (void)yDesc; (void)dyDesc; (void)dxDesc;
    auto* rd=(RNNDesc*)rnnDesc;
    int H=rd->hiddenSize, nL=rd->numLayers;
    auto* xt=(TensorDesc*)dxDesc[0];
    int B=xt->n, I=xt->c*xt->h*xt->w;
    cudnnRNNMode_t mode=rd->mode;
    bool isLSTM=(mode==CUDNN_LSTM), isGRU=(mode==CUDNN_GRU), relu=(mode==CUDNN_RNN_RELU);
    int fs=fwd_slots(mode), bs=bwd_slots(mode);

    const float* wf=(const float*)w;
    const float* hxf=(const float*)hx;
    const float* cxf=(const float*)cx;
    const float* dyf=(const float*)dy;
    const float* dhyf=(const float*)dhy;
    const float* dcyf=(const float*)dcy;
    float* dxf=(float*)dx;
    float* dhxf=(float*)dhx;
    float* dcxf=(float*)dcx;
    float* rf=(float*)rsv;

    float* pa=rA(rf);
    float* pb=rB(rf,nL,T,B,H);
    float* pc=rC(rf,nL,T,B,H,fs);

    memset(dxf,0,(size_t)T*B*I*sizeof(float));

    // Running gradients: dh[l][B×H], dc[l][B×H] (carry from t to t-1)
    std::vector<float> dh((size_t)nL*B*H,0.f), dc((size_t)nL*B*H,0.f);
    if (dhyf) memcpy(dh.data(),dhyf,(size_t)nL*B*H*sizeof(float));
    if (isLSTM&&dcyf) memcpy(dc.data(),dcyf,(size_t)nL*B*H*sizeof(float));

    // Temp buffers per timestep
    std::vector<float> dh_new(B*H), dx_l(B*H);  // dx_l is input-grad for intermediate layers

    for (int t=T-1;t>=0;--t) {
        for (int l=nL-1;l>=0;--l) {
            int li=(l==0)?I:H;
            LW lw=get_lw(mode,H,li,wf+lw_off(mode,H,I,l));
            float* dh_l=dh.data()+l*B*H;
            float* dc_l=dc.data()+l*B*H;

            // Top layer receives external dy
            if (l==nL-1) {
                const float* dy_t=dyf+(size_t)t*B*H;
                for (int i=0;i<B*H;++i) dh_l[i]+=dy_t[i];
            }

            // Pointers to stored forward gates for this (l,t)
            const float* g_fwd=gAtc(pb,l,t,0,0,fs,T,B,H); // Part B slot base [n*fs*H+s*H+j]
            // Part C base for this (l,t): where we write dpre
            float* g_bwd=gAt(pc,l,t,0,0,bs,T,B,H);        // [n*bs*H+s*H+j]

            // h_{t-1} for this layer
            const float* hprev_lt = (t>0) ? hAtc(pa,l,t-1,0,T,B,H)
                                           : (hxf ? hxf+l*B*H : nullptr);
            // c_{t-1} (LSTM): stored in Part B slot 4 of t-1, or cx for t=0
            // We'll load per (n,j) inside the loop.

            std::fill(dh_new.begin(),dh_new.end(),0.f);
            float* dx_out=(l==0)?(dxf+(size_t)t*B*li):(dx_l.data());
            if (l>0) std::fill(dx_l.begin(),dx_l.end(),0.f);
            else memset(dx_out,0,(size_t)B*li*sizeof(float));

            if (isLSTM) {
                // Part C layout for LSTM: s=0→di,1→df,2→dg,3→do
                for (int n=0;n<B;++n) {
                    for (int j=0;j<H;++j) {
                        float iv=g_fwd[(n*5+0)*H+j], fv=g_fwd[(n*5+1)*H+j];
                        float gv=g_fwd[(n*5+2)*H+j], ov=g_fwd[(n*5+3)*H+j];
                        float cn=g_fwd[(n*5+4)*H+j];
                        float cp_j = (t>0)
                            ? gAtc(pb,l,t-1,n,4,fs,T,B,H)[j]
                            : (cxf ? cxf[l*B*H+n*H+j] : 0.f);

                        float tanh_cn=tanhf(cn);
                        float dh_j=dh_l[n*H+j], dc_j=dc_l[n*H+j];

                        float do_act=dh_j*tanh_cn;
                        dc_j+=dh_j*ov*(1.f-tanh_cn*tanh_cn);
                        float df_act=dc_j*cp_j, di_act=dc_j*gv, dg_act=dc_j*iv;
                        dc_l[n*H+j]=dc_j*fv;  // propagate to c_{t-1}

                        float di=di_act*iv*(1.f-iv), df=df_act*fv*(1.f-fv);
                        float dg=dg_act*(1.f-gv*gv), doo=do_act*ov*(1.f-ov);
                        g_bwd[(n*4+0)*H+j]=di; g_bwd[(n*4+1)*H+j]=df;
                        g_bwd[(n*4+2)*H+j]=dg; g_bwd[(n*4+3)*H+j]=doo;

                        float dp[4]={di,df,dg,doo};
                        for (int s=0;s<4;++s) {
                            for (int k=0;k<li;++k)
                                dx_out[n*li+k]+=lw.W_ih[(s*H+j)*li+k]*dp[s];
                            const float* hp_n=hprev_lt?hprev_lt+n*H:nullptr;(void)hp_n;
                            for (int k=0;k<H;++k)
                                dh_new[n*H+k]+=lw.W_hh[(s*H+j)*H+k]*dp[s];
                        }
                    }
                }
            } else if (isGRU) {
                // Part C for GRU: s=0→dr,1→dz,2→dn,3→dn*r (for W_n_hh)
                for (int n=0;n<B;++n) {
                    const float* hp_n=hprev_lt?hprev_lt+n*H:nullptr;
                    for (int j=0;j<H;++j) {
                        float rv=g_fwd[(n*3+0)*H+j];
                        float zv=g_fwd[(n*3+1)*H+j];
                        float nv=g_fwd[(n*3+2)*H+j];
                        float hp_j=hp_n?hp_n[j]:0.f;
                        float dh_j=dh_l[n*H+j];

                        float dn_act=dh_j*(1.f-zv);
                        float dz_act=dh_j*(hp_j-nv);
                        dh_new[n*H+j]+=dh_j*zv;  // direct path

                        float dn_pre=dn_act*(1.f-nv*nv);

                        // Recompute sn_h = W_n_hh * h_prev + b_n_hh  (for dr_act)
                        float snh=lw.b_hh?lw.b_hh[2*H+j]:0.f;
                        if (hp_n) for (int k=0;k<H;++k) snh+=lw.W_hh[(2*H+j)*H+k]*hp_n[k];

                        float dr_act=dn_pre*snh;
                        float dr_pre=dr_act*rv*(1.f-rv);
                        float dz_pre=dz_act*zv*(1.f-zv);
                        float dn_r=dn_pre*rv;  // for dW_n_hh

                        g_bwd[(n*4+0)*H+j]=dr_pre;
                        g_bwd[(n*4+1)*H+j]=dz_pre;
                        g_bwd[(n*4+2)*H+j]=dn_pre;
                        g_bwd[(n*4+3)*H+j]=dn_r;

                        // dx via W_r_ih, W_z_ih, W_n_ih
                        for (int k=0;k<li;++k)
                            dx_out[n*li+k]+=lw.W_ih[(0*H+j)*li+k]*dr_pre
                                           +lw.W_ih[(1*H+j)*li+k]*dz_pre
                                           +lw.W_ih[(2*H+j)*li+k]*dn_pre;
                        // dh_{t-1} via W_r_hh, W_z_hh, W_n_hh (with r-gate factor for n)
                        for (int k=0;k<H;++k)
                            dh_new[n*H+k]+=lw.W_hh[(0*H+j)*H+k]*dr_pre
                                          +lw.W_hh[(1*H+j)*H+k]*dz_pre
                                          +lw.W_hh[(2*H+j)*H+k]*dn_r;
                    }
                }
            } else {
                // Vanilla RNN: h_t in Part A
                for (int n=0;n<B;++n) {
                    for (int j=0;j<H;++j) {
                        float ht=hAtc(pa,l,t,n,T,B,H)[j];
                        float dh_j=dh_l[n*H+j];
                        float dpre=relu?(ht>0.f?dh_j:0.f):((1.f-ht*ht)*dh_j);
                        g_bwd[(n*1+0)*H+j]=dpre;
                        for (int k=0;k<li;++k) dx_out[n*li+k]+=lw.W_ih[j*li+k]*dpre;
                        for (int k=0;k<H;++k)  dh_new[n*H+k]+=lw.W_hh[j*H+k]*dpre;
                    }
                }
            }

            // Update dh_l with the gradient to h_{t-1} (carry to next t iteration)
            memcpy(dh_l,dh_new.data(),(size_t)B*H*sizeof(float));

            // Propagate dx from intermediate layer to layer below
            if (l>0) {
                float* dh_below=dh.data()+(l-1)*B*H;
                for (int i=0;i<B*H;++i) dh_below[i]+=dx_l[i];
            }
        }
    }

    if (dhxf) memcpy(dhxf,dh.data(),(size_t)nL*B*H*sizeof(float));
    if (isLSTM&&dcxf) memcpy(dcxf,dc.data(),(size_t)nL*B*H*sizeof(float));
    return CUDNN_STATUS_SUCCESS;
}

// ── Backward weights ───────────────────────────────────────────────────────────
// Accumulates into dw (caller must zero it first if needed).
// Reads dpre from Part C written by BackwardData.
cudnnStatus_t cudnnRNNBackwardWeights(
    cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc, int T,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t* yDesc, const void* y,
    void*, size_t,
    cudnnFilterDescriptor_t, void* dw,
    void* rsv, size_t)
{
    if (!rnnDesc||!xDesc||!x||!y||!dw||!rsv) return CUDNN_STATUS_INVALID_VALUE;
    (void)hxDesc; (void)yDesc;
    auto* rd=(RNNDesc*)rnnDesc;
    int H=rd->hiddenSize, nL=rd->numLayers;
    auto* xt=(TensorDesc*)xDesc[0];
    int B=xt->n, I=xt->c*xt->h*xt->w;
    cudnnRNNMode_t mode=rd->mode;
    bool isLSTM=(mode==CUDNN_LSTM), isGRU=(mode==CUDNN_GRU);
    int fs=fwd_slots(mode), bs=bwd_slots(mode);

    const float* xf=(const float*)x;
    const float* hxf=(const float*)hx;
    float* dwf=(float*)dw;
    float* rf=(float*)rsv;

    float* pa=rA(rf);
    float* pc=rC(rf,nL,T,B,H,fs);

    // Zero dw
    { size_t tot=lpc(mode,H,I); for(int l=1;l<nL;++l) tot+=lpc(mode,H,H);
      memset(dwf,0,tot*sizeof(float)); }

    for (int l=0;l<nL;++l) {
        int li=(l==0)?I:H;
        LWm dlw=get_lwm(mode,H,li,dwf+lw_off(mode,H,I,l));
        int G=gate_count(mode);

        for (int t=0;t<T;++t) {
            // x input to this layer at time t
            const float* x_t=(l==0)?(xf+(size_t)t*B*I)
                                    :hAtc(pa,l-1,t,0,T,B,H);
            // h_{t-1} for this layer
            const float* hp=(t>0)?hAtc(pa,l,t-1,0,T,B,H)
                                  :(hxf?hxf+l*B*H:nullptr);
            // dpre for this (l,t)
            const float* dp=gAtc(pc,l,t,0,0,bs,T,B,H); // [n*bs*H+s*H+j]

            for (int n=0;n<B;++n) {
                const float* xt_n=x_t?x_t+n*li:nullptr;
                const float* hp_n=hp?hp+n*H:nullptr;

                if (isLSTM) {
                    // bs=4: s=0→di,1→df,2→dg,3→do
                    for (int s=0;s<4;++s) {
                        const float* dps=dp+(n*4+s)*H;
                        for (int j=0;j<H;++j) {
                            float d=dps[j];
                            if (!d) continue;
                            dlw.b_ih[s*H+j]+=d;
                            dlw.b_hh[s*H+j]+=d;
                            if (xt_n) for(int k=0;k<li;++k) dlw.W_ih[(s*H+j)*li+k]+=d*xt_n[k];
                            if (hp_n) for(int k=0;k<H;++k)  dlw.W_hh[(s*H+j)*H+k] +=d*hp_n[k];
                        }
                    }
                } else if (isGRU) {
                    // bs=4: s=0→dr,1→dz,2→dn_ih,3→dn_hh (=dn*r for W_n_hh)
                    // W_ih gates: r(s=0), z(s=1), n(s=2)  → use dpre s=0,1,2
                    // W_hh gates: r(s=0), z(s=1), n(s=2)  → use dpre s=0,1,3
                    // b_ih: r→s=0, z→s=1, n→s=2
                    // b_hh: r→s=0, z→s=1, n→s=3
                    for (int j=0;j<H;++j) {
                        float dr=dp[(n*4+0)*H+j], dz=dp[(n*4+1)*H+j];
                        float dn=dp[(n*4+2)*H+j], dnr=dp[(n*4+3)*H+j];
                        dlw.b_ih[0*H+j]+=dr; dlw.b_ih[1*H+j]+=dz; dlw.b_ih[2*H+j]+=dn;
                        if (dlw.b_hh) { dlw.b_hh[0*H+j]+=dr; dlw.b_hh[1*H+j]+=dz; dlw.b_hh[2*H+j]+=dnr; }
                        if (xt_n) for(int k=0;k<li;++k) {
                            dlw.W_ih[(0*H+j)*li+k]+=dr*xt_n[k];
                            dlw.W_ih[(1*H+j)*li+k]+=dz*xt_n[k];
                            dlw.W_ih[(2*H+j)*li+k]+=dn*xt_n[k];
                        }
                        if (hp_n) for(int k=0;k<H;++k) {
                            dlw.W_hh[(0*H+j)*H+k]+=dr *hp_n[k];
                            dlw.W_hh[(1*H+j)*H+k]+=dz *hp_n[k];
                            dlw.W_hh[(2*H+j)*H+k]+=dnr*hp_n[k];
                        }
                    }
                } else {
                    // RNN: bs=1, s=0→dpre
                    for (int j=0;j<H;++j) {
                        float d=dp[(n*1+0)*H+j];
                        if (!d) continue;
                        dlw.b_ih[j]+=d;
                        if (xt_n) for(int k=0;k<li;++k) dlw.W_ih[j*li+k]+=d*xt_n[k];
                        if (hp_n) for(int k=0;k<H;++k)  dlw.W_hh[j*H+k] +=d*hp_n[k];
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── RNNData descriptor (for Ex variants) ────────────────────────────────────
cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t* d) {
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    *d = new cudnnRNNData_st();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t d) {
    delete d;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t d,
    cudnnDataType_t dataType,
    cudnnRNNDataLayout_t layout,
    int maxSeqLength,
    int batchSize,
    int vectorSize,
    const int* seqLengthArray,
    void* paddingFill)
{
    if (!d || !seqLengthArray) return CUDNN_STATUS_INVALID_VALUE;
    d->dataType     = dataType;
    d->layout       = layout;
    d->maxSeqLength = maxSeqLength;
    d->batchSize    = batchSize;
    d->vectorSize   = vectorSize;
    d->seqLengthArray.assign(seqLengthArray, seqLengthArray + batchSize);
    (void)paddingFill;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t d,
    cudnnDataType_t* dataType,
    cudnnRNNDataLayout_t* layout,
    int* maxSeqLength,
    int* batchSize,
    int* vectorSize,
    int arrayLengthRequested,
    int* seqLengthArray)
{
    if (!d) return CUDNN_STATUS_INVALID_VALUE;
    if (dataType)     *dataType    = d->dataType;
    if (layout)       *layout      = d->layout;
    if (maxSeqLength) *maxSeqLength = d->maxSeqLength;
    if (batchSize)    *batchSize   = d->batchSize;
    if (vectorSize)   *vectorSize  = d->vectorSize;
    if (seqLengthArray && arrayLengthRequested > 0) {
        int n = std::min(arrayLengthRequested, (int)d->seqLengthArray.size());
        for (int i = 0; i < n; ++i) seqLengthArray[i] = d->seqLengthArray[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── RNNForwardInferenceEx / RNNForwardTrainingEx ─────────────────────────────
static cudnnStatus_t rnn_forward_ex(
    cudnnRNNDescriptor_t      rnnDesc,
    cudnnRNNDataDescriptor_t  xD,   const void* x,
    cudnnTensorDescriptor_t   hxDesc, const void* hx,
    cudnnTensorDescriptor_t   cxDesc, const void* cx,
    cudnnFilterDescriptor_t   wDesc,  const void* w,
    cudnnRNNDataDescriptor_t  yD,   void* y,
    cudnnTensorDescriptor_t   hyDesc, void* hy,
    cudnnTensorDescriptor_t   cyDesc, void* cy,
    void* rsv, size_t rsvSz)
{
    if (!rnnDesc || !xD || !x || !yD || !y || !w) return CUDNN_STATUS_INVALID_VALUE;
    (void)hxDesc; (void)cxDesc; (void)wDesc; (void)hyDesc; (void)cyDesc;

    auto* rd = (RNNDesc*)rnnDesc;
    int T  = xD->maxSeqLength;
    int B  = xD->batchSize;
    int I  = xD->vectorSize;
    int H  = rd->hiddenSize;

    if (rsv) memset(rsv, 0, rsvSz);
    cudnnStatus_t st = rnn_forward(
        rd->mode, rd->numLayers, H, T, B, I,
        (const float*)w, (const float*)hx, (const float*)cx,
        (const float*)x, (float*)y, (float*)hy, (float*)cy, (float*)rsv);
    if (st != CUDNN_STATUS_SUCCESS) return st;

    // Mask output past each sample's actual sequence length.
    int outVec = H;
    bool seqFirst = (yD->layout == 0 || yD->layout == 1);
    for (int n = 0; n < B; ++n) {
        int slen = (n < (int)yD->seqLengthArray.size()) ? yD->seqLengthArray[n] : T;
        for (int t = slen; t < T; ++t) {
            float* dst;
            if (seqFirst)
                dst = (float*)y + (t * B + n) * outVec;
            else
                dst = (float*)y + (n * T + t) * outVec;
            memset(dst, 0, outVec * sizeof(float));
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInferenceEx(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNDataDescriptor_t xD, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnRNNDataDescriptor_t yD, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void*, size_t,
    void*, size_t)
{
    return rnn_forward_ex(rnnDesc, xD, x, hxDesc, hx, cxDesc, cx,
                          wDesc, w, yD, y, hyDesc, hy, cyDesc, cy,
                          nullptr, 0);
}

cudnnStatus_t cudnnRNNForwardTrainingEx(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNDataDescriptor_t xD, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnRNNDataDescriptor_t yD, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void*, size_t,
    void* rsv, size_t rsvSz,
    void*, size_t)
{
    return rnn_forward_ex(rnnDesc, xD, x, hxDesc, hx, cxDesc, cx,
                          wDesc, w, yD, y, hyDesc, hy, cyDesc, cy,
                          rsv, rsvSz);
}

cudnnStatus_t cudnnRNNBackwardDataEx(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNDataDescriptor_t yD,   const void* y,
    cudnnRNNDataDescriptor_t dyD,  const void* dy,
    cudnnRNNDataDescriptor_t,      const void*,
    cudnnTensorDescriptor_t,       const void* dhy,
    cudnnTensorDescriptor_t,       const void* dcy,
    cudnnFilterDescriptor_t,       const void* w,
    cudnnTensorDescriptor_t,       const void* hx,
    cudnnTensorDescriptor_t,       const void* cx,
    cudnnRNNDataDescriptor_t dxD,  void* dx,
    cudnnTensorDescriptor_t,       void* dhx,
    cudnnTensorDescriptor_t,       void* dcx,
    void*, size_t,
    void* rsv, size_t rsvSz,
    void*, size_t)
{
    if (!rnnDesc || !dy || !dx || !w || !rsv) return CUDNN_STATUS_INVALID_VALUE;
    (void)y; (void)dxD;
    auto* rd = (RNNDesc*)rnnDesc;
    int H  = rd->hiddenSize, nL = rd->numLayers;
    int T  = (dyD && dyD->maxSeqLength > 0) ? dyD->maxSeqLength : 1;
    int B  = (dyD && dyD->batchSize    > 0) ? dyD->batchSize    : 1;
    int I  = (yD  && yD->vectorSize    > 0) ? yD->vectorSize    : H;
    (void)yD; (void)dyD;

    cudnnRNNMode_t mode = rd->mode;
    bool isLSTM = (mode == CUDNN_LSTM), isGRU = (mode == CUDNN_GRU);
    bool relu = (mode == CUDNN_RNN_RELU);
    int fs = fwd_slots(mode), bs = bwd_slots(mode);

    const float* wf  = (const float*)w;
    const float* hxf = (const float*)hx;
    const float* cxf = (const float*)cx;
    const float* dyf = (const float*)dy;
    const float* dhyf= (const float*)dhy;
    const float* dcyf= (const float*)dcy;
    float*       dxf = (float*)dx;
    float*       dhxf= (float*)dhx;
    float*       dcxf= (float*)dcx;

    const float* rsv_f = (const float*)rsv;
    size_t partA = (size_t)nL * T * B * H;
    size_t partB = (size_t)nL * T * B * fs * H;
    const float* fwd_g = rsv_f + partA;
    float*       dpre  = (float*)rsv + partA + partB;   // Part C: bwd_dpre

    memset(dpre, 0, (size_t)nL * T * B * bs * H * sizeof(float));
    if (dxf) memset(dxf, 0, (size_t)T * B * I * sizeof(float));

    // Simplified BPTT: reverse timesteps per layer.
    std::vector<float> dh((size_t)nL * B * H, 0.0f);
    std::vector<float> dc((size_t)nL * B * H, 0.0f);
    if (dhyf) memcpy(dh.data(), dhyf, (size_t)nL * B * H * sizeof(float));
    if (dcyf) memcpy(dc.data(), dcyf, (size_t)nL * B * H * sizeof(float));

    for (int l = nL - 1; l >= 0; --l) {
        int li = (l == 0) ? I : H;
        size_t wOff = lw_off(mode, H, I, l);
        LW lw = get_lw(mode, H, I, wf + wOff);
        float* dh_l = dh.data() + (size_t)l * B * H;
        float* dc_l = dc.data() + (size_t)l * B * H;
        const float* h_store = rsv_f + (size_t)l * T * B * H;
        const float* fg_l    = fwd_g + (size_t)l * T * B * fs * H;
        float* dp_l          = dpre  + (size_t)l * T * B * bs * H;

        for (int t = T - 1; t >= 0; --t) {
            const float* dyo_t = (l == nL-1) ? dyf + (size_t)t*B*H : nullptr;
            float* dx_t  = (l == 0)    ? dxf + (size_t)t*B*I : nullptr;
            const float* fg_t = fg_l + (size_t)t * B * fs * H;
            float* dp_t       = dp_l + (size_t)t * B * bs * H;
            const float* hp_t = (t > 0) ? h_store + (size_t)(t-1)*B*H : hxf ? hxf+(size_t)l*B*H : nullptr;

            for (int n = 0; n < B; ++n) {
                float* dh_n  = dh_l + n*H;
                float* dc_n  = dc_l + n*H;
                const float* dyo_n = dyo_t ? dyo_t + n*H : nullptr;
                float* dp_n  = dp_t + n*bs*H;

                if (isLSTM) {
                    // fg layout: i,f,g,o,c  (indices 0-4)
                    const float* fg_n = fg_t + n*fs*H;
                    std::vector<float> dct(H);
                    for (int j = 0; j < H; ++j) {
                        float do_ = dyo_n ? dyo_n[j] : 0.0f;
                        float dh_j = dh_n[j] + do_;
                        float ct = fg_n[4*H+j];
                        float ot = fg_n[3*H+j];
                        // dC from output gate path
                        float tanh_ct = tanhf(ct);
                        float dct_j = dh_j * ot * (1.f - tanh_ct*tanh_ct) + dc_n[j];
                        dct[j] = dct_j;
                        // gate gradients (pre-activation)
                        float ft = fg_n[1*H+j];
                        dp_n[0*H+j] = dct_j * fg_n[2*H+j] * fg_n[0*H+j]*(1.f-fg_n[0*H+j]); // di
                        // Exact: multiply by previous cell state C_{t-1}, not hidden state H_{t-1}
                        float cp_j = 0.f;
                        if (t > 0) {
                            const float* fg_prev = fg_l + (size_t)(t-1) * B * fs * H + (size_t)n * fs * H;
                            cp_j = fg_prev[4*H+j];
                        } else if (cxf) {
                            cp_j = cxf[l*B*H + n*H + j];
                        }
                        dp_n[1*H+j] = dct_j * cp_j * ft*(1.f-ft);                             // df (exact)
                        dp_n[2*H+j] = dct_j * fg_n[0*H+j] * (1.f - fg_n[2*H+j]*fg_n[2*H+j]);// dg
                        dp_n[3*H+j] = dh_j  * tanh_ct * ot*(1.f-ot);                          // do
                        dc_n[j]     = dct_j * ft;
                    }
                    // propagate dh to previous timestep/layer via W_hh/W_ih
                    if (l == nL-1) {
                        for (int j=0;j<H;++j) dh_n[j] = 0.f;
                        for (int s=0;s<4;++s)
                            for (int j=0;j<H;++j)
                                for (int k=0;k<H;++k)
                                    dh_n[k] += dp_n[s*H+j]*lw.W_hh[s*H*H+j*H+k];
                    }
                    if (dx_t)
                        for (int s=0;s<4;++s)
                            for (int j=0;j<H;++j)
                                for (int k=0;k<li;++k)
                                    dx_t[n*li+k] += dp_n[s*H+j]*lw.W_ih[s*H*li+j*li+k];
                } else if (isGRU) {
                    const float* fg_n = fg_t + n*fs*H;
                    for (int j=0;j<H;++j) {
                        float dyo_j = (dyo_n?dyo_n[j]:0.f) + dh_n[j];
                        float zt = fg_n[1*H+j], nt = fg_n[2*H+j];
                        float dnt = dyo_j*(1.f-zt)*(1.f-nt*nt);
                        float dzt = dyo_j*(-(hp_t?hp_t[n*H+j]:0.f)+nt)*zt*(1.f-zt);
                        float rt  = fg_n[0*H+j];
                        dp_n[0*H+j] = dnt*(hp_t?hp_t[n*H+j]:0.f)*rt*(1.f-rt);  // dr
                        dp_n[1*H+j] = dzt;                                        // dz
                        dp_n[2*H+j] = dnt;                                        // dn_ih
                        dp_n[3*H+j] = dnt*rt;                                     // dn_hh
                        dh_n[j]     = dyo_j * zt;
                    }
                    if (dx_t)
                        for (int s=0;s<3;++s)
                            for (int j=0;j<H;++j)
                                for (int k=0;k<li;++k)
                                    dx_t[n*li+k] += dp_n[s*H+j]*lw.W_ih[s*H*li+j*li+k];
                } else {
                    const float* fg_n = fg_t + n*fs*H;
                    for (int j=0;j<H;++j) {
                        float dyo_j = (dyo_n?dyo_n[j]:0.f)+dh_n[j];
                        float ht = fg_n[j];
                        dp_n[j] = relu ? (ht>0?dyo_j:0.f) : dyo_j*(1.f-ht*ht);
                        dh_n[j] = 0.f;
                        for (int k=0;k<H;++k) dh_n[k] += dp_n[j]*lw.W_hh[j*H+k];
                    }
                    if (dx_t)
                        for (int j=0;j<H;++j)
                            for (int k=0;k<li;++k)
                                dx_t[n*li+k] += dp_n[j]*lw.W_ih[j*li+k];
                }
            }
        }
        if (dhxf) memcpy(dhxf+(size_t)l*B*H, dh_l, (size_t)B*H*sizeof(float));
        if (dcxf) memcpy(dcxf+(size_t)l*B*H, dc_l, (size_t)B*H*sizeof(float));
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(
    cudnnHandle_t,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnRNNDataDescriptor_t xD, const void* x,
    cudnnTensorDescriptor_t,     const void* hx,
    cudnnRNNDataDescriptor_t yD, const void* y,
    void*, size_t,
    void* rsv, size_t rsvSz,
    cudnnFilterDescriptor_t, void* dw,
    void*, size_t)
{
    if (!rnnDesc || !x || !y || !dw || !rsv) return CUDNN_STATUS_INVALID_VALUE;
    (void)hx;
    auto* rd = (RNNDesc*)rnnDesc;
    int H  = rd->hiddenSize;
    int nL = rd->numLayers;
    int T  = (xD && xD->maxSeqLength > 0) ? xD->maxSeqLength : 1;
    int B  = (xD && xD->batchSize    > 0) ? xD->batchSize    : 1;
    int I  = (xD && xD->vectorSize   > 0) ? xD->vectorSize   : H;
    (void)yD;

    cudnnRNNMode_t mode = rd->mode;
    bool isLSTM = (mode == CUDNN_LSTM), isGRU = (mode == CUDNN_GRU);
    int gs = isLSTM ? 4 : (isGRU ? 3 : 1);
    int fs = fwd_slots(mode), bs_slots = bwd_slots(mode);
    size_t lpc_val = lpc(mode, H, I);

    const float* xf  = (const float*)x;
    const float* rsv_f = (const float*)rsv;
    float*       dwf = (float*)dw;

    // Reserve layout: Part A h_store [nL×T×B×H], Part B fwd_gates, Part C bwd_dpre
    size_t partA = (size_t)nL * T * B * H;
    size_t partB = (size_t)nL * T * B * fs * H;
    const float* dpre = rsv_f + partA + partB;  // Part C

    for (int l = 0; l < nL; ++l) {
        int li = (l == 0) ? I : H;
        float* dw_l = dwf + lw_off(mode, H, I, l);
        LWm dlw = get_lwm(mode, H, I, dw_l);

        const float* h_store = rsv_f + l * T * B * H;
        const float* dp_l    = dpre  + l * T * B * bs_slots * H;

        for (int t = 0; t < T; ++t) {
            const float* x_t  = (l == 0) ? xf + t * B * I : (h_store - (size_t)T * B * H) + t * B * H;
            if (l == 0) x_t = xf + t * B * I;
            const float* hp   = (t > 0) ? h_store + (t-1) * B * H : nullptr;
            const float* dp   = dp_l + t * B * bs_slots * H;

            for (int n = 0; n < B; ++n) {
                const float* xt_n = x_t ? x_t + n * li : nullptr;
                const float* hp_n = hp  ? hp  + n * H  : nullptr;
                if (isLSTM) {
                    for (int s = 0; s < 4; ++s) {
                        for (int j = 0; j < H; ++j) {
                            float d = dp[(n*4+s)*H+j];
                            if (!d) continue;
                            dlw.b_ih[s*H+j] += d;
                            if (dlw.b_hh) dlw.b_hh[s*H+j] += d;
                            if (xt_n) for (int k=0;k<li;++k) dlw.W_ih[(s*H+j)*li+k] += d*xt_n[k];
                            if (hp_n) for (int k=0;k<H; ++k) dlw.W_hh[(s*H+j)*H +k] += d*hp_n[k];
                        }
                    }
                } else if (isGRU) {
                    for (int j = 0; j < H; ++j) {
                        float dr=dp[(n*4+0)*H+j], dz=dp[(n*4+1)*H+j];
                        float dn=dp[(n*4+2)*H+j], dnr=dp[(n*4+3)*H+j];
                        dlw.b_ih[0*H+j]+=dr; dlw.b_ih[1*H+j]+=dz; dlw.b_ih[2*H+j]+=dn;
                        if (dlw.b_hh) { dlw.b_hh[0*H+j]+=dr; dlw.b_hh[1*H+j]+=dz; dlw.b_hh[2*H+j]+=dnr; }
                        if (xt_n) for (int k=0;k<li;++k) {
                            dlw.W_ih[(0*H+j)*li+k]+=dr*xt_n[k];
                            dlw.W_ih[(1*H+j)*li+k]+=dz*xt_n[k];
                            dlw.W_ih[(2*H+j)*li+k]+=dn*xt_n[k];
                        }
                        if (hp_n) for (int k=0;k<H;++k) {
                            dlw.W_hh[(0*H+j)*H+k]+=dr *hp_n[k];
                            dlw.W_hh[(1*H+j)*H+k]+=dz *hp_n[k];
                            dlw.W_hh[(2*H+j)*H+k]+=dnr*hp_n[k];
                        }
                    }
                } else {
                    for (int j = 0; j < H; ++j) {
                        float d = dp[(n*1+0)*H+j];
                        if (!d) continue;
                        dlw.b_ih[j] += d;
                        if (xt_n) for (int k=0;k<li;++k) dlw.W_ih[j*li+k] += d*xt_n[k];
                        if (hp_n) for (int k=0;k<H; ++k) dlw.W_hh[j*H +k] += d*hp_n[k];
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── cudnnRNNForward (cuDNN v8.5+ unified API) ─────────────────────────────────
// Dispatches to rnn_forward_ex: training mode fills reserve space; inference
// passes nullptr/0 to skip reserve allocation.
cudnnStatus_t cudnnRNNForward(
    cudnnHandle_t /*handle*/,
    cudnnRNNDescriptor_t rnnDesc,
    cudnnForwardMode_t fwdMode,
    const int32_t* /*devSeqLengths*/,   // optional; sequence lengths embedded in xDesc
    cudnnRNNDataDescriptor_t xDesc, const void* x,
    cudnnRNNDataDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t hDesc, const void* hx, void* hy,
    cudnnTensorDescriptor_t cDesc, const void* cx, void* cy,
    size_t weightSpaceSize, const void* weightSpace,
    size_t workSpaceSize, void* /*workSpace*/,
    size_t reserveSpaceSize, void* reserveSpace)
{
    if (!rnnDesc || !xDesc || !x || !yDesc || !y || !weightSpace)
        return CUDNN_STATUS_INVALID_VALUE;
    (void)hDesc; (void)cDesc; (void)weightSpaceSize; (void)workSpaceSize;

    if (fwdMode == CUDNN_FWD_MODE_TRAINING) {
        return rnn_forward_ex(rnnDesc, xDesc, x,
                              hDesc, hx, cDesc, cx,
                              nullptr, weightSpace,
                              yDesc, y,
                              hDesc, hy, cDesc, cy,
                              reserveSpace, reserveSpaceSize);
    } else {
        return rnn_forward_ex(rnnDesc, xDesc, x,
                              hDesc, hx, cDesc, cx,
                              nullptr, weightSpace,
                              yDesc, y,
                              hDesc, hy, cDesc, cy,
                              nullptr, 0);
    }
}

} // extern "C"
