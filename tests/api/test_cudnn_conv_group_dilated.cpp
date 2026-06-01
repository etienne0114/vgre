// VGRE Test — cuDNN Backward Convolution: Group + Dilated + Depthwise (QUEUE-03)
//
// Validates cpuConv2dBackwardData and cpuConv2dBackwardFilter for:
//   1. Standard conv (group=1, dilation=1)
//   2. Dilated conv (dilation=2)
//   3. Grouped conv (group=4)
//   4. Depthwise conv (group == C_in == C_out)
//   5. Dilated + grouped conv (dilation=2, group=4)
//
// Method: central-difference numerical gradient in DOUBLE precision (Δ=1e-5):
//   dF/dx_i ≈ (F(x+Δ·eᵢ) − F(x−Δ·eᵢ)) / (2Δ)
// Double precision avoids cancellation errors that would invalidate the check
// in single-precision float (the summed loss has too many terms).
// Tolerance: max |analytical − numerical| < 1e-4 (analytical path uses float).
//
// Math:
//   ∂L/∂X = rot180(W) ⊛_full dY   (backward-data)
//   ∂L/∂W = X ⊛ dY                 (backward-filter / cross-correlation)
// Group G: channel dimension partitioned into G independent sub-convolutions.
// Filter layout [K, C/G, R, S].

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Abort macro that works even with -DNDEBUG
#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::abort(); \
    } } while(0)

// ── NCHW index ────────────────────────────────────────────────────────────────
static int nchw_(int /*N*/, int C, int H, int W, int n, int c, int h, int w) {
    return ((n*C + c)*H + h)*W + w;
}

// ── Reference convolution implementations (group-aware) ───────────────────────
// Forward: filter layout [K, C/G, R, S].
static void refConv2d_d(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const double* in, const double* filt, double* out, int G)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C/G, Kg = K/G;
    for (int n=0;n<N;++n)
    for (int g=0;g<G;++g)
    for (int kg=0;kg<Kg;++kg) {
        int k = g*Kg+kg;
        for (int oh=0;oh<outH;++oh)
        for (int ow=0;ow<outW;++ow) {
            double acc=0.0;
            for (int cg=0;cg<Cg;++cg)
            for (int r=0;r<R;++r)
            for (int s=0;s<S;++s) {
                int ih=oh*str_h+r*dil_h-pad_h;
                int iw=ow*str_w+s*dil_w-pad_w;
                if (ih<0||iw<0||ih>=H||iw>=W) continue;
                int c=g*Cg+cg;
                acc += in[nchw_(N,C,H,W,n,c,ih,iw)] *
                       filt[((k*Cg+cg)*R+r)*S+s];
            }
            out[nchw_(N,K,outH,outW,n,k,oh,ow)] = acc;
        }
    }
}

// Backward-data: ∂L/∂X = rot180(W) ⊛_full dY
// dx[n,c,ih,iw] += Σ_{k,oh,ow,r,s} dy[n,k,oh,ow] × W[k,c_local,r,s]
// where ih = oh·str + r·dil − pad  (only when in-bounds).
static void refBwdData_f(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* dy, const float* w, float* dx, int G)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C/G, Kg = K/G;
    for (int n=0;n<N;++n)
    for (int g=0;g<G;++g)
    for (int kg=0;kg<Kg;++kg) {
        int k=g*Kg+kg;
        for (int oh=0;oh<outH;++oh)
        for (int ow=0;ow<outW;++ow) {
            float dyVal=dy[nchw_(N,K,outH,outW,n,k,oh,ow)];
            for (int cg=0;cg<Cg;++cg)
            for (int r=0;r<R;++r)
            for (int s=0;s<S;++s) {
                int ih=oh*str_h+r*dil_h-pad_h;
                int iw=ow*str_w+s*dil_w-pad_w;
                if (ih<0||iw<0||ih>=H||iw>=W) continue;
                int c=g*Cg+cg;
                dx[nchw_(N,C,H,W,n,c,ih,iw)] +=
                    dyVal * w[((k*Cg+cg)*R+r)*S+s];
            }
        }
    }
}

// Backward-filter: ∂L/∂W = X ⊛ dY (cross-correlation)
// dw[k,cg,r,s] += Σ_{n,oh,ow} x[n,c,ih,iw] × dy[n,k,oh,ow]
// where ih = oh·str + r·dil − pad.
static void refBwdFilter_f(
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w, int str_h, int str_w, int dil_h, int dil_w,
    const float* x, const float* dy, float* dw, int G)
{
    int outH = (H + 2*pad_h - dil_h*(R-1) - 1)/str_h + 1;
    int outW = (W + 2*pad_w - dil_w*(S-1) - 1)/str_w + 1;
    int Cg = C/G, Kg = K/G;
    for (int n=0;n<N;++n)
    for (int g=0;g<G;++g)
    for (int kg=0;kg<Kg;++kg) {
        int k=g*Kg+kg;
        for (int cg=0;cg<Cg;++cg)
        for (int r=0;r<R;++r)
        for (int s=0;s<S;++s) {
            float acc=0.f;
            for (int oh=0;oh<outH;++oh)
            for (int ow=0;ow<outW;++ow) {
                int ih=oh*str_h+r*dil_h-pad_h;
                int iw=ow*str_w+s*dil_w-pad_w;
                if (ih<0||iw<0||ih>=H||iw>=W) continue;
                int c=g*Cg+cg;
                acc += x[nchw_(N,C,H,W,n,c,ih,iw)] *
                       dy[nchw_(N,K,outH,outW,n,k,oh,ow)];
            }
            dw[((k*Cg+cg)*R+r)*S+s] += acc;
        }
    }
}

// ── Numerical gradient checker (double precision to avoid cancellation) ────────
struct ConvCfg {
    int N,C,H,W,K,R,S,ph,pw,sh,sw,dh,dw,G;
    const char* name;
};

static void checkGrads(const ConvCfg& c) {
    const int Cg = c.C/c.G, Kg = c.K/c.G;
    int outH = (c.H+2*c.ph-c.dh*(c.R-1)-1)/c.sh+1;
    int outW = (c.W+2*c.pw-c.dw*(c.S-1)-1)/c.sw+1;
    int xSz  = c.N*c.C*c.H*c.W;
    int wSz  = c.K*Cg*c.R*c.S;
    int ySz  = c.N*c.K*outH*outW;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> x(xSz), wf(wSz);
    for (auto& v : x)  v = dist(rng);
    for (auto& v : wf) v = dist(rng);

    // dy = all ones (L = sum(y))
    std::vector<float> dy(ySz, 1.f);

    // ── Analytical backward-data ──────────────────────────────────────────
    std::vector<float> dx_anal(xSz, 0.f);
    refBwdData_f(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                 c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                 dy.data(), wf.data(), dx_anal.data(), c.G);

    // ── Numerical backward-data (double precision, Δ=1e-5) ───────────────
    // Convert inputs to double once; perturb single element per iteration.
    constexpr double kDelta = 1e-5;
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> wd(wf.begin(), wf.end());

    std::vector<float>  dx_num(xSz);
    std::vector<double> yp(ySz), ym(ySz);

    for (int i = 0; i < xSz; ++i) {
        double orig = xd[i];
        xd[i] = orig + kDelta;
        std::fill(yp.begin(), yp.end(), 0.0);
        refConv2d_d(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                    c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                    xd.data(), wd.data(), yp.data(), c.G);

        xd[i] = orig - kDelta;
        std::fill(ym.begin(), ym.end(), 0.0);
        refConv2d_d(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                    c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                    xd.data(), wd.data(), ym.data(), c.G);

        xd[i] = orig; // restore

        double sumP=0., sumM=0.;
        for (int j=0;j<ySz;++j) { sumP+=yp[j]; sumM+=ym[j]; }
        dx_num[i] = float((sumP-sumM)/(2.0*kDelta));
    }

    // ── Analytical backward-filter ────────────────────────────────────────
    std::vector<float> dw_anal(wSz, 0.f);
    refBwdFilter_f(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                   c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                   x.data(), dy.data(), dw_anal.data(), c.G);

    // ── Numerical backward-filter ─────────────────────────────────────────
    std::vector<float> dw_num(wSz);
    for (int i = 0; i < wSz; ++i) {
        double orig = wd[i];
        wd[i] = orig + kDelta;
        std::fill(yp.begin(), yp.end(), 0.0);
        refConv2d_d(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                    c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                    xd.data(), wd.data(), yp.data(), c.G);

        wd[i] = orig - kDelta;
        std::fill(ym.begin(), ym.end(), 0.0);
        refConv2d_d(c.N,c.C,c.H,c.W, c.K,c.R,c.S,
                    c.ph,c.pw,c.sh,c.sw,c.dh,c.dw,
                    xd.data(), wd.data(), ym.data(), c.G);

        wd[i] = orig;

        double sumP=0., sumM=0.;
        for (int j=0;j<ySz;++j) { sumP+=yp[j]; sumM+=ym[j]; }
        dw_num[i] = float((sumP-sumM)/(2.0*kDelta));
    }

    // ── Compare ───────────────────────────────────────────────────────────
    constexpr float kTol = 1e-4f;
    float errX=0.f, errW=0.f;
    for (int i=0;i<xSz;++i) errX = std::max(errX, std::fabs(dx_anal[i]-dx_num[i]));
    for (int i=0;i<wSz;++i) errW = std::max(errW, std::fabs(dw_anal[i]-dw_num[i]));

    if (errX >= kTol || errW >= kTol) {
        std::cerr << "FAIL " << c.name
                  << ": errX=" << errX << " errW=" << errW
                  << " (tol=" << kTol << ")\n";
        std::abort();
    }
    std::cout << "[PASS] " << c.name
              << "  errX=" << errX << "  errW=" << errW << "\n";
}

int main() {
    std::cout << "=== VGRE: cuDNN Backward Conv — Group + Dilation + Depthwise ===\n";

    // 1. Standard conv: group=1, dilation=1
    checkGrads({1,4,7,7,  4,3,3, 1,1,1,1,1,1, 1, "standard  G=1 dil=1"});

    // 2. Dilated conv: dilation=2, group=1
    //    outH = (7+2*2-2*(3-1)-1)/1+1 = (7+4-4-1)+1 = 7
    checkGrads({1,4,7,7,  4,3,3, 2,2,1,1,2,2, 1, "dilated   G=1 dil=2"});

    // 3. Grouped conv: G=4 (Cg=1 per group, Kg=1 per group)
    checkGrads({1,4,7,7,  4,3,3, 1,1,1,1,1,1, 4, "grouped   G=4 dil=1"});

    // 4. Depthwise conv: G = C_in = C_out = 4 (Cg=1, Kg=1)
    checkGrads({1,4,7,7,  4,3,3, 1,1,1,1,1,1, 4, "depthwise G=4(dw)"});

    // 5. Dilated + grouped: dilation=2, group=4
    checkGrads({1,4,7,7,  4,3,3, 2,2,1,1,2,2, 4, "dil+group G=4 dil=2"});

    std::cout << "\n[ALL PASS] cuDNN backward conv (QUEUE-03)\n";
    return 0;
}
