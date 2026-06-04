// QUEUE-67: cudnnNormalizationBackward full dX/dScale/dBias (BN backward).
// Math: dX[n,c,h,w] = γ[c]·σ⁻¹[c]/M·(M·dY - Σ_all dY - x̂·Σ_all dY·x̂)
//       dScale[c] = Σ_{n,h,w} dY[n,c,h,w]·x̂[n,c,h,w]
//       dBias[c]  = Σ_{n,h,w} dY[n,c,h,w]
// where M = N·H·W, sums over all (n,h,w) for the same channel c.
// Tests verify against hand-computed ground truth.
// Tests:
//   1. N=1 C=1 HW=4: per-element dx matches formula.
//   2. N=2 C=1 HW=2: exposes bug where old code summed over HW only.
//   3. N=2 C=2 HW=1: multi-channel correctness.
//   4. dScale and dBias formulas.
//   5. betaDataDiff accumulation.

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((double)(a)-(double)(b)) < (double)(eps))

static constexpr double EPS = 1e-4;

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;

enum cudnnStatus_t { CUDNN_STATUS_SUCCESS=0, CUDNN_STATUS_INVALID_VALUE=6 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW=0 };
enum cudnnDataType_t { CUDNN_DATA_FLOAT=0 };
enum cudnnNormMode_t { CUDNN_NORM_PER_ACTIVATION=0, CUDNN_NORM_PER_CHANNEL=1 };
enum cudnnNormOps_t  { CUDNN_NORM_OPS_NORM=0 };
enum cudnnNormAlgo_t { CUDNN_NORM_ALGO_STANDARD=0 };

typedef void* cudnnActivationDescriptor_t;

extern "C" {
cudnnStatus_t cudnnCreate(cudnnHandle_t*);
cudnnStatus_t cudnnDestroy(cudnnHandle_t);
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t,
    cudnnTensorFormat_t, cudnnDataType_t, int,int,int,int);
cudnnStatus_t cudnnNormalizationBackward(
    cudnnHandle_t, cudnnNormMode_t, cudnnNormOps_t, cudnnNormAlgo_t,
    const void*, const void*, const void*, const void*,
    const cudnnTensorDescriptor_t, const void*,
    const cudnnTensorDescriptor_t, const void*,
    const cudnnTensorDescriptor_t, const void*,
    const cudnnTensorDescriptor_t, void*,
    const cudnnTensorDescriptor_t, void*,
    const cudnnTensorDescriptor_t,
    const void*, const void*,
    void*, void*,
    double,
    const cudnnTensorDescriptor_t,
    const void*, const void*,
    const cudnnActivationDescriptor_t,
    void*, size_t, void*, size_t);
}

static cudnnHandle_t mkHandle() {
    cudnnHandle_t h = nullptr; cudnnCreate(&h); return h;
}
static cudnnTensorDescriptor_t mkDesc(int n, int c, int h, int w) {
    cudnnTensorDescriptor_t d = nullptr;
    cudnnCreateTensorDescriptor(&d);
    cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n,c,h,w);
    return d;
}

// ── 1. N=1 C=1 HW=4: verify per-element dx ───────────────────────────────────
// Matches hand-calculated values for γ=2, specific dy and xhat.
int test_n1_c1_hw4() {
    // x layout: NCHW, N=1,C=1,H=2,W=2 (HW=4), idx = hw
    // mu=1.5, inv_var=1, xhat = x - 1.5
    float mu[1] = {1.5f}, iv[1] = {1.0f};
    float sc[1] = {2.0f};
    float x[4]  = {0, 1, 2, 3};      // xhat = [-1.5,-0.5,0.5,1.5]
    float dy[4] = {1, 2, 1, 0};       // sum_dy=4, sum_dy_xhat=-1.5*1-0.5*2+0.5*1+1.5*0 = -2
    float dx[4] = {0,0,0,0};
    float dsc[1]={0}, dbi[1]={0};
    float aD=1,bD=0,aP=1,bP=0;
    // M=4, dx[i] = 2*1/4*(4*dy[i]-4-xhat[i]*(-2)) = 0.5*(4*dy[i]-4+2*xhat[i])
    // dx[0] = 0.5*(4-4+2*(-1.5)) = 0.5*(-3) = -1.5
    // dx[1] = 0.5*(8-4+2*(-0.5)) = 0.5*(3) = 1.5
    // dx[2] = 0.5*(4-4+2*0.5)   = 0.5*1   = 0.5
    // dx[3] = 0.5*(0-4+2*1.5)   = 0.5*(-1) = -0.5

    cudnnHandle_t h = mkHandle();
    cudnnTensorDescriptor_t xD = mkDesc(1,1,2,2);
    cudnnTensorDescriptor_t sD = mkDesc(1,1,1,1);

    cudnnStatus_t st = cudnnNormalizationBackward(h,
        CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &aD, &bD, &aP, &bP,
        xD,x, xD,nullptr, xD,dy, nullptr,nullptr, xD,dx,
        sD, sc, nullptr, dsc, dbi, 1e-8,
        sD, mu, iv, nullptr, nullptr,0,nullptr,0);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(sD);
    cudnnDestroy(h);

    if (st != CUDNN_STATUS_SUCCESS) FAIL("status N1C1HW4");
    if (!NEAR(dx[0], -1.5f, EPS)) FAIL("dx[0] got " << dx[0] << " exp -1.5");
    if (!NEAR(dx[1],  1.5f, EPS)) FAIL("dx[1] got " << dx[1] << " exp  1.5");
    if (!NEAR(dx[2],  0.5f, EPS)) FAIL("dx[2] got " << dx[2] << " exp  0.5");
    if (!NEAR(dx[3], -0.5f, EPS)) FAIL("dx[3] got " << dx[3] << " exp -0.5");
    PASS("N=1 C=1 HW=4: per-element dx matches BN formula");
    return 0;
}

// ── 2. N=2 C=1 HW=2: exposes old bug (sum only over HW vs over all N·HW) ────
// With γ=2, mu=1.5, iv=1, x=[0,1,2,3] (N=2,HW=2), dy=[1,2,1,0]
// Correct: same as test 1 since same data (N*HW=4, same sums)
int test_n2_c1_hw2() {
    float mu[1] = {1.5f}, iv[1] = {1.0f};
    float sc[1] = {2.0f};
    // NCHW N=2,C=1,H=1,W=2: idx = n*2+hw
    // n=0: x=[0,1], dy=[1,2]
    // n=1: x=[2,3], dy=[1,0]
    float x[4]  = {0,1,2,3};
    float dy[4] = {1,2,1,0};
    float dx[4] = {0,0,0,0};
    float dsc[1]={0}, dbi[1]={0};
    float aD=1,bD=0,aP=1,bP=0;
    // M=4, sum_dy=4, sum_dy_xhat=-2 (same as test 1)

    cudnnHandle_t h = mkHandle();
    cudnnTensorDescriptor_t xD = mkDesc(2,1,1,2);
    cudnnTensorDescriptor_t sD = mkDesc(1,1,1,1);

    cudnnStatus_t st = cudnnNormalizationBackward(h,
        CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &aD, &bD, &aP, &bP,
        xD,x, xD,nullptr, xD,dy, nullptr,nullptr, xD,dx,
        sD, sc, nullptr, dsc, dbi, 1e-8,
        sD, mu, iv, nullptr, nullptr,0,nullptr,0);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(sD);
    cudnnDestroy(h);

    if (st != CUDNN_STATUS_SUCCESS) FAIL("status N2C1HW2");
    if (!NEAR(dx[0], -1.5f, EPS)) FAIL("dx[0] got " << dx[0] << " exp -1.5");
    if (!NEAR(dx[1],  1.5f, EPS)) FAIL("dx[1] got " << dx[1] << " exp  1.5");
    if (!NEAR(dx[2],  0.5f, EPS)) FAIL("dx[2] got " << dx[2] << " exp  0.5");
    if (!NEAR(dx[3], -0.5f, EPS)) FAIL("dx[3] got " << dx[3] << " exp -0.5");
    PASS("N=2 C=1 HW=2: full-group sum (old bug: summed over HW only)");
    return 0;
}

// ── 3. N=2 C=2 HW=1: multi-channel ──────────────────────────────────────────
// C=2: for c=0: x[0,2]=[1,3], for c=1: x[1,3]=[2,4]
// NCHW idx: n=0,c=0→0; n=0,c=1→1; n=1,c=0→2; n=1,c=1→3
// gamma=[1,1], mu=[2,3], iv=[1,1]
// xhat c=0: [(1-2)*1,(3-2)*1]=[-1,1]; c=1: [(2-3)*1,(4-3)*1]=[-1,1]
// dy=[1,2,3,4]
// c=0: sum_dy=1+3=4, sum_dy_xhat=1*(-1)+3*1=2, M=2
//   dx[0]= 1*1/2*(2*1-4-(-1)*2)=0.5*(2-4+2)=0
//   dx[2]= 0.5*(2*3-4-1*2)=0.5*(6-4-2)=0
// c=1: sum_dy=2+4=6, sum_dy_xhat=2*(-1)+4*1=2, M=2
//   dx[1]= 1*1/2*(2*2-6-(-1)*2)=0.5*(4-6+2)=0
//   dx[3]= 0.5*(2*4-6-1*2)=0.5*(8-6-2)=0
// All dx=0 because dy is uniform-gap+xhat is symmetric here, but dScale/dBias non-trivial
int test_n2_c2_hw1() {
    float mu[2] = {2.f,3.f}, iv[2] = {1.f,1.f};
    float sc[2] = {1.f,1.f};
    float x[4]  = {1,2,3,4};
    float dy[4] = {1,2,3,4};
    float dx[4] = {99,99,99,99};
    float dsc[2]={0,0}, dbi[2]={0,0};
    float aD=1,bD=0,aP=1,bP=0;

    cudnnHandle_t h = mkHandle();
    cudnnTensorDescriptor_t xD = mkDesc(2,2,1,1);
    cudnnTensorDescriptor_t sD = mkDesc(1,2,1,1);

    cudnnNormalizationBackward(h,
        CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &aD, &bD, &aP, &bP,
        xD,x, xD,nullptr, xD,dy, nullptr,nullptr, xD,dx,
        sD, sc, nullptr, dsc, dbi, 1e-8,
        sD, mu, iv, nullptr, nullptr,0,nullptr,0);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(sD);
    cudnnDestroy(h);

    // All dx = 0 for this symmetric config
    for (int i=0;i<4;++i)
        if (!NEAR(dx[i], 0.f, EPS)) FAIL("dx[" << i << "] expected 0 got " << dx[i]);
    // dScale[c=0] = sum_dy_xhat_c0 = 2; dScale[c=1] = sum_dy_xhat_c1 = 2
    if (!NEAR(dsc[0], 2.f, EPS)) FAIL("dScale[0] got " << dsc[0] << " exp 2");
    if (!NEAR(dsc[1], 2.f, EPS)) FAIL("dScale[1] got " << dsc[1] << " exp 2");
    // dBias[c=0] = sum_dy_c0 = 4; dBias[c=1] = sum_dy_c1 = 6
    if (!NEAR(dbi[0], 4.f, EPS)) FAIL("dBias[0] got " << dbi[0] << " exp 4");
    if (!NEAR(dbi[1], 6.f, EPS)) FAIL("dBias[1] got " << dbi[1] << " exp 6");
    PASS("N=2 C=2 HW=1: dX=0, dScale/dBias verified");
    return 0;
}

// ── 4. betaDataDiff accumulation ──────────────────────────────────────────────
int test_beta_accum() {
    float mu[1]={0.f}, iv[1]={1.f}, sc[1]={1.f};
    float x[2]={-1.f,1.f}, dy[2]={1.f,1.f};
    float dx[2]={10.f,20.f};   // pre-filled to test accumulation
    float dsc[1]={0},dbi[1]={0};
    float aD=1,bD=0.5f,aP=1,bP=0;
    // sum_dy=2, sum_dy_xhat=0, M=2
    // dx[i]_new = 1/2*(2*1-2-0)=0, dx[i] = 1*0 + 0.5*old
    // dx[0] = 0.5*10=5; dx[1] = 0.5*20=10

    cudnnHandle_t h = mkHandle();
    cudnnTensorDescriptor_t xD = mkDesc(2,1,1,1);
    cudnnTensorDescriptor_t sD = mkDesc(1,1,1,1);

    cudnnNormalizationBackward(h,
        CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &aD, &bD, &aP, &bP,
        xD,x, xD,nullptr, xD,dy, nullptr,nullptr, xD,dx,
        sD, sc, nullptr, dsc, dbi, 1e-8,
        sD, mu, iv, nullptr, nullptr,0,nullptr,0);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(sD);
    cudnnDestroy(h);

    if (!NEAR(dx[0], 5.f, EPS)) FAIL("betaD accum dx[0] got " << dx[0] << " exp 5");
    if (!NEAR(dx[1], 10.f, EPS)) FAIL("betaD accum dx[1] got " << dx[1] << " exp 10");
    PASS("betaDataDiff accumulation dx[i] = aD*dx_new + bD*dx_old");
    return 0;
}

// ── 5. sum(dx) = 0 per channel (conservation property) ───────────────────────
int test_dx_sum_zero() {
    // For any valid BN backward, Σ_all dx[n,c,h,w] = 0 for each c.
    const int N=4, C=3, HW=8, M=N*HW;
    float mu[C], iv[C], sc[C], x[N*C*HW], dy[N*C*HW], dx[N*C*HW]={};
    for (int c=0;c<C;++c) sc[c] = 1.f+c*0.5f;
    for (int i=0;i<N*C*HW;++i) { x[i]=(float)i*0.1f; dy[i]=(float)(i%17)*0.07f-0.5f; }
    // sum(dx)=0 only when savedMean is the true channel mean — compute it from x.
    for (int c=0;c<C;++c) {
        double s=0, s2=0;
        for (int n=0;n<N;++n) for (int hw=0;hw<HW;++hw) {
            float v = x[(n*C+c)*HW+hw];
            s += v; s2 += (double)v*v;
        }
        float mean = (float)(s/M);
        float var  = (float)(s2/M - (double)mean*mean);
        mu[c] = mean;
        iv[c] = 1.f / std::sqrt(var + 1e-8f);
    }
    float dsc[C]={},dbi[C]={};
    float aD=1,bD=0,aP=1,bP=0;

    cudnnHandle_t h = mkHandle();
    cudnnTensorDescriptor_t xD = mkDesc(N,C,2,4);
    cudnnTensorDescriptor_t sD = mkDesc(1,C,1,1);

    cudnnNormalizationBackward(h,
        CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &aD, &bD, &aP, &bP,
        xD,x, xD,nullptr, xD,dy, nullptr,nullptr, xD,dx,
        sD, sc, nullptr, dsc, dbi, 1e-8,
        sD, mu, iv, nullptr, nullptr,0,nullptr,0);
    cudnnDestroyTensorDescriptor(xD); cudnnDestroyTensorDescriptor(sD);
    cudnnDestroy(h);

    for (int c=0;c<C;++c) {
        double sum=0;
        for (int n=0;n<N;++n) for (int hw=0;hw<HW;++hw)
            sum += dx[(n*C+c)*HW+hw];
        if (std::fabs(sum) > 1e-3)
            FAIL("sum(dx) for c=" << c << " = " << sum << " != 0");
    }
    PASS("sum(dx) = 0 per channel (N=4 C=3 HW=8)");
    return 0;
}

int main() {
    std::cout << "=== cudnnNormalizationBackward full BN tests (QUEUE-67) ===\n";
    int fails = 0;
    fails += test_n1_c1_hw4();
    fails += test_n2_c1_hw2();
    fails += test_n2_c2_hw1();
    fails += test_beta_accum();
    fails += test_dx_sum_zero();
    if (fails == 0) std::cout << "All QUEUE-67 BN backward tests passed.\n";
    return fails;
}
