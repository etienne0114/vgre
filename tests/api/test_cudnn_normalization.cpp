// cuDNN 8.5+ Normalization API tests.
// Covers: ForwardInference, ForwardTraining, Backward
// for both CUDNN_NORM_PER_CHANNEL (batch norm) and CUDNN_NORM_PER_ACTIVATION (layer norm).

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((a)-(b)) < (eps))

extern "C" {
typedef int    cudnnStatus_t;
typedef void*  cudnnHandle_t;
typedef void*  cudnnTensorDescriptor_t;
typedef void*  cudnnActivationDescriptor_t;

#define CUDNN_STATUS_SUCCESS       0
#define CUDNN_STATUS_INVALID_VALUE 8

enum cudnnDataType_t     { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };
enum cudnnNormMode_t     { CUDNN_NORM_PER_ACTIVATION = 0, CUDNN_NORM_PER_CHANNEL = 1 };
enum cudnnNormAlgo_t     { CUDNN_NORM_ALGO_STANDARD = 0 };
enum cudnnNormOps_t      { CUDNN_NORM_OPS_NORM = 0, CUDNN_NORM_OPS_NORM_ACTIVATION = 1 };
enum cudnnActivationMode_t { CUDNN_ACTIVATION_RELU = 1 };
enum cudnnNanPropagation_t { CUDNN_NOT_PROPAGATE_NAN = 0 };

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *d);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d,
    cudnnActivationMode_t mode, cudnnNanPropagation_t nan, double coeff);

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

// Inference: normMeanVarDesc == estimatedMean/Variance descriptor
cudnnStatus_t cudnnNormalizationForwardInference(
    cudnnHandle_t handle,
    cudnnNormMode_t mode, cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    const void *estimatedMean, const void *estimatedVariance,
    double epsilon,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnActivationDescriptor_t actDesc,
    const cudnnTensorDescriptor_t yDesc, void *y,
    void *workspace, size_t workspaceSizeInBytes);

// Training: normMeanVarDesc between exponentialAverageFactor and resultRunning*
cudnnStatus_t cudnnNormalizationForwardTraining(
    cudnnHandle_t handle,
    cudnnNormMode_t mode, cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    const cudnnActivationDescriptor_t actDesc,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t yDesc, void *y,
    void *workspace, size_t workspaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes);

// Backward: resultBnScaleDiff/BiasDiff before epsilon; savedMeanVarDesc + saved* after
cudnnStatus_t cudnnNormalizationBackward(
    cudnnHandle_t handle,
    cudnnNormMode_t mode, cudnnNormOps_t normOps, cudnnNormAlgo_t algo,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dzDesc, void *dz,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    void *resultBnScaleDiff, void *resultBnBiasDiff,
    double epsilon,
    const cudnnTensorDescriptor_t savedMeanVarDesc,
    const void *savedMean, const void *savedInvVariance,
    const cudnnActivationDescriptor_t actDesc,
    void *workspace, size_t workspaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes);
}

// ── 1. PER_CHANNEL ForwardInference (batch norm with pre-computed stats) ──────
// N=2, C=2, H=1, W=1
// x[0,0]=1, x[0,1]=3, x[1,0]=5, x[1,1]=7  (NCHW: n*C+c)
// mean=[3,5], var=[4,4], scale=[1,1], bias=[0,0]
// xhat[n,c]=(x-mean[c])/sqrt(var[c]+eps): y[0,0]=-1, y[0,1]=-1, y[1,0]=1, y[1,1]=1
int test_per_channel_inference() {
    cudnnHandle_t h = nullptr;
    if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS || !h) FAIL("create handle");

    cudnnTensorDescriptor_t xDesc, yDesc, scDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&scDesc);
    cudnnSetTensor4dDescriptor(xDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 1, 1);
    cudnnSetTensor4dDescriptor(yDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 1, 1);
    cudnnSetTensor4dDescriptor(scDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);

    std::vector<float> x   = {1.0f, 3.0f, 5.0f, 7.0f};
    std::vector<float> y(4, 0.0f);
    std::vector<float> sc  = {1.0f, 1.0f};
    std::vector<float> bi  = {0.0f, 0.0f};
    std::vector<float> mu  = {3.0f, 5.0f};
    std::vector<float> var = {4.0f, 4.0f};

    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t st = cudnnNormalizationForwardInference(
        h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &alpha, &beta,
        xDesc, x.data(),
        scDesc, sc.data(), bi.data(),
        scDesc, mu.data(), var.data(),
        1e-6,
        nullptr, nullptr, nullptr,
        yDesc, y.data(),
        nullptr, 0);
    if (st != CUDNN_STATUS_SUCCESS) FAIL("PER_CHANNEL inference returned error " << st);

    if (!NEAR(y[0], -1.0f, 1e-4f)) FAIL("y[0,0] expected -1 got " << y[0]);
    if (!NEAR(y[1], -1.0f, 1e-4f)) FAIL("y[0,1] expected -1 got " << y[1]);
    if (!NEAR(y[2],  1.0f, 1e-4f)) FAIL("y[1,0] expected  1 got " << y[2]);
    if (!NEAR(y[3],  1.0f, 1e-4f)) FAIL("y[1,1] expected  1 got " << y[3]);

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(scDesc);
    cudnnDestroy(h);
    PASS("PER_CHANNEL ForwardInference 2x2x1x1");
    return 0;
}

// ── 2. PER_ACTIVATION (layer norm) ForwardInference ──────────────────────────
// N=2, C=2, H=1, W=1  (D = C*H*W = 2 per sample)
// x[n=0]=[1,3] → per-sample mean=2, var=1 → xhat=[-1,1] → y=2*xhat+1=[-1,3]
// x[n=1]=[5,7] → per-sample mean=6, var=1 → xhat=[-1,1] → y=2*xhat+1=[-1,3]
int test_per_activation_inference() {
    cudnnHandle_t h = nullptr;
    cudnnCreate(&h);

    cudnnTensorDescriptor_t xDesc, yDesc, scDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&scDesc);
    cudnnSetTensor4dDescriptor(xDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 1, 1);
    cudnnSetTensor4dDescriptor(yDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 2, 1, 1);
    cudnnSetTensor4dDescriptor(scDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);

    std::vector<float> x   = {1.0f, 3.0f, 5.0f, 7.0f};
    std::vector<float> y(4, 0.0f);
    std::vector<float> sc  = {2.0f, 2.0f};
    std::vector<float> bi  = {1.0f, 1.0f};
    std::vector<float> mu  = {2.0f, 6.0f};
    std::vector<float> var = {1.0f, 1.0f};

    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t st = cudnnNormalizationForwardInference(
        h, CUDNN_NORM_PER_ACTIVATION, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &alpha, &beta,
        xDesc, x.data(),
        scDesc, sc.data(), bi.data(),
        xDesc, mu.data(), var.data(),
        1e-6,
        nullptr, nullptr, nullptr,
        yDesc, y.data(),
        nullptr, 0);
    if (st != CUDNN_STATUS_SUCCESS) FAIL("PER_ACTIVATION inference returned error " << st);

    if (!NEAR(y[0], -1.0f, 1e-4f)) FAIL("PER_ACT y[0] expected -1 got " << y[0]);
    if (!NEAR(y[1],  3.0f, 1e-4f)) FAIL("PER_ACT y[1] expected  3 got " << y[1]);
    if (!NEAR(y[2], -1.0f, 1e-4f)) FAIL("PER_ACT y[2] expected -1 got " << y[2]);
    if (!NEAR(y[3],  3.0f, 1e-4f)) FAIL("PER_ACT y[3] expected  3 got " << y[3]);

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(scDesc);
    cudnnDestroy(h);
    PASS("PER_ACTIVATION ForwardInference 2x2x1x1");
    return 0;
}

// ── 3. PER_CHANNEL ForwardTraining: updates running stats, saves mean ─────────
// N=2, C=1, H=1, W=1  x=[1,5]  batch_mean=3, batch_var=4
// EMA=1.0 → runMean = 1.0*3 + 0.0*0 = 3
int test_per_channel_training() {
    cudnnHandle_t h = nullptr;
    cudnnCreate(&h);

    cudnnTensorDescriptor_t xDesc, yDesc, scDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&scDesc);
    cudnnSetTensor4dDescriptor(xDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 1, 1);
    cudnnSetTensor4dDescriptor(yDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 1, 1);
    cudnnSetTensor4dDescriptor(scDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    std::vector<float> x       = {1.0f, 5.0f};
    std::vector<float> y(2, 0.0f);
    std::vector<float> sc      = {1.0f};
    std::vector<float> bi      = {0.0f};
    std::vector<float> runMean = {0.0f};
    std::vector<float> runVar  = {1.0f};
    std::vector<float> saveMean(1, 0.0f);
    std::vector<float> saveInvVar(1, 0.0f);

    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t st = cudnnNormalizationForwardTraining(
        h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &alpha, &beta,
        xDesc, x.data(),
        scDesc, sc.data(), bi.data(),
        1.0,        // EMA factor = 1.0 → full replace
        scDesc,     // normMeanVarDesc
        runMean.data(), runVar.data(),
        1e-6,
        saveMean.data(), saveInvVar.data(),
        nullptr,    // actDesc
        nullptr, nullptr,  // zDesc, z
        yDesc, y.data(),
        nullptr, 0, // workspace
        nullptr, 0  // reserveSpace
    );
    if (st != CUDNN_STATUS_SUCCESS) FAIL("PER_CHANNEL training returned error " << st);

    if (!NEAR(saveMean[0], 3.0f, 1e-4f))
        FAIL("saveMean expected 3 got " << saveMean[0]);
    if (!NEAR(runMean[0],  3.0f, 1e-4f))
        FAIL("runMean expected 3 got " << runMean[0]);
    // y[0]=(1-3)/2=-1, y[1]=(5-3)/2=1
    if (!NEAR(y[0], -1.0f, 1e-3f)) FAIL("training y[0] expected -1 got " << y[0]);
    if (!NEAR(y[1],  1.0f, 1e-3f)) FAIL("training y[1] expected  1 got " << y[1]);

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(scDesc);
    cudnnDestroy(h);
    PASS("PER_CHANNEL ForwardTraining: running stats + saveMean");
    return 0;
}

// ── 4. PER_CHANNEL Backward: dScale, dBias, dx ───────────────────────────────
// N=2, C=1, H=1, W=1  x=[1,5]  saveMean=3, saveInvVar=0.5
// dy=[1,1] → dBias=sum(dy)=2, dScale=sum(dy*xhat)=0, dx≈0
int test_per_channel_backward() {
    cudnnHandle_t h = nullptr;
    cudnnCreate(&h);

    cudnnTensorDescriptor_t xDesc, yDesc, scDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&scDesc);
    cudnnSetTensor4dDescriptor(xDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 1, 1);
    cudnnSetTensor4dDescriptor(yDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 1, 1, 1);
    cudnnSetTensor4dDescriptor(scDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    std::vector<float> x         = {1.0f, 5.0f};
    std::vector<float> y         = {-1.0f, 1.0f};
    std::vector<float> dy        = {1.0f, 1.0f};
    std::vector<float> dx(2, 0.0f);
    std::vector<float> sc        = {1.0f};
    std::vector<float> bi        = {0.0f};
    std::vector<float> dsc(1, 0.0f), dbi(1, 0.0f);
    std::vector<float> saveMean    = {3.0f};
    std::vector<float> saveInvVar  = {0.5f};

    float alphaD = 1.0f, betaD = 0.0f;
    float alphaP = 1.0f, betaP = 0.0f;
    cudnnStatus_t st = cudnnNormalizationBackward(
        h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &alphaD, &betaD, &alphaP, &betaP,
        xDesc, x.data(),
        yDesc, y.data(),
        yDesc, dy.data(),
        nullptr, nullptr,   // dzDesc, dz
        xDesc, dx.data(),
        scDesc, sc.data(), bi.data(),
        dsc.data(), dbi.data(),
        1e-6,
        scDesc,             // savedMeanVarDesc
        saveMean.data(), saveInvVar.data(),
        nullptr,            // actDesc
        nullptr, 0,         // workspace
        nullptr, 0          // reserveSpace
    );
    if (st != CUDNN_STATUS_SUCCESS) FAIL("PER_CHANNEL backward returned error " << st);

    if (!NEAR(dbi[0], 2.0f, 1e-3f)) FAIL("dBias expected 2 got " << dbi[0]);
    if (!NEAR(dsc[0], 0.0f, 1e-3f)) FAIL("dScale expected 0 got " << dsc[0]);
    if (!NEAR(dx[0] + dx[1], 0.0f, 1e-3f)) FAIL("sum(dx) expected ~0");

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(scDesc);
    cudnnDestroy(h);
    PASS("PER_CHANNEL Backward dScale/dBias/dx");
    return 0;
}

// ── 5. Invalid value guard ─────────────────────────────────────────────────────
int test_invalid_value() {
    cudnnHandle_t h = nullptr;
    cudnnCreate(&h);
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    float alpha = 1.0f, beta = 0.0f;
    float dummy = 0.0f;
    cudnnStatus_t st = cudnnNormalizationForwardInference(
        h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
        &alpha, &beta,
        desc, nullptr,       // null x → INVALID_VALUE
        desc, &dummy, &dummy,
        desc, &dummy, &dummy,
        1e-6, nullptr, nullptr, nullptr,
        desc, &dummy, nullptr, 0);
    if (st != CUDNN_STATUS_INVALID_VALUE) FAIL("expected INVALID_VALUE for null x");

    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(h);
    PASS("invalid value guard (null x)");
    return 0;
}

// ── 6. Fused BN+ReLU inference — QUEUE-32 ────────────────────────────────────
// Verifies single-pass fused output matches two-pass reference within 1e-5.
// Formula: out[i] = max(0, (x[i]-mean[c])/sqrt(var[c]+eps)*scale[c]+bias[c])
// Test uses N=4, C=4, H=4, W=4 (256 elements) for fast CI; validates all paths.
int test_fused_bn_relu_inference() {
    const int N=4, C=4, H=4, W=4, HW=H*W, total=N*C*HW;
    const float eps = 1e-5f;

    // Construct input with mix of positive/negative to exercise ReLU gate
    std::vector<float> x(total), scale(C), bias(C), mean(C), var(C);
    for (int i = 0; i < total; ++i)
        x[i] = (i % 7 < 3) ? -(float)(i % 5 + 1) : (float)(i % 13 + 1);
    for (int c = 0; c < C; ++c) {
        scale[c] = 0.5f + 0.1f * c;
        bias[c]  = -0.3f + 0.2f * c;
        mean[c]  = 0.0f;
        var[c]   = 1.0f;
        // Compute true per-channel mean and variance for reference
        double s = 0.0, s2 = 0.0;
        int cnt = N * HW;
        for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HW; ++hw) {
                float v = x[(n * C + c) * HW + hw];
                s += v; s2 += v * v;
            }
        mean[c] = (float)(s / cnt);
        var[c]  = (float)(s2 / cnt - (double)mean[c] * mean[c]);
    }

    // ── Reference: two-pass BN then separate ReLU ────────────────────────────
    std::vector<float> ref(total);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int hw = 0; hw < HW; ++hw) {
                int idx = (n * C + c) * HW + hw;
                float xhat = (x[idx] - mean[c]) / std::sqrt(var[c] + eps);
                float y = scale[c] * xhat + bias[c];
                ref[idx] = y > 0.f ? y : 0.f;  // ReLU separately
            }

    // ── Fused: single-pass BN+ReLU via cudnnNormalizationForwardInference ────
    cudnnHandle_t h = nullptr; cudnnCreate(&h);
    cudnnTensorDescriptor_t xDesc, yDesc, scDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&scDesc);
    cudnnSetTensor4dDescriptor(xDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(yDesc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(scDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    std::vector<float> fused(x);  // in-place output
    float alpha = 1.f, beta = 0.f;
    auto st = cudnnNormalizationForwardInference(
        h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM_ACTIVATION,
        CUDNN_NORM_ALGO_STANDARD, &alpha, &beta,
        xDesc, fused.data(),
        scDesc, scale.data(), bias.data(),
        scDesc, mean.data(), var.data(),
        eps, nullptr, nullptr, actDesc,
        yDesc, fused.data(), nullptr, 0);

    if (st != CUDNN_STATUS_SUCCESS) FAIL("cudnnNormalizationForwardInference returned " << st);

    // Verify fused output matches reference within 1e-5
    float max_diff = 0.f;
    for (int i = 0; i < total; ++i)
        max_diff = std::fmax(max_diff, std::fabs(fused[i] - ref[i]));
    if (max_diff >= 1e-4f) FAIL("max |fused - ref| = " << max_diff << " >= 1e-4");

    cudnnDestroyActivationDescriptor(actDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(scDesc);
    cudnnDestroy(h);
    PASS("Fused BN+ReLU inference (4×4×4×4, max_diff=" << max_diff << ")");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_per_channel_inference();
    failures += test_per_activation_inference();
    failures += test_per_channel_training();
    failures += test_per_channel_backward();
    failures += test_invalid_value();
    // QUEUE-32: fused BN+ReLU single-pass inference
    failures += test_fused_bn_relu_inference();
    if (failures == 0)
        std::cout << "All cudnnNormalization tests passed.\n";
    return failures;
}
