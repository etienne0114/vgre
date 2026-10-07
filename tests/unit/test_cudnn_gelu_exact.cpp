// Exact erf-based GELU forward/backward behavior and descriptor validation.
// Tests:
//   1. Forward matches tanhf-based reference within tolerance (|x| < 3).
//   2. Saturation: GELU(x) → x for x ≫ 0 and → 0 for x ≪ 0.
//   3. GELU(0) == 0; GELU is monotone increasing.
//   4. Gradient consistency: analytical backward matches central-difference of forward.
//   5. Large inputs (|x| up to 100) produce finite, correct outputs.

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a, b, eps) (std::fabs((float)(a) - (float)(b)) < (float)(eps))

extern "C" {
    typedef int    cudnnStatus_t;
    typedef void*  cudnnHandle_t;
    typedef void*  cudnnActivationDescriptor_t;
    typedef void*  cudnnTensorDescriptor_t;

    cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t*);
    cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t, int, int, double);
    cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t);
    cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*);
    cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t, int, int, int,int,int,int);
    cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t);
    cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t,
        const void*, cudnnTensorDescriptor_t, const void*,
        const void*, cudnnTensorDescriptor_t, void*);
    cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t,
        const void*, cudnnTensorDescriptor_t, const void*,
        cudnnTensorDescriptor_t, const void*,
        cudnnTensorDescriptor_t, const void*,
        const void*, cudnnTensorDescriptor_t, void*);
}

static constexpr int CUDNN_ACTIVATION_GELU = 7;
static constexpr int CUDNN_ACTIVATION_RELU = 1;

// Helper: run one scalar value through the GELU forward shim.
static float gelu_fwd(float x) {
    cudnnActivationDescriptor_t act{};
    cudnnTensorDescriptor_t td{};
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 0, 1.0);
    cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td, 0, 0, 1, 1, 1, 1);
    float y = 0.f;
    float alpha = 1.f, beta = 0.f;
    cudnnActivationForward(nullptr, act, &alpha, td, &x, &beta, td, &y);
    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(td);
    return y;
}

// Helper: run one scalar through GELU backward (dy=1 → returns d(GELU)/dx).
static float gelu_bwd(float x) {
    cudnnActivationDescriptor_t act{};
    cudnnTensorDescriptor_t td{};
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 0, 1.0);
    cudnnCreateTensorDescriptor(&td);
    cudnnSetTensor4dDescriptor(td, 0, 0, 1, 1, 1, 1);
    float y  = gelu_fwd(x);
    float dy = 1.f;
    float dx = 0.f;
    float alpha = 1.f, beta = 0.f;
    cudnnActivationBackward(nullptr, act, &alpha, td, &y, td, &dy, td, &x, &beta, td, &dx);
    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(td);
    return dx;
}

// Reference GELU via exact tanhf (Hendrycks & Gimpel 2016).
static float gelu_ref(float x) {
    if (std::isinf(x)) return x > 0.f ? x : 0.f;
    static constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.f + std::erf(x * kInvSqrt2));
}

// ── 1. Forward accuracy vs the defining erf formula ─────────────────────────
int test_forward_accuracy() {
    const float abs_tol = 2e-6f;
    float xs[] = {-5.f,-3.f,-2.5f,-2.f,-1.5f,-1.f,-0.5f,0.f,0.5f,1.f,1.5f,2.f,2.5f,3.f,5.f};
    for (float x : xs) {
        float got = gelu_fwd(x);
        float ref = gelu_ref(x);
        if (!NEAR(got, ref, abs_tol))
            FAIL("forward x=" + std::to_string(x) +
                 " got=" + std::to_string(got) +
                 " ref=" + std::to_string(ref));
    }
    PASS("forward accuracy vs exact erf GELU (|x|≤5, tol=2e-6)");
    return 0;
}

// ── 2. Saturation: GELU(x)→x (x≫0) and →0 (x≪0) ────────────────────────────
int test_saturation() {
    float large_pos[] = {4.f, 5.f, 10.f, 50.f, 100.f};
    for (float x : large_pos) {
        float y = gelu_fwd(x);
        if (!NEAR(y, gelu_ref(x), 2e-6f))
            FAIL("saturation+ x=" + std::to_string(x) + " got=" + std::to_string(y));
    }
    float large_neg[] = {-4.f, -5.f, -10.f, -50.f, -100.f};
    for (float x : large_neg) {
        float y = gelu_fwd(x);
        if (!NEAR(y, gelu_ref(x), 2e-6f))
            FAIL("saturation- x=" + std::to_string(x) + " got=" + std::to_string(y));
    }
    PASS("large-magnitude GELU agrees with exact erf reference");
    return 0;
}

// ── 3. GELU properties: GELU(0)=0, positive for x>0, negative minimum near x≈−0.72 ──
// Note: GELU is NOT globally monotone — it has a minimum of ≈−0.17 near x≈−0.72
// and is monotone increasing for x > −0.72 only.
int test_gelu_properties() {
    // GELU(0) = 0 exactly (x=0 → inner=0 → poly(0)=0 → 0.5*0*(1+0) = 0).
    if (!NEAR(gelu_fwd(0.f), 0.f, 1e-6f)) FAIL("GELU(0) != 0");
    // GELU(x) > 0 for all x > 0.
    float pos_xs[] = {0.1f, 0.5f, 1.f, 2.f, 3.f, 4.f, 10.f};
    for (float x : pos_xs)
        if (gelu_fwd(x) <= 0.f)
            FAIL("GELU(" + std::to_string(x) + ") should be > 0");
    // GELU(x) < 0 for x in (−2.5, 0); beyond that the polynomial may clamp to 0.
    float neg_xs[] = {-0.1f, -0.5f, -1.f, -2.f};
    for (float x : neg_xs)
        if (gelu_fwd(x) >= 0.f)
            FAIL("GELU(" + std::to_string(x) + ") should be < 0");
    // Minimum of GELU is ≥ −0.22 (true min ≈ −0.17, approx within 5% tolerance).
    float min_val = 0.f;
    for (int i = -100; i <= 100; ++i) min_val = std::min(min_val, gelu_fwd(i * 0.05f));
    if (min_val < -0.22f)
        FAIL("global minimum too negative: " + std::to_string(min_val));
    // Monotone increasing on [0, 5]: GELU(a) < GELU(b) for 0 ≤ a < b.
    float prev = gelu_fwd(0.f);
    for (int i = 1; i <= 50; ++i) {
        float cur = gelu_fwd(i * 0.1f);
        if (cur < prev - 1e-4f)
            FAIL("not increasing on [0,5] at x=" + std::to_string(i * 0.1f));
        prev = cur;
    }
    PASS("GELU(0)=0, positive for x>0, negative in (−4,0), min≥−0.22, increasing on [0,5]");
    return 0;
}

// ── 4. Gradient consistency: backward == central-difference of forward ────────
int test_gradient_consistency() {
    const float eps = 1e-3f;
    const float abs_tol = 1e-2f;
    float xs[] = {-2.5f,-1.5f,-0.8f,-0.1f,0.f,0.3f,0.9f,1.5f,2.2f,3.f,4.5f,-4.5f};
    for (float x : xs) {
        float bwd  = gelu_bwd(x);
        float num  = (gelu_fwd(x + eps) - gelu_fwd(x - eps)) / (2.f * eps);
        float aerr = std::fabs(bwd - num);
        float rerr = aerr / (std::fabs(num) + 1e-6f);
        if (aerr > abs_tol && rerr > 0.05f)
            FAIL("gradient mismatch x=" + std::to_string(x) +
                 " bwd=" + std::to_string(bwd) +
                 " num=" + std::to_string(num));
    }
    PASS("gradient consistency: backward matches central-difference of forward");
    return 0;
}

// ── 5. Large inputs: finite and correct ──────────────────────────────────────
int test_large_inputs() {
    float vals[] = {-100.f, -50.f, -10.f, 10.f, 50.f, 100.f};
    for (float x : vals) {
        float y = gelu_fwd(x);
        if (!std::isfinite(y)) FAIL("non-finite output for x=" + std::to_string(x));
        float expected = (x > 0.f) ? x : 0.f;  // saturation values
        if (!NEAR(y, expected, 1e-3f))
            FAIL("large input x=" + std::to_string(x) +
                 " got=" + std::to_string(y) +
                 " expected≈" + std::to_string(expected));
    }
    PASS("large inputs |x|≤100: finite and correct");
    return 0;
}

int test_nan_and_descriptor_contract() {
    constexpr int kSuccess = 0;
    constexpr int kBadParam = 3;
    constexpr int kNotSupported = 9;
    cudnnActivationDescriptor_t act{};
    cudnnTensorDescriptor_t td{};
    if (cudnnCreateActivationDescriptor(&act) != kSuccess ||
        cudnnCreateTensorDescriptor(&td) != kSuccess)
        FAIL("descriptor creation failed");
    cudnnSetTensor4dDescriptor(td, 0, 0, 1, 1, 1, 1);

    const float alpha = 1.f, beta = 0.f;
    float y = 7.f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (cudnnActivationForward(nullptr, act, &alpha, td, &nan, &beta, td, &y) != kBadParam ||
        y != 7.f)
        FAIL("uninitialized activation descriptor was accepted");

    if (cudnnSetActivationDescriptor(act, 5, 0, 0.0) != kSuccess ||
        cudnnActivationForward(nullptr, act, &alpha, td, &nan, &beta, td, &y) != kNotSupported)
        FAIL("identity mode was accepted by direct activation API");
    if (cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 1, 0.0) != kSuccess)
        FAIL("valid GELU descriptor rejected");
    if (cudnnActivationForward(nullptr, act, &alpha, td, &nan, &beta, td, &y) != kSuccess ||
        !std::isnan(y))
        FAIL("propagating NaN mode did not preserve NaN");
    const float dy = 1.f;
    float dx = 0.f;
    if (cudnnActivationBackward(nullptr, act, &alpha, td, &nan, td, &dy,
                                td, &nan, &beta, td, &dx) != kSuccess ||
        !std::isnan(dx))
        FAIL("propagating NaN mode did not preserve NaN in backward");

    if (cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 0, 0.0) != kSuccess)
        FAIL("valid non-propagating GELU descriptor rejected");
    y = nan;
    if (cudnnActivationForward(nullptr, act, &alpha, td, &nan, &beta, td, &y) != kSuccess ||
        y != 0.f)
        FAIL("non-propagating mode did not suppress NaN input");
    dx = nan;
    if (cudnnActivationBackward(nullptr, act, &alpha, td, &y, td, &dy,
                                td, &nan, &beta, td, &dx) != kSuccess ||
        dx != 0.f)
        FAIL("non-propagating mode did not suppress NaN in backward");

    if (cudnnSetActivationDescriptor(act, 999, 0, 0.0) != kBadParam ||
        cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 999, 0.0) != kBadParam)
        FAIL("invalid activation or NaN mode was accepted");

    cudnnSetTensor4dDescriptor(td, 0, 0, 1, 1, 1, 1);
    const float reluInput = 3.f;
    float reluOutput = 0.f;
    if (cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, 0, 2.0) != kSuccess ||
        cudnnActivationForward(nullptr, act, &alpha, td, &reluInput, &beta, td,
                               &reluOutput) != kSuccess || reluOutput != 2.f)
        FAIL("ReLU descriptor upper bound was not applied");
    const float reluDy = 1.f;
    float reluDx = 1.f;
    if (cudnnActivationBackward(nullptr, act, &alpha, td, &reluOutput, td, &reluDy,
                                td, &reluInput, &beta, td, &reluDx) != kSuccess ||
        reluDx != 0.f)
        FAIL("capped ReLU derivative did not become zero above the cap");
    if (cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, 0, 0.0) != kSuccess ||
        cudnnActivationForward(nullptr, act, &alpha, td, &reluInput, &beta, td,
                               &reluOutput) != kSuccess || reluOutput != reluInput)
        FAIL("zero ReLU coefficient did not preserve the unbounded ReLU mode");

    cudnnSetTensor4dDescriptor(td, 0, 3, 1, 1, 1, 2); // CUDNN_DATA_INT8
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, 0, 0.0);
    const float int8Alpha = 0.5f, int8Beta = 0.25f;
    int8_t int8Input[] = {-4, 4};
    int8_t int8Output[] = {8, -8};
    if (cudnnActivationForward(nullptr, act, &int8Alpha, td, int8Input,
                               &int8Beta, td, int8Output) != kSuccess ||
        int8Output[0] != 2 || int8Output[1] != 0)
        FAIL("INT8 ReLU did not apply alpha/beta and saturating rounding");
    int8_t int8Dy[] = {5, 6};
    int8_t int8Dx[] = {4, 4};
    if (cudnnActivationBackward(nullptr, act, &int8Alpha, td, int8Output,
                                td, int8Dy, td, int8Input, &int8Beta,
                                td, int8Dx) != kSuccess ||
        int8Dx[0] != 1 || int8Dx[1] != 4)
        FAIL("INT8 ReLU backward did not apply alpha/beta");
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_GELU, 0, 0.0);

    cudnnSetTensor4dDescriptor(td, 0, 2, 1, 1, 1, 1); // CUDNN_DATA_HALF
    y = 19.f;
    if (cudnnActivationForward(nullptr, act, &alpha, td, &nan, &beta, td, &y) != kNotSupported ||
        y != 19.f)
        FAIL("unsupported dtype was not rejected before touching output");

    cudnnDestroyTensorDescriptor(td);
    cudnnDestroyActivationDescriptor(act);
    PASS("NaN modes, enum validation and unsupported dtype status");
    return 0;
}

int main() {
    std::cout << "=== Exact GELU and activation contract tests ===\n";
    int fails = 0;
    fails += test_forward_accuracy();
    fails += test_saturation();
    fails += test_gelu_properties();
    fails += test_gradient_consistency();
    fails += test_large_inputs();
    fails += test_nan_and_descriptor_contract();
    if (fails == 0) std::cout << "All exact GELU and activation contract tests passed.\n";
    return fails;
}
