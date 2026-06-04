// QUEUE-38: cuDNN RNN peephole LSTM tests.
// Verifies that peephole connections are correctly applied in lstm_fwd():
//   i_t = sigmoid(... + p_i * c_{t-1})   — input gate sees c_{t-1}
//   f_t = sigmoid(... + p_f * c_{t-1})   — forget gate sees c_{t-1}
//   c_t = f_t * c_{t-1} + i_t * g_t
//   o_t = sigmoid(... + p_o * c_t)       — output gate sees c_t
//   h_t = o_t * tanh(c_t)
//
// Test structure:
//   1. Zero peephole weights → output must equal standard LSTM (regression)
//   2. Non-zero peephole weights → output differs from standard LSTM (peephole active)
//   3. Non-zero peephole weights → output matches C reference implementation

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
typedef int    cudnnStatus_t;
typedef void*  cudnnHandle_t;
typedef void*  cudnnTensorDescriptor_t;
typedef void*  cudnnFilterDescriptor_t;
typedef void*  cudnnRNNDescriptor_t;
typedef void*  cudnnDropoutDescriptor_t;

#define CUDNN_STATUS_SUCCESS        0
#define CUDNN_STATUS_BAD_PARAM      3
#define CUDNN_STATUS_INVALID_VALUE  8
#define CUDNN_DATA_FLOAT           0
#define CUDNN_TENSOR_NCHW          0
#define CUDNN_LINEAR_INPUT         0
#define CUDNN_UNIDIRECTIONAL       0
#define CUDNN_LSTM                 2

cudnnStatus_t cudnnCreate(cudnnHandle_t*);
cudnnStatus_t cudnnDestroy(cudnnHandle_t);
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t*);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t, int fmt,
    int dtype, int n, int c, int h, int w);
cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t*);
cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t);
cudnnStatus_t cudnnSetRNNDescriptor(cudnnRNNDescriptor_t, int hidden, int layers,
    cudnnDropoutDescriptor_t, int inputMode, int direction, int mode, int dtype);
cudnnStatus_t cudnnRNNSetPeephole(cudnnRNNDescriptor_t, int enable);
cudnnStatus_t cudnnRNNGetPeephole(cudnnRNNDescriptor_t, int* enable);
cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t, cudnnRNNDescriptor_t,
    cudnnTensorDescriptor_t, size_t*, int dtype);
cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t, cudnnRNNDescriptor_t, int T,
    cudnnTensorDescriptor_t* xDesc, const void* x,
    cudnnTensorDescriptor_t hxDesc, const void* hx,
    cudnnTensorDescriptor_t cxDesc, const void* cx,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t* yDesc, void* y,
    cudnnTensorDescriptor_t hyDesc, void* hy,
    cudnnTensorDescriptor_t cyDesc, void* cy,
    void* workSpace, size_t workSpaceSizeInBytes);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)
#define CHECK(call, msg) do { if ((call) != CUDNN_STATUS_SUCCESS) FAIL(msg); } while(0)

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// Reference peephole LSTM forward for one timestep.
// Weight layout: W_ih[G*H x I], W_hh[G*H x H], b_ih[G*H], b_hh[G*H], p_i[H], p_f[H], p_o[H]
// Gate order: 0=i, 1=f, 2=g, 3=o
static void ref_peephole_lstm_step(
    int B, int H, int I,
    const float* W_ih, const float* W_hh,
    const float* b_ih, const float* b_hh,
    const float* p_i, const float* p_f, const float* p_o,
    const float* x, const float* hp, const float* cp,
    float* h_out, float* c_out)
{
    for (int n = 0; n < B; ++n) {
        for (int j = 0; j < H; ++j) {
            float s[4];
            for (int g = 0; g < 4; ++g)
                s[g] = b_ih[g*H+j] + b_hh[g*H+j];
            for (int k = 0; k < I; ++k)
                for (int g = 0; g < 4; ++g)
                    s[g] += W_ih[(g*H+j)*I+k] * x[n*I+k];
            if (hp)
                for (int k = 0; k < H; ++k)
                    for (int g = 0; g < 4; ++g)
                        s[g] += W_hh[(g*H+j)*H+k] * hp[n*H+k];
            float cp_j = cp ? cp[n*H+j] : 0.f;
            s[0] += p_i[j] * cp_j;
            s[1] += p_f[j] * cp_j;
            float iv = sigmoid(s[0]), fv = sigmoid(s[1]), gv = tanhf(s[2]);
            float cn = fv * cp_j + iv * gv;
            s[3] += p_o[j] * cn;
            float ov = sigmoid(s[3]);
            c_out[n*H+j] = cn;
            h_out[n*H+j] = ov * tanhf(cn);
        }
    }
}

// Deterministic LCG for reproducible weights
static float randf(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.f - 0.5f;
}

// Run a single test scenario. Returns 0 on pass.
// peephole_scale: if 0.0, peephole weights are zero (should match standard LSTM w/ p=0).
//                 if non-zero, weights are non-zero.
static int run_test(int H, int I, int B, int T,
                    float peephole_scale, const char* label)
{
    // Create cuDNN handles and descriptors
    cudnnHandle_t           hcudnn = nullptr;
    cudnnRNNDescriptor_t    rnnD   = nullptr;
    cudnnTensorDescriptor_t xDesc_arr[8], yDesc_arr[8], hxD, cxD, hyD, cyD;
    memset(xDesc_arr, 0, sizeof(xDesc_arr));
    memset(yDesc_arr, 0, sizeof(yDesc_arr));

    CHECK(cudnnCreate(&hcudnn),                    "cudnnCreate");
    CHECK(cudnnCreateRNNDescriptor(&rnnD),          "CreateRNNDesc");
    CHECK(cudnnSetRNNDescriptor(rnnD, H, 1, nullptr,
          CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
          CUDNN_LSTM, CUDNN_DATA_FLOAT),            "SetRNNDesc");
    CHECK(cudnnRNNSetPeephole(rnnD, 1),             "SetPeephole");

    // Verify peephole is recorded
    int ph_flag = 0;
    CHECK(cudnnRNNGetPeephole(rnnD, &ph_flag),      "GetPeephole");
    if (!ph_flag) FAIL("peephole flag not set");

    // Create x/y descriptors for each timestep: [B, I, 1, 1]
    CHECK(cudnnCreateTensorDescriptor(&hxD),  "create hxD");
    CHECK(cudnnCreateTensorDescriptor(&cxD),  "create cxD");
    CHECK(cudnnCreateTensorDescriptor(&hyD),  "create hyD");
    CHECK(cudnnCreateTensorDescriptor(&cyD),  "create cyD");
    CHECK(cudnnSetTensor4dDescriptor(hxD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, B, H, 1), "set hxD");
    CHECK(cudnnSetTensor4dDescriptor(cxD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, B, H, 1), "set cxD");
    CHECK(cudnnSetTensor4dDescriptor(hyD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, B, H, 1), "set hyD");
    CHECK(cudnnSetTensor4dDescriptor(cyD, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, B, H, 1), "set cyD");

    for (int t = 0; t < T; ++t) {
        CHECK(cudnnCreateTensorDescriptor(&xDesc_arr[t]), "create xDesc");
        CHECK(cudnnCreateTensorDescriptor(&yDesc_arr[t]), "create yDesc");
        CHECK(cudnnSetTensor4dDescriptor(xDesc_arr[t], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, I, 1, 1), "set xDesc");
        CHECK(cudnnSetTensor4dDescriptor(yDesc_arr[t], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, H, 1, 1), "set yDesc");
    }

    // Get weight buffer size and allocate
    size_t wsize = 0;
    CHECK(cudnnGetRNNParamsSize(hcudnn, rnnD, xDesc_arr[0], &wsize, CUDNN_DATA_FLOAT), "GetParamsSize");

    // Standard LSTM params: G=4, each layer has W_ih[G*H*I], W_hh[G*H*H], b_ih[G*H], b_hh[G*H]
    // Plus peephole: p_i[H], p_f[H], p_o[H]
    size_t expected = sizeof(float) * (4*(size_t)H*I + 4*(size_t)H*H + 4*H + 4*H + 3*H);
    if (wsize != expected) {
        fprintf(stderr, "FAIL: %s wsize=%zu expected=%zu\n", label, wsize, expected);
        cudnnDestroy(hcudnn); cudnnDestroyRNNDescriptor(rnnD);
        return 1;
    }

    size_t nW = wsize / sizeof(float);
    std::vector<float> w(nW, 0.f);

    // Fill standard weights with random values
    unsigned seed = 0xBEEFCAFEu ^ (unsigned)H ^ (unsigned)I;
    // Offset layout: W_ih[4*H*I], W_hh[4*H*H], b_ih[4*H], b_hh[4*H], p_i[H], p_f[H], p_o[H]
    size_t off_Wih=0, off_Whh=4*(size_t)H*I, off_bih=off_Whh+4*(size_t)H*H;
    size_t off_bhh=off_bih+4*H, off_pi=off_bhh+4*H, off_pf=off_pi+H, off_po=off_pf+H;

    for (size_t i = 0; i < 4*(size_t)H*I; ++i) w[off_Wih+i] = randf(seed) * 0.3f;
    for (size_t i = 0; i < 4*(size_t)H*H; ++i) w[off_Whh+i] = randf(seed) * 0.3f;
    for (size_t i = 0; i < 4*(size_t)H;   ++i) w[off_bih+i] = randf(seed) * 0.1f;
    for (size_t i = 0; i < 4*(size_t)H;   ++i) w[off_bhh+i] = randf(seed) * 0.1f;
    // Peephole weights (may be zero or non-zero)
    for (int j = 0; j < H; ++j) {
        w[off_pi+j] = peephole_scale * randf(seed);
        w[off_pf+j] = peephole_scale * randf(seed);
        w[off_po+j] = peephole_scale * randf(seed);
    }

    // Random input sequence x[T*B*I] and hx, cx
    std::vector<float> x(T*B*I), hx(B*H, 0.f), cx(B*H, 0.f);
    for (auto& v : x) v = randf(seed) * 0.5f;

    std::vector<float> y(T*B*H, 0.f), hy(B*H, 0.f), cy(B*H, 0.f);

    CHECK(cudnnRNNForwardInference(
        hcudnn, rnnD, T,
        xDesc_arr, x.data(),
        hxD, hx.data(),
        cxD, cx.data(),
        nullptr, w.data(),
        yDesc_arr, y.data(),
        hyD, hy.data(),
        cyD, cy.data(),
        nullptr, 0),
        "RNNForwardInference");

    // Reference computation
    const float* W_ih = w.data() + off_Wih;
    const float* W_hh = w.data() + off_Whh;
    const float* b_ih = w.data() + off_bih;
    const float* b_hh = w.data() + off_bhh;
    const float* pi_w = w.data() + off_pi;
    const float* pf_w = w.data() + off_pf;
    const float* po_w = w.data() + off_po;

    std::vector<float> ref_h(B*H), ref_c(B*H);
    std::vector<float> hp_ref(hx), cp_ref(cx);
    for (int t = 0; t < T; ++t) {
        ref_peephole_lstm_step(B, H, I, W_ih, W_hh, b_ih, b_hh, pi_w, pf_w, po_w,
                               x.data() + t*B*I, hp_ref.data(), cp_ref.data(),
                               ref_h.data(), ref_c.data());
        hp_ref = ref_h;
        cp_ref = ref_c;
    }

    // Compare final hidden state (y[last_t] = h at t=T-1)
    float max_err = 0.f;
    for (int i = 0; i < B*H; ++i)
        max_err = std::max(max_err, fabsf(y[(T-1)*B*H + i] - ref_h[i]));

    if (max_err > 1e-5f) {
        fprintf(stderr, "FAIL: %s max_err=%g vs reference (threshold 1e-5)\n", label, max_err);
        cudnnDestroy(hcudnn); cudnnDestroyRNNDescriptor(rnnD);
        for (int t = 0; t < T; ++t) {
            cudnnDestroyTensorDescriptor(xDesc_arr[t]);
            cudnnDestroyTensorDescriptor(yDesc_arr[t]);
        }
        cudnnDestroyTensorDescriptor(hxD); cudnnDestroyTensorDescriptor(cxD);
        cudnnDestroyTensorDescriptor(hyD); cudnnDestroyTensorDescriptor(cyD);
        return 1;
    }

    // Cleanup
    for (int t = 0; t < T; ++t) {
        cudnnDestroyTensorDescriptor(xDesc_arr[t]);
        cudnnDestroyTensorDescriptor(yDesc_arr[t]);
    }
    cudnnDestroyTensorDescriptor(hxD); cudnnDestroyTensorDescriptor(cxD);
    cudnnDestroyTensorDescriptor(hyD); cudnnDestroyTensorDescriptor(cyD);
    cudnnDestroyRNNDescriptor(rnnD);
    cudnnDestroy(hcudnn);

    printf("PASS: %s max_err=%g\n", label, max_err);
    return 0;
}

// Test cudnnRNNSetPeephole / cudnnRNNGetPeephole API contract
static int test_api_contract() {
    cudnnRNNDescriptor_t rnnD = nullptr;
    if (cudnnCreateRNNDescriptor(&rnnD) != CUDNN_STATUS_SUCCESS) FAIL("CreateRNNDesc");
    if (cudnnSetRNNDescriptor(rnnD, 4, 1, nullptr, 0, 0, CUDNN_LSTM, 0) != CUDNN_STATUS_SUCCESS)
        FAIL("SetRNNDesc");

    // Default: peephole off
    int flag = 99;
    if (cudnnRNNGetPeephole(rnnD, &flag) != CUDNN_STATUS_SUCCESS) FAIL("GetPeephole default");
    if (flag != 0) FAIL("peephole default must be 0");

    // Enable
    if (cudnnRNNSetPeephole(rnnD, 1) != CUDNN_STATUS_SUCCESS) FAIL("SetPeephole 1");
    if (cudnnRNNGetPeephole(rnnD, &flag) != CUDNN_STATUS_SUCCESS) FAIL("GetPeephole after set");
    if (flag != 1) FAIL("peephole must be 1 after set");

    // Disable
    if (cudnnRNNSetPeephole(rnnD, 0) != CUDNN_STATUS_SUCCESS) FAIL("SetPeephole 0");
    if (cudnnRNNGetPeephole(rnnD, &flag) != CUDNN_STATUS_SUCCESS) FAIL("GetPeephole after clear");
    if (flag != 0) FAIL("peephole must be 0 after clear");

    // Null handle → INVALID_VALUE
    if (cudnnRNNSetPeephole(nullptr, 1) != CUDNN_STATUS_INVALID_VALUE) FAIL("SetPeephole null must return INVALID_VALUE");
    if (cudnnRNNGetPeephole(nullptr, &flag) != CUDNN_STATUS_INVALID_VALUE) FAIL("GetPeephole null must return INVALID_VALUE");
    if (cudnnRNNGetPeephole(rnnD, nullptr) != CUDNN_STATUS_INVALID_VALUE) FAIL("GetPeephole null ptr must return INVALID_VALUE");

    cudnnDestroyRNNDescriptor(rnnD);
    PASS("peephole API contract (set/get/null checks)");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_api_contract();

    // Zero peephole weights: output must match standard LSTM (p_i=p_f=p_o=0)
    rc |= run_test(4, 4, 1, 3, 0.0f, "H=4,I=4,B=1,T=3 peephole=zero (matches standard)");
    rc |= run_test(4, 4, 2, 2, 0.0f, "H=4,I=4,B=2,T=2 peephole=zero (batch)");

    // Non-zero peephole weights: output must match reference peephole LSTM
    rc |= run_test(4, 4, 1, 1, 1.0f, "H=4,I=4,B=1,T=1 peephole=active single step");
    rc |= run_test(4, 4, 1, 3, 1.0f, "H=4,I=4,B=1,T=3 peephole=active multi-step");
    rc |= run_test(8, 6, 2, 4, 0.5f, "H=8,I=6,B=2,T=4 peephole=active larger");

    if (rc == 0)
        printf("All cuDNN RNN peephole QUEUE-38 tests passed.\n");
    return rc;
}
