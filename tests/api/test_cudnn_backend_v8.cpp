/**
 * Test: cuDNN Backend v8 Graph Execution
 *
 * Verifies that all backend op types can be built into a descriptor graph
 * and executed via cudnnBackendExecute without silent crashes or errors.
 * Covers: conv-forward, activation, batch-norm, pooling, matmul.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>

// ── cuDNN status / handle stubs ───────────────────────────────────────────────
extern "C" {

typedef int            cudnnStatus_t;
typedef void*          cudnnHandle_t;
typedef void*          cudnnTensorDescriptor_t;
typedef void*          cudnnFilterDescriptor_t;
typedef void*          cudnnConvolutionDescriptor_t;
typedef void*          cudnnPoolingDescriptor_t;
typedef void*          cudnnActivationDescriptor_t;

#define CUDNN_STATUS_SUCCESS       0
#define CUDNN_STATUS_INVALID_VALUE 8
#define CUDNN_STATUS_NOT_SUPPORTED 9

enum cudnnDataType_t      { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t  { CUDNN_TENSOR_NCHW = 0 };
enum cudnnNanPropagation_t{ CUDNN_NOT_PROPAGATE_NAN = 0 };
enum cudnnActivationMode_t{ CUDNN_ACTIVATION_RELU = 1 };
enum cudnnPoolingMode_t   { CUDNN_POOLING_MAX = 0 };
enum cudnnConvolutionFwdAlgo_t { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 };

// Legacy descriptor lifecycle (used to build ActDesc / PoolDesc pointers)
cudnnStatus_t cudnnCreate(cudnnHandle_t*);
cudnnStatus_t cudnnDestroy(cudnnHandle_t);

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t*);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t,
    cudnnActivationMode_t, cudnnNanPropagation_t, double);

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t*);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t,
    cudnnPoolingMode_t, int, int, int, int, int, int, int);

// ── Backend v8 API ────────────────────────────────────────────────────────────

enum cudnnBackendDescriptorType_t {
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 1,
    CUDNN_BACKEND_OPERATION_ACTIVATION_FORWARD_DESCRIPTOR  = 6,
    CUDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR     = 4,
    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR              = 8,
    CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = 11,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR      = 20,
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = 21,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR   = 24,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR      = 25,
    CUDNN_BACKEND_MATMUL_DESCRIPTOR      = 26,
    CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR= 27,
    CUDNN_BACKEND_HANDLE_DESCRIPTOR      = 28,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR= 33
};

enum cudnnBackendAttributeName_t {
    CUDNN_ATTR_TENSOR_DIMENSIONS                    = 30,
    CUDNN_ATTR_TENSOR_DATA_TYPE                     = 32,
    CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS             = 13,
    CUDNN_ATTR_CONVOLUTION_STRIDES                  = 15,
    CUDNN_ATTR_CONVOLUTION_DILATIONS                = 11,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X      = 40,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W      = 41,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y      = 42,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 43,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA  = 160,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA   = 161,
    CUDNN_ATTR_OPERATION_ACTIVATION_XDESC           = 80,
    CUDNN_ATTR_OPERATION_ACTIVATION_YDESC           = 81,
    CUDNN_ATTR_OPERATION_ACTIVATION_DESC            = 82,
    CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_ALPHA   = 162,
    CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_BETA    = 163,
    CUDNN_ATTR_OPERATIONPOOLING_XDESC               = 70,
    CUDNN_ATTR_OPERATIONPOOLING_YDESC               = 71,
    CUDNN_ATTR_OPERATIONPOOLING_PDESC               = 73,
    CUDNN_ATTR_OPERATION_POOLING_FORWARD_ALPHA      = 164,
    CUDNN_ATTR_OPERATION_POOLING_FORWARD_BETA       = 165,
    CUDNN_ATTR_OPERATION_MATMUL_ADESC               = 60,
    CUDNN_ATTR_OPERATION_MATMUL_BDESC               = 61,
    CUDNN_ATTR_OPERATION_MATMUL_CDESC               = 62,
    CUDNN_ATTR_OPERATION_MATMUL_ALPHA               = 179,
    CUDNN_ATTR_OPERATION_MATMUL_BETA                = 180,
    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS              = 100,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS           = 101,
    CUDNN_ATTR_ENGINECFG_ENGINE                     = 120,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG         = 131,
    CUDNN_ATTR_ENGINE_OPERATION_GRAPH               = 500,
    CUDNN_ATTR_OPERATIONSET_OPS                     = 501
};

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t, void**);
cudnnStatus_t cudnnBackendDestroyDescriptor(void*);
cudnnStatus_t cudnnBackendSetAttribute(void*, cudnnBackendAttributeName_t,
    int, int64_t, const void*);
cudnnStatus_t cudnnBackendFinalize(void*);
cudnnStatus_t cudnnBackendExecute(cudnnHandle_t, void* plan, void* variantPack);

} // extern "C"

// ── attribute type codes ──────────────────────────────────────────────────────
// 0=char, 1=float, 2=double, 3=int32, 4=int64, 5=uint64, 6=descriptor handle
static constexpr int kAttrFloat   = 1;
static constexpr int kAttrInt64   = 4;
static constexpr int kAttrUint64  = 5;
static constexpr int kAttrHandle  = 6;

// ── Helpers ───────────────────────────────────────────────────────────────────

static bool setDims(void* desc, cudnnBackendAttributeName_t attr,
                    std::initializer_list<int64_t> dims)
{
    std::vector<int64_t> v(dims);
    return cudnnBackendSetAttribute(desc, attr, kAttrInt64,
                                    (int64_t)v.size(), v.data())
           == CUDNN_STATUS_SUCCESS;
}

static bool setUint64(void* desc, cudnnBackendAttributeName_t attr, uint64_t val)
{
    return cudnnBackendSetAttribute(desc, attr, kAttrUint64, 1, &val)
           == CUDNN_STATUS_SUCCESS;
}

static bool setFloat(void* desc, cudnnBackendAttributeName_t attr, float val)
{
    return cudnnBackendSetAttribute(desc, attr, kAttrFloat, 1, &val)
           == CUDNN_STATUS_SUCCESS;
}

static bool setHandle(void* desc, cudnnBackendAttributeName_t attr, void* h)
{
    uint64_t id = reinterpret_cast<uintptr_t>(h);
    return cudnnBackendSetAttribute(desc, attr, kAttrHandle, 1, &id)
           == CUDNN_STATUS_SUCCESS;
}

// Build engine → config → plan chain pointing at opSet
static void* buildPlan(cudnnHandle_t h, void* opSet)
{
    (void)h;
    void* engine  = nullptr;
    void* config  = nullptr;
    void* plan    = nullptr;
    uint64_t opSetId = reinterpret_cast<uintptr_t>(opSet);

    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine)
            != CUDNN_STATUS_SUCCESS) return nullptr;
    cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                             kAttrHandle, 1, &opSetId);
    cudnnBackendFinalize(engine);

    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &config)
            != CUDNN_STATUS_SUCCESS) return nullptr;
    uint64_t engineId = reinterpret_cast<uintptr_t>(engine);
    cudnnBackendSetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE,
                             kAttrHandle, 1, &engineId);
    cudnnBackendFinalize(config);

    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_HANDLE_DESCRIPTOR, &plan)
            != CUDNN_STATUS_SUCCESS) return nullptr;
    uint64_t configId = reinterpret_cast<uintptr_t>(config);
    cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                             kAttrHandle, 1, &configId);
    cudnnBackendFinalize(plan);
    return plan;
}

// Build variant pack mapping tensor IDs to host buffers
static void* buildVariantPack(const std::vector<void*>& descs,
                               const std::vector<void*>& ptrs)
{
    void* vp = nullptr;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp)
            != CUDNN_STATUS_SUCCESS) return nullptr;

    std::vector<uint64_t> ids;
    std::vector<uint64_t> dps;
    ids.reserve(descs.size());
    dps.reserve(ptrs.size());
    for (auto* d : descs) ids.push_back(reinterpret_cast<uintptr_t>(d));
    for (auto* p : ptrs)  dps.push_back(reinterpret_cast<uintptr_t>(p));

    cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                             kAttrUint64, (int64_t)ids.size(), ids.data());
    cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             kAttrUint64, (int64_t)dps.size(), dps.data());
    cudnnBackendFinalize(vp);
    return vp;
}

// ── Individual sub-tests ──────────────────────────────────────────────────────

static bool testConvForward(cudnnHandle_t h, int& pass, int& total)
{
    // Conv: 1×1×4×4 input, 1×1×2×2 filter → 1×1×3×3 output (no padding, stride 1)
    static const int N=1, C=1, H=4, W=4;
    static const int K=1, R=2, S=2, OH=3, OW=3;

    std::vector<float> x(N*C*H*W, 1.0f);
    std::vector<float> w(K*C*R*S, 1.0f);  // all-ones filter
    std::vector<float> y(N*K*OH*OW, 0.0f);

    // Tensor descriptors
    void *xDesc=nullptr, *wDesc=nullptr, *yDesc=nullptr, *convDesc=nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc);
    setDims(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, C, H, W});
    { uint64_t dt = CUDNN_DATA_FLOAT;
      cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, kAttrInt64, 1, &dt); }
    cudnnBackendFinalize(xDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &wDesc);
    setDims(wDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {K, C, R, S});
    { uint64_t dt = CUDNN_DATA_FLOAT;
      cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, kAttrInt64, 1, &dt); }
    cudnnBackendFinalize(wDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc);
    setDims(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, K, OH, OW});
    { uint64_t dt = CUDNN_DATA_FLOAT;
      cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, kAttrInt64, 1, &dt); }
    cudnnBackendFinalize(yDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc);
    setDims(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, {0, 0});
    setDims(convDesc, CUDNN_ATTR_CONVOLUTION_STRIDES,      {1, 1});
    setDims(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS,    {1, 1});
    cudnnBackendFinalize(convDesc);

    // Operation
    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &op);
    setHandle(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, xDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, yDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, convDesc);
    setFloat(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0f);
    setFloat(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,  0.0f);
    cudnnBackendFinalize(op);

    // OpSet
    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    { uint64_t opId = reinterpret_cast<uintptr_t>(op);
      cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                               kAttrHandle, 1, &opId); }
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp   = buildVariantPack({xDesc, wDesc, yDesc},
                                  {x.data(), w.data(), y.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    if (st == CUDNN_STATUS_SUCCESS) {
        // Each output element = sum of 2×2 all-ones patch × all-ones filter = 4.0
        bool ok = true;
        for (float v : y) if (std::abs(v - 4.0f) > 1e-3f) { ok = false; break; }
        if (ok) { ++pass; std::cout << "  PASS ConvForward\n"; }
        else    { std::cout << "  FAIL ConvForward: unexpected output " << y[0] << "\n"; }
        return ok;
    } else {
        std::cout << "  FAIL ConvForward: execute returned " << st << "\n";
        return false;
    }
}

static bool testActivationForward(cudnnHandle_t h, int& pass, int& total)
{
    // ReLU on a 1×2×2×2 tensor with values [-1, 0, 0.5, 1, -2, 0, 3, -0.5]
    std::vector<float> x = {-1.f, 0.f, 0.5f, 1.f, -2.f, 0.f, 3.f, -0.5f};
    std::vector<float> y(x.size(), -99.f);

    cudnnActivationDescriptor_t actD = nullptr;
    cudnnCreateActivationDescriptor(&actD);
    cudnnSetActivationDescriptor(actD, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    void *xDesc=nullptr, *yDesc=nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc);
    setDims(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {1, 2, 2, 2});
    cudnnBackendFinalize(xDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc);
    setDims(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {1, 2, 2, 2});
    cudnnBackendFinalize(yDesc);

    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_ACTIVATION_FORWARD_DESCRIPTOR, &op);
    setHandle(op, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC, xDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC, yDesc);
    // Store actual ActDesc* pointer as the attribute value
    { uint64_t aId = reinterpret_cast<uintptr_t>(actD);
      cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_ACTIVATION_DESC,
                               kAttrUint64, 1, &aId); }
    setFloat(op, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_ALPHA, 1.0f);
    setFloat(op, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_BETA,  0.0f);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    { uint64_t opId = reinterpret_cast<uintptr_t>(op);
      cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                               kAttrHandle, 1, &opId); }
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp   = buildVariantPack({xDesc, yDesc}, {x.data(), y.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    bool ok = false;
    if (st == CUDNN_STATUS_SUCCESS) {
        static const float expected[] = {0.f, 0.f, 0.5f, 1.f, 0.f, 0.f, 3.f, 0.f};
        ok = true;
        for (size_t i = 0; i < y.size(); ++i)
            if (std::abs(y[i] - expected[i]) > 1e-3f) { ok = false; break; }
        if (ok) { ++pass; std::cout << "  PASS ActivationForward (ReLU)\n"; }
        else    { std::cout << "  FAIL ActivationForward: y[0]=" << y[0] << " y[2]=" << y[2] << "\n"; }
    } else {
        std::cout << "  FAIL ActivationForward: execute returned " << st << "\n";
    }
    cudnnDestroyActivationDescriptor(actD);
    return ok;
}

static bool testBNFinalize(cudnnHandle_t h, int& pass, int& total)
{
    // BN finalize on a 1×4×2×2 tensor (ones) — verify no crash and success
    int N=1, C=4, H=2, W=2;
    int sz = N*C*H*W;
    std::vector<float> x(sz, 1.0f);
    std::vector<float> y(sz, 0.0f);

    void *xDesc=nullptr, *yDesc=nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc);
    setDims(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, C, H, W});
    cudnnBackendFinalize(xDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc);
    setDims(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, C, H, W});
    cudnnBackendFinalize(yDesc);

    void* op = nullptr;
    cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR, &op);
    // BN finalize reuses xdesc/ydesc attribute slots
    setHandle(op, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC, xDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC, yDesc);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    { uint64_t opId = reinterpret_cast<uintptr_t>(op);
      cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                               kAttrHandle, 1, &opId); }
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp   = buildVariantPack({xDesc, yDesc}, {x.data(), y.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    if (st == CUDNN_STATUS_SUCCESS) {
        ++pass;
        std::cout << "  PASS BNFinalize\n";
        return true;
    }
    std::cout << "  FAIL BNFinalize: execute returned " << st << "\n";
    return false;
}

static bool testPoolingForward(cudnnHandle_t h, int& pass, int& total)
{
    // Max pooling: 1×1×4×4 → 1×1×2×2 with 2×2 window, stride 2
    static const int N=1, C=1, IH=4, IW=4, OH=2, OW=2;
    // clang-format off
    std::vector<float> x = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    // clang-format on
    std::vector<float> y(N*C*OH*OW, 0.0f);

    cudnnPoolingDescriptor_t poolD = nullptr;
    cudnnCreatePoolingDescriptor(&poolD);
    // mode, nanOpt, windowH, windowW, padH, padW, strideH, strideW
    cudnnSetPooling2dDescriptor(poolD, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                2, 2, 0, 0, 2, 2);

    void *xDesc=nullptr, *yDesc=nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc);
    setDims(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, C, IH, IW});
    cudnnBackendFinalize(xDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc);
    setDims(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {N, C, OH, OW});
    cudnnBackendFinalize(yDesc);

    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR, &op);
    setHandle(op, CUDNN_ATTR_OPERATIONPOOLING_XDESC, xDesc);
    setHandle(op, CUDNN_ATTR_OPERATIONPOOLING_YDESC, yDesc);
    { uint64_t pId = reinterpret_cast<uintptr_t>(poolD);
      cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATIONPOOLING_PDESC,
                               kAttrUint64, 1, &pId); }
    setFloat(op, CUDNN_ATTR_OPERATION_POOLING_FORWARD_ALPHA, 1.0f);
    setFloat(op, CUDNN_ATTR_OPERATION_POOLING_FORWARD_BETA,  0.0f);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    { uint64_t opId = reinterpret_cast<uintptr_t>(op);
      cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                               kAttrHandle, 1, &opId); }
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp   = buildVariantPack({xDesc, yDesc}, {x.data(), y.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    bool ok = false;
    if (st == CUDNN_STATUS_SUCCESS) {
        // Expected max-pool 2×2/stride-2: top-left=6, top-right=8, bot-left=14, bot-right=16
        float expected[] = {6.f, 8.f, 14.f, 16.f};
        ok = true;
        for (int i = 0; i < 4; ++i)
            if (std::abs(y[i] - expected[i]) > 1e-3f) { ok = false; break; }
        if (ok) { ++pass; std::cout << "  PASS PoolingForward (MaxPool 2×2)\n"; }
        else    { std::cout << "  FAIL PoolingForward: got " << y[0] << "," << y[1]
                             << "," << y[2] << "," << y[3] << "\n"; }
    } else {
        std::cout << "  FAIL PoolingForward: execute returned " << st << "\n";
    }
    cudnnDestroyPoolingDescriptor(poolD);
    return ok;
}

static bool testMatmul(cudnnHandle_t h, int& pass, int& total)
{
    // C = A * B, all 2×2 identity matrices → result should be identity
    // Tensor shape stored as (n,c,1,1) where n=rows, c=cols for the matmul path
    static const int M = 2, NK = 2;
    // clang-format off
    std::vector<float> A = {1, 0,
                             0, 1};
    std::vector<float> B = {2, 3,
                             4, 5};
    std::vector<float> C(M * NK, 0.0f);
    // clang-format on

    void *aDesc=nullptr, *bDesc=nullptr, *cDesc=nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &aDesc);
    setDims(aDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {M, NK, 1, 1});
    cudnnBackendFinalize(aDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &bDesc);
    setDims(bDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {NK, NK, 1, 1});
    cudnnBackendFinalize(bDesc);

    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &cDesc);
    setDims(cDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, {M, NK, 1, 1});
    cudnnBackendFinalize(cDesc);

    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &op);
    setHandle(op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, aDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, bDesc);
    setHandle(op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, cDesc);
    setFloat(op, CUDNN_ATTR_OPERATION_MATMUL_ALPHA, 1.0f);
    setFloat(op, CUDNN_ATTR_OPERATION_MATMUL_BETA,  0.0f);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    { uint64_t opId = reinterpret_cast<uintptr_t>(op);
      cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                               kAttrHandle, 1, &opId); }
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp   = buildVariantPack({aDesc, bDesc, cDesc},
                                  {A.data(), B.data(), C.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    if (st == CUDNN_STATUS_SUCCESS) {
        // I * B = B, so C should equal B
        bool ok = true;
        for (int i = 0; i < M * NK; ++i)
            if (std::abs(C[i] - B[i]) > 1e-3f) { ok = false; break; }
        if (ok) { ++pass; std::cout << "  PASS Matmul (I*B=B)\n"; }
        else    { std::cout << "  FAIL Matmul: C[0]=" << C[0] << " C[1]=" << C[1] << "\n"; }
        return ok;
    }
    std::cout << "  FAIL Matmul: execute returned " << st << "\n";
    return false;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== Test: cuDNN Backend v8 Graph Execution ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t h = nullptr;
    if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    testConvForward(h, pass, total);
    testActivationForward(h, pass, total);
    testBNFinalize(h, pass, total);
    testPoolingForward(h, pass, total);
    testMatmul(h, pass, total);

    cudnnDestroy(h);

    std::cout << "\nResult: " << pass << "/" << total << " passed\n";
    return (pass == total) ? 0 : 1;
}
