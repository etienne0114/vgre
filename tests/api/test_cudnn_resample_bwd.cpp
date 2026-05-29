/**
 * Test: cuDNN Backend v8 Resample FWD + BWD
 *
 * Verifies that bilinear resample forward (upsample) and backward (gradient)
 * produce correct results through the backend descriptor graph API.
 *
 * FWD: 1×1×2×2 → 1×1×4×4  bilinear upsample
 * BWD: upstream gradient 1×1×4×4 → input gradient 1×1×2×2
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>
#include <numeric>

// ── cuDNN backend API declarations ────────────────────────────────────────────
extern "C" {

typedef int   cudnnStatus_t;
typedef void* cudnnHandle_t;

#define CUDNN_STATUS_SUCCESS       0
#define CUDNN_STATUS_INVALID_VALUE 8

enum cudnnBackendDescriptorType_t {
    CUDNN_BACKEND_TENSOR_DESCRIPTOR       = 20,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR    = 24,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR       = 25,
    CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR = 27,
    CUDNN_BACKEND_HANDLE_DESCRIPTOR       = 28,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 33,
    CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = 38,
    CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = 39
};

enum cudnnBackendAttributeName_t {
    CUDNN_ATTR_TENSOR_DIMENSIONS             = 30,
    CUDNN_ATTR_TENSOR_DATA_TYPE              = 32,
    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS       = 100,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS    = 101,
    CUDNN_ATTR_ENGINECFG_ENGINE              = 120,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG  = 131,
    CUDNN_ATTR_ENGINE_OPERATION_GRAPH        = 500,
    CUDNN_ATTR_OPERATIONSET_OPS              = 501,
    CUDNN_ATTR_OPERATION_RESAMPLE_XDESC      = 620,
    CUDNN_ATTR_OPERATION_RESAMPLE_YDESC      = 621
};

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t*);
cudnnStatus_t cudnnDestroy(cudnnHandle_t);
cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t, void**);
cudnnStatus_t cudnnBackendDestroyDescriptor(void*);
cudnnStatus_t cudnnBackendSetAttribute(void*, cudnnBackendAttributeName_t,
    int, int64_t, const void*);
cudnnStatus_t cudnnBackendFinalize(void*);
cudnnStatus_t cudnnBackendExecute(cudnnHandle_t, void* plan, void* variantPack);

} // extern "C"

// Attribute type codes
static constexpr int kAttrInt64  = 4;
static constexpr int kAttrUint64 = 5;
static constexpr int kAttrHandle = 6;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void* makeTensor(std::initializer_list<int64_t> dims) {
    void* desc = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &desc);
    std::vector<int64_t> v(dims);
    cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                             kAttrInt64, (int64_t)v.size(), v.data());
    uint64_t dt = CUDNN_DATA_FLOAT;
    cudnnBackendSetAttribute(desc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                             kAttrInt64, 1, &dt);
    cudnnBackendFinalize(desc);
    return desc;
}

static void* buildPlan(cudnnHandle_t h, void* opSet) {
    (void)h;
    uint64_t opSetId = reinterpret_cast<uintptr_t>(opSet);

    void* engine = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
    cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                             kAttrHandle, 1, &opSetId);
    cudnnBackendFinalize(engine);

    void* config = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &config);
    uint64_t engineId = reinterpret_cast<uintptr_t>(engine);
    cudnnBackendSetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE,
                             kAttrHandle, 1, &engineId);
    cudnnBackendFinalize(config);

    void* plan = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_HANDLE_DESCRIPTOR, &plan);
    uint64_t configId = reinterpret_cast<uintptr_t>(config);
    cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                             kAttrHandle, 1, &configId);
    cudnnBackendFinalize(plan);
    return plan;
}

static void* buildVariantPack(const std::vector<void*>& descs,
                               const std::vector<void*>& ptrs) {
    void* vp = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp);
    std::vector<uint64_t> ids, dps;
    for (auto* d : descs) ids.push_back(reinterpret_cast<uintptr_t>(d));
    for (auto* p : ptrs)  dps.push_back(reinterpret_cast<uintptr_t>(p));
    cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                             kAttrUint64, (int64_t)ids.size(), ids.data());
    cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                             kAttrUint64, (int64_t)dps.size(), dps.data());
    cudnnBackendFinalize(vp);
    return vp;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static bool testResampleFwd(cudnnHandle_t h, int& pass, int& total) {
    // Bilinear upsample: 1×1×2×2 → 1×1×4×4
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> y(16, 0.f);

    void* xDesc = makeTensor({1, 1, 2, 2});
    void* yDesc = makeTensor({1, 1, 4, 4});

    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR, &op);
    uint64_t xId = reinterpret_cast<uintptr_t>(xDesc);
    uint64_t yId = reinterpret_cast<uintptr_t>(yDesc);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC,
                             kAttrUint64, 1, &xId);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC,
                             kAttrUint64, 1, &yId);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    uint64_t opId = reinterpret_cast<uintptr_t>(op);
    cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                             kAttrHandle, 1, &opId);
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp = buildVariantPack({xDesc, yDesc}, {x.data(), y.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    if (st != CUDNN_STATUS_SUCCESS) {
        std::cout << "  FAIL ResampleFWD: execute returned " << st << "\n";
        return false;
    }

    // Verify output is non-zero and all values are in the range of input values
    bool ok = true;
    for (float v : y) {
        if (v < 0.5f || v > 4.5f) { ok = false; break; }
    }
    if (ok) { ++pass; std::cout << "  PASS ResampleFWD (bilinear upsample 2×2→4×4)\n"; }
    else    { std::cout << "  FAIL ResampleFWD: output out of range, y[0]=" << y[0] << "\n"; }
    return ok;
}

static bool testResampleBwd(cudnnHandle_t h, int& pass, int& total) {
    // BWD: gradient of upsample 2×2→4×4
    // dY is 4×4 all-ones upstream gradient, dX should be 2×2 gradient
    std::vector<float> dY(16, 1.f);  // upstream gradient
    std::vector<float> dX(4, 0.f);   // input gradient (to be computed)

    // xDesc = dX dimensions (2×2), yDesc = dY dimensions (4×4)
    void* xDesc = makeTensor({1, 1, 2, 2});
    void* yDesc = makeTensor({1, 1, 4, 4});

    void* op = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR, &op);
    uint64_t xId = reinterpret_cast<uintptr_t>(xDesc);
    uint64_t yId = reinterpret_cast<uintptr_t>(yDesc);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC,
                             kAttrUint64, 1, &xId);
    cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC,
                             kAttrUint64, 1, &yId);
    cudnnBackendFinalize(op);

    void* opSet = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR, &opSet);
    uint64_t opId = reinterpret_cast<uintptr_t>(op);
    cudnnBackendSetAttribute(opSet, CUDNN_ATTR_OPERATIONSET_OPS,
                             kAttrHandle, 1, &opId);
    cudnnBackendFinalize(opSet);

    void* plan = buildPlan(h, opSet);
    void* vp = buildVariantPack({xDesc, yDesc}, {dX.data(), dY.data()});

    ++total;
    cudnnStatus_t st = cudnnBackendExecute(h, plan, vp);
    if (st != CUDNN_STATUS_SUCCESS) {
        std::cout << "  FAIL ResampleBWD: execute returned " << st << "\n";
        return false;
    }

    // Energy conservation: sum(dX) should equal sum(dY) = 16.0
    // (bilinear scatter-add preserves total gradient mass)
    float sumDX = 0.f;
    for (float v : dX) sumDX += v;
    float sumDY = 16.f;

    bool ok = std::abs(sumDX - sumDY) < 1e-3f;

    // Also verify all dX values are positive (all-positive upstream → all-positive gradient)
    for (float v : dX) {
        if (v <= 0.f) { ok = false; break; }
    }

    if (ok) { ++pass; std::cout << "  PASS ResampleBWD (gradient energy conservation, sum=" << sumDX << ")\n"; }
    else    { std::cout << "  FAIL ResampleBWD: sumDX=" << sumDX << " expected " << sumDY
                        << " dX=[" << dX[0] << "," << dX[1] << "," << dX[2] << "," << dX[3] << "]\n"; }
    return ok;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Test: cuDNN Backend Resample FWD + BWD ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t h = nullptr;
    if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    testResampleFwd(h, pass, total);
    testResampleBwd(h, pass, total);

    cudnnDestroy(h);

    std::cout << "\nResult: " << pass << "/" << total << " passed\n";
    return (pass == total) ? 0 : 1;
}
