// cuDNN v8+ Backend API — minimal stub that routes descriptor graphs to
// existing legacy cuDNN shim paths (conv, pool, activation, etc.).
//
// For unimplemented operations, returns CUDNN_STATUS_NOT_SUPPORTED.

#include "cudnn_internal.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

// ── Backend descriptor type IDs ──────────────────────────────────────────────
enum cudnnBackendDescriptorType_t {
    CUDNN_BACKEND_OPERATION_DESCRIPTOR = 0,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 1,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 2,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = 3,
    CUDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR = 4,
    CUDNN_BACKEND_OPERATION_POOLING_BACKWARD_DESCRIPTOR = 5,
    CUDNN_BACKEND_OPERATION_ACTIVATION_FORWARD_DESCRIPTOR = 6,
    CUDNN_BACKEND_OPERATION_ACTIVATION_BACKWARD_DESCRIPTOR = 7,
    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 8,
    CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 9,
    CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR = 10,
    CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = 11,
    CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = 12,
    CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = 13,
    CUDNN_BACKEND_OPERATION_RESAMPLE_DESCRIPTOR = 14,
    CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = 15,
    CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = 16,
    CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = 17,
    CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = 18,
    CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = 19,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR = 20,
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = 21,
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 22,
    CUDNN_BACKEND_GENSTATS_DESCRIPTOR = 23,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 24,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR = 25,
    CUDNN_BACKEND_MATMUL_DESCRIPTOR = 26,
    CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR = 27,
    CUDNN_BACKEND_HANDLE_DESCRIPTOR = 28,
    CUDNN_BACKEND_HEURIN_DESCRIPTOR = 29,
    CUDNN_BACKEND_ENGINHEUR_DESCRIPTOR = 30,
    CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR = 31,
    CUDNN_BACKEND_KNOB_DESCRIPTOR = 32,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 33,
    CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR = 34,
    CUDNN_BACKEND_OPERATION_RNN_DESCRIPTOR = 35
};

// ── Attribute IDs ────────────────────────────────────────────────────────────
enum cudnnBackendAttributeName_t {
    CUDNN_ATTR_POINTWISE_MODE = 0,
    CUDNN_ATTR_POINTWISE_MATH_PREC = 1,
    CUDNN_ATTR_POINTWISE_NAN_PROPAGATION = 2,
    CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = 10,
    CUDNN_ATTR_CONVOLUTION_DILATIONS = 11,
    CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 12,
    CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = 13,
    CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = 14,
    CUDNN_ATTR_CONVOLUTION_STRIDES = 15,
    CUDNN_ATTR_CONVOLUTION_GROUP_COUNT = 16,
    CUDNN_ATTR_CONVOLUTION_BWD_DATA_ALGO = 17,
    CUDNN_ATTR_CONVOLUTION_BWD_FILTER_ALGO = 18,
    CUDNN_ATTR_CONVOLUTION_FWD_ALGO = 19,
    CUDNN_ATTR_TENSOR_DIMENSIONS = 30,
    CUDNN_ATTR_TENSOR_STRIDES = 31,
    CUDNN_ATTR_TENSOR_DATA_TYPE = 32,
    CUDNN_ATTR_TENSOR_VECTORIZED_DIM = 33,
    CUDNN_ATTR_TENSOR_VECTOR_COUNT = 34,
    CUDNN_ATTR_TENSOR_IS_VIRTUAL = 35,
    CUDNN_ATTR_TENSOR_IS_BY_VALUE = 36,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = 40,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = 41,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = 42,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = 43,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = 44,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = 45,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = 46,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = 47,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = 48,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = 49,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = 50,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 51,
    CUDNN_ATTR_OPERATION_MATMUL_ADESC = 60,
    CUDNN_ATTR_OPERATION_MATMUL_BDESC = 61,
    CUDNN_ATTR_OPERATION_MATMUL_CDESC = 62,
    CUDNN_ATTR_OPERATION_MATMUL_DDESC = 63,
    CUDNN_ATTR_OPERATION_MATMUL_MDESC = 64,
    CUDNN_ATTR_OPERATIONPOOLING_XDESC = 70,
    CUDNN_ATTR_OPERATIONPOOLING_YDESC = 71,
    CUDNN_ATTR_OPERATIONPOOLING_IDXDESC = 72,
    CUDNN_ATTR_OPERATIONPOOLING_PDESC = 73,
    CUDNN_ATTR_OPERATIONPOOLING_BWD_DXDESC = 74,
    CUDNN_ATTR_OPERATIONPOOLING_BWD_DYDESC = 75,
    CUDNN_ATTR_OPERATIONPOOLING_BWD_IDXDESC = 76,
    CUDNN_ATTR_OPERATIONPOOLING_BWD_PDESC = 77,
    CUDNN_ATTR_OPERATION_ACTIVATION_XDESC = 80,
    CUDNN_ATTR_OPERATION_ACTIVATION_YDESC = 81,
    CUDNN_ATTR_OPERATION_ACTIVATION_DESC = 82,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC = 83,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC = 84,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_YDESC = 85,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DESC = 86,
    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 100,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 101,
    CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 102,
    CUDNN_ATTR_VARIANT_PACK_WORKSPACE = 103,
    CUDNN_ATTR_ENGINEHEUR_MODE = 110,
    CUDNN_ATTR_ENGINECFG_ENGINE = 120,
    CUDNN_ATTR_ENGINECFG_KNOB_CHOICES = 121,
    CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 122,
    CUDNN_ATTR_EXECUTION_PLAN_HANDLE = 130,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 131,
    CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 132,
    // Alpha / Beta scalars for backend operations
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = 160,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = 161,
    CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_ALPHA = 162,
    CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_BETA = 163,
    CUDNN_ATTR_OPERATION_POOLING_FORWARD_ALPHA = 164,
    CUDNN_ATTR_OPERATION_POOLING_FORWARD_BETA = 165,
    CUDNN_ATTR_OPERATION_SOFTMAX_ALPHA = 166,
    CUDNN_ATTR_OPERATION_SOFTMAX_BETA = 167,
    CUDNN_ATTR_OPERATION_REDUCTION_ALPHA = 168,
    CUDNN_ATTR_OPERATION_REDUCTION_BETA = 169,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = 170,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = 171,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = 172,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = 173,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_ALPHA = 174,
    CUDNN_ATTR_OPERATION_ACTIVATION_BWD_BETA = 175,
    CUDNN_ATTR_OPERATION_POOLING_BWD_ALPHA = 176,
    CUDNN_ATTR_OPERATION_POOLING_BWD_BETA = 177,
    CUDNN_ATTR_OPERATION_REDUCTION_DESC = 178,
    CUDNN_ATTR_OPERATION_MATMUL_ALPHA = 179,
    CUDNN_ATTR_OPERATION_MATMUL_BETA = 180,
    // Graph traversal
    CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 500,
    CUDNN_ATTR_OPERATIONSET_OPS = 501
};

// ── Internal backend node ────────────────────────────────────────────────────
struct BackendNode {
    cudnnBackendDescriptorType_t type;
    std::unordered_map<int, std::vector<uint64_t>> attrs; // attrId -> list of values
    bool finalized = false;
};

namespace {
std::unordered_map<uintptr_t, BackendNode> g_backendNodes;
uintptr_t g_nextBackendId = 1;

static BackendNode* getNode(uintptr_t id) {
    auto it = g_backendNodes.find(id);
    return (it != g_backendNodes.end()) ? &it->second : nullptr;
}

static const std::vector<uint64_t>* getAttrVec(const BackendNode* node, int attr) {
    if (!node) return nullptr;
    auto it = node->attrs.find(attr);
    return (it != node->attrs.end()) ? &it->second : nullptr;
}

static uintptr_t getAttrUint64(const BackendNode* node, int attr, uintptr_t def = 0) {
    auto* v = getAttrVec(node, attr);
    return (v && !v->empty()) ? static_cast<uintptr_t>((*v)[0]) : def;
}

static float getAttrFloat(const BackendNode* node, int attr, float def = 0.0f) {
    auto* v = getAttrVec(node, attr);
    if (!v || v->empty()) return def;
    float f = 0.0f;
    std::memcpy(&f, &(*v)[0], sizeof(float));
    return f;
}

static TensorDesc buildTensorDesc(uintptr_t nodeId) {
    TensorDesc td{};
    td.n = 1; td.c = 1; td.h = 1; td.w = 1;
    td.dtype = CUDNN_DATA_FLOAT;
    td.fmt = CUDNN_TENSOR_NCHW;
    const BackendNode* node = getNode(nodeId);
    if (!node) return td;
    auto* dims = getAttrVec(node, CUDNN_ATTR_TENSOR_DIMENSIONS);
    if (dims) {
        if (dims->size() > 0) td.n = static_cast<int>((*dims)[0]);
        if (dims->size() > 1) td.c = static_cast<int>((*dims)[1]);
        if (dims->size() > 2) td.h = static_cast<int>((*dims)[2]);
        if (dims->size() > 3) td.w = static_cast<int>((*dims)[3]);
    }
    auto* dt = getAttrVec(node, CUDNN_ATTR_TENSOR_DATA_TYPE);
    if (dt && !dt->empty()) td.dtype = static_cast<cudnnDataType_t>((*dt)[0]);
    return td;
}

static FilterDesc buildFilterDesc(uintptr_t nodeId) {
    FilterDesc fd{};
    fd.k = 1; fd.c = 1; fd.r = 1; fd.s = 1;
    fd.dtype = CUDNN_DATA_FLOAT;
    const BackendNode* node = getNode(nodeId);
    if (!node) return fd;
    auto* dims = getAttrVec(node, CUDNN_ATTR_TENSOR_DIMENSIONS);
    if (dims) {
        if (dims->size() > 0) fd.k = static_cast<int>((*dims)[0]);
        if (dims->size() > 1) fd.c = static_cast<int>((*dims)[1]);
        if (dims->size() > 2) fd.r = static_cast<int>((*dims)[2]);
        if (dims->size() > 3) fd.s = static_cast<int>((*dims)[3]);
    }
    auto* dt = getAttrVec(node, CUDNN_ATTR_TENSOR_DATA_TYPE);
    if (dt && !dt->empty()) fd.dtype = static_cast<cudnnDataType_t>((*dt)[0]);
    return fd;
}

static ConvDesc buildConvDesc(uintptr_t nodeId) {
    ConvDesc cd{};
    cd.pad_h = 0; cd.pad_w = 0; cd.str_h = 1; cd.str_w = 1; cd.dil_h = 1; cd.dil_w = 1;
    const BackendNode* node = getNode(nodeId);
    if (!node) return cd;
    auto* pre = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS);
    if (pre) {
        if (pre->size() > 0) cd.pad_h = static_cast<int>((*pre)[0]);
        if (pre->size() > 1) cd.pad_w = static_cast<int>((*pre)[1]);
    }
    auto* str = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_STRIDES);
    if (str) {
        if (str->size() > 0) cd.str_h = static_cast<int>((*str)[0]);
        if (str->size() > 1) cd.str_w = static_cast<int>((*str)[1]);
    }
    auto* dil = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_DILATIONS);
    if (dil) {
        if (dil->size() > 0) cd.dil_h = static_cast<int>((*dil)[0]);
        if (dil->size() > 1) cd.dil_w = static_cast<int>((*dil)[1]);
    }
    return cd;
}

static std::unordered_map<uintptr_t, void*> parseVariantPack(void* variantPack) {
    std::unordered_map<uintptr_t, void*> ptrs;
    if (!variantPack) return ptrs;
    const BackendNode* vp = getNode(reinterpret_cast<uintptr_t>(variantPack));
    if (!vp) return ptrs;
    auto* ids = getAttrVec(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    auto* dps = getAttrVec(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    if (!ids || !dps) return ptrs;
    size_t n = std::min(ids->size(), dps->size());
    for (size_t i = 0; i < n; ++i) {
        ptrs[static_cast<uintptr_t>((*ids)[i])] = reinterpret_cast<void*>((*dps)[i]);
    }
    return ptrs;
}

static std::vector<uintptr_t> parseOperationSet(uintptr_t opSetId) {
    std::vector<uintptr_t> ops;
    const BackendNode* node = getNode(opSetId);
    if (!node) return ops;
    auto* v = getAttrVec(node, CUDNN_ATTR_OPERATIONSET_OPS);
    if (!v) return ops;
    for (auto val : *v) ops.push_back(static_cast<uintptr_t>(val));
    return ops;
}

static uintptr_t traversePlanToOpSet(uintptr_t planId) {
    const BackendNode* plan = getNode(planId);
    if (!plan) return 0;
    // If the plan itself is an operation set, return it directly
    if (plan->type == CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR) return planId;
    // Plan → Engine Config
    uintptr_t ecId = getAttrUint64(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    if (!ecId) return 0;
    const BackendNode* ec = getNode(ecId);
    if (!ec) return 0;
    // Engine Config → Engine
    uintptr_t engId = getAttrUint64(ec, CUDNN_ATTR_ENGINECFG_ENGINE);
    if (!engId) return 0;
    const BackendNode* eng = getNode(engId);
    if (!eng) return 0;
    // Engine → Operation Graph
    return getAttrUint64(eng, CUDNN_ATTR_ENGINE_OPERATION_GRAPH);
}

} // namespace

extern "C" {

// ── Descriptor lifecycle ───────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t type, void** descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    uintptr_t id = g_nextBackendId++;
    BackendNode& node = g_backendNodes[id];
    node.type = type;
    node.finalized = false;
    *descriptor = reinterpret_cast<void*>(id);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_SUCCESS;
    g_backendNodes.erase(reinterpret_cast<uintptr_t>(descriptor));
    return CUDNN_STATUS_SUCCESS;
}

// ── Attribute set/get ──────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendSetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int /*attributeType*/, // 0=char,1=float,2=double,3=int32,4=int64,5=uint64,6=descriptor
                                       int64_t elementCount,
                                       const void* arrayOfElements) {
    if (!descriptor || !arrayOfElements || elementCount <= 0) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;

    BackendNode& node = it->second;
    std::vector<uint64_t>& vec = node.attrs[static_cast<int>(attrName)];
    vec.resize(elementCount);
    std::memcpy(vec.data(), arrayOfElements, elementCount * sizeof(uint64_t));
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int /*attributeType*/,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) {
    if (!descriptor || !elementCount || !arrayOfElements) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;

    const BackendNode& node = it->second;
    auto attrIt = node.attrs.find(static_cast<int>(attrName));
    if (attrIt == node.attrs.end()) {
        *elementCount = 0;
        return CUDNN_STATUS_INVALID_VALUE;
    }

    int64_t n = static_cast<int64_t>(attrIt->second.size());
    *elementCount = n;
    int64_t copyN = std::min(requestedElementCount, n);
    std::memcpy(arrayOfElements, attrIt->second.data(), copyN * sizeof(uint64_t));
    return (requestedElementCount >= n) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_INVALID_VALUE;
}

// ── Finalize ─────────────────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendFinalize(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;
    it->second.finalized = true;
    return CUDNN_STATUS_SUCCESS;
}

// ── Initialize / Populate / Execute ──────────────────────────────────────────

cudnnStatus_t cudnnBackendInitialize(void* /*descriptor*/) {
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendPopulate(void* /*handle*/, void* /*descriptor*/) {
    return CUDNN_STATUS_SUCCESS;
}

// Minimal cublas typedefs needed for matmul dispatch
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
static constexpr int CUBLAS_OP_N = 0;
static constexpr int CUDA_R_32F = 0;
static constexpr int CUBLAS_GEMM_DEFAULT = 0;
static constexpr int CUBLAS_STATUS_SUCCESS = 0;

// ── Forward declarations of legacy cuDNN shim functions ────────────────────
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnFilterDescriptor_t, const void*, cudnnConvolutionDescriptor_t, int, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);
cublasStatus_t cublasGemmEx(cublasHandle_t, int, int, int, int, int, const void*, const void*, int, int, const void*, int, int, const void*, void*, int, int, int, int);
cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t, const void*, cudnnFilterDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*, cudnnFilterDescriptor_t, void*);
cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t, int, int, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t, int, int, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnReduceTensor(cudnnHandle_t, cudnnReduceTensorDescriptor_t, void*, size_t, void*, size_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, void* plan, void* variantPack) {
    if (!plan || !variantPack) return CUDNN_STATUS_INVALID_VALUE;

    // Parse variant pack: map from backend node ID -> data pointer
    auto dataPtrs = parseVariantPack(variantPack);

    // Traverse plan -> engine config -> engine -> operation graph
    uintptr_t opSetId = traversePlanToOpSet(reinterpret_cast<uintptr_t>(plan));
    if (!opSetId) return CUDNN_STATUS_INVALID_VALUE;

    // Get list of operations
    std::vector<uintptr_t> ops = parseOperationSet(opSetId);
    if (ops.empty()) return CUDNN_STATUS_INVALID_VALUE;

    // Execute each operation in order (topological sort omitted for simple graphs)
    for (uintptr_t opId : ops) {
        const BackendNode* opNode = getNode(opId);
        if (!opNode) return CUDNN_STATUS_INVALID_VALUE;

        switch (opNode->type) {
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);
            uintptr_t wId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y);
            uintptr_t cId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* wPtr = dataPtrs[wId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !wPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            FilterDesc wDesc = buildFilterDesc(wId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ConvDesc   cDesc = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionForward(handle, &alpha, &xDesc, xPtr,
                                                        &wDesc, wPtr, &cDesc,
                                                        0, nullptr, 0, &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: {
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY);
            uintptr_t wId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX);
            uintptr_t cId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, 0.0f);
            void* dyPtr = dataPtrs[dyId];
            void* wPtr  = dataPtrs[wId];
            void* dxPtr = dataPtrs[dxId];
            if (!dyPtr || !wPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc dyDesc = buildTensorDesc(dyId);
            FilterDesc wDesc  = buildFilterDesc(wId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            ConvDesc   cDesc  = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionBackwardData(handle, &alpha, &wDesc, wPtr,
                                                             &dyDesc, dyPtr, &cDesc,
                                                             static_cast<cudnnConvolutionBwdDataAlgo_t>(0),
                                                             nullptr, 0, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: {
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X);
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY);
            uintptr_t dwId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW);
            uintptr_t cId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, 0.0f);
            void* xPtr  = dataPtrs[xId];
            void* dyPtr = dataPtrs[dyId];
            void* dwPtr = dataPtrs[dwId];
            if (!xPtr || !dyPtr || !dwPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc  = buildTensorDesc(xId);
            TensorDesc dyDesc = buildTensorDesc(dyId);
            FilterDesc dwDesc = buildFilterDesc(dwId);
            ConvDesc   cDesc  = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionBackwardFilter(handle, &alpha, &xDesc, xPtr,
                                                               &dyDesc, dyPtr, &cDesc,
                                                               static_cast<cudnnConvolutionBwdFilterAlgo_t>(0),
                                                               nullptr, 0, &beta, &dwDesc, dwPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_ACTIVATION_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            uintptr_t aId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ActDesc* actDesc = reinterpret_cast<ActDesc*>(aId);
            if (!actDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnActivationForward(handle, actDesc, &alpha, &xDesc, xPtr,
                                                       &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_ACTIVATION_BACKWARD_DESCRIPTOR: {
            uintptr_t yId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_YDESC);
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);
            uintptr_t aId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_BETA, 0.0f);
            void* yPtr  = dataPtrs[yId];
            void* dyPtr = dataPtrs[dyId];
            void* dxPtr = dataPtrs[dxId];
            if (!yPtr || !dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc yDesc  = buildTensorDesc(yId);
            TensorDesc dyDesc = buildTensorDesc(dyId);
            // Activation backward also needs xDesc; assume same shape as dxDesc
            TensorDesc xDesc  = buildTensorDesc(dxId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            ActDesc* actDesc = reinterpret_cast<ActDesc*>(aId);
            if (!actDesc) return CUDNN_STATUS_INVALID_VALUE;
            // x and dx share the same buffer in this Backend API mapping
            void* xPtr = dxPtr;
            cudnnStatus_t s = cudnnActivationBackward(handle, actDesc, &alpha,
                                                        &yDesc, yPtr, &dyDesc, dyPtr,
                                                        &xDesc, xPtr, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_YDESC);
            uintptr_t pId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_PDESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            PoolDesc* poolDesc = reinterpret_cast<PoolDesc*>(pId);
            if (!poolDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnPoolingForward(handle, poolDesc, &alpha, &xDesc, xPtr,
                                                    &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_POOLING_BACKWARD_DESCRIPTOR: {
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_DXDESC);
            uintptr_t pId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_PDESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_BWD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_BWD_BETA, 0.0f);
            void* dyPtr = dataPtrs[dyId];
            void* dxPtr = dataPtrs[dxId];
            if (!dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            // Pooling backward needs yDesc and xDesc; assume same shape as dy/dx
            TensorDesc dyDesc = buildTensorDesc(dyId);
            TensorDesc yDesc  = buildTensorDesc(dyId);
            TensorDesc xDesc  = buildTensorDesc(dxId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            PoolDesc* poolDesc = reinterpret_cast<PoolDesc*>(pId);
            if (!poolDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnPoolingBackward(handle, poolDesc, &alpha,
                                                     &yDesc, dyPtr, &dyDesc, dyPtr,
                                                     &xDesc, dxPtr, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_SOFTMAX_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_SOFTMAX_BETA, 0.0f);
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            cudnnStatus_t s = cudnnSoftmaxForward(handle, 0, 0, &alpha, &xDesc, xPtr,
                                                    &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_REDUCTION_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_REDUCTION_BETA, 0.0f);
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            uintptr_t rId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_REDUCTION_DESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ReduceTensorDesc* reduceDesc = reinterpret_cast<ReduceTensorDesc*>(rId);
            if (!reduceDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnReduceTensor(handle, reduceDesc,
                                                  nullptr, 0, nullptr, 0,
                                                  &alpha, &xDesc, xPtr, &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_MATMUL_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_MATMUL_BETA, 0.0f);
            uintptr_t aId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_ADESC);
            uintptr_t bId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_BDESC);
            uintptr_t cId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_CDESC);
            void* aPtr = dataPtrs[aId];
            void* bPtr = dataPtrs[bId];
            void* cPtr = dataPtrs[cId];
            if (!aPtr || !bPtr || !cPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc aDesc = buildTensorDesc(aId);
            TensorDesc bDesc = buildTensorDesc(bId);
            TensorDesc cDesc = buildTensorDesc(cId);
            int m = cDesc.n;
            int n = cDesc.c;
            int k = aDesc.c;
            if (m == 0 || n == 0 || k == 0) return CUDNN_STATUS_INVALID_VALUE;
            cublasStatus_t s = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             m, n, k, &alpha,
                                             aPtr, CUDA_R_32F, m,
                                             bPtr, CUDA_R_32F, k,
                                             &beta, cPtr, CUDA_R_32F, m,
                                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (s != CUBLAS_STATUS_SUCCESS) return CUDNN_STATUS_INTERNAL_ERROR;
            break;
        }
        default:
            return CUDNN_STATUS_NOT_SUPPORTED;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Engine / Plan / Heuristics stubs ─────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateReexecutable(cudnnHandle_t /*handle*/, void* /*opGraph*/, void** /*plan*/) {
    // Re-executable plans are not supported in CPU emulation
    return CUDNN_STATUS_NOT_SUPPORTED;
}

} // extern "C"
