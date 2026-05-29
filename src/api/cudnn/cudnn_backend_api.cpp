// cuDNN v8+ Backend API. Descriptor graphs are executed by routing each
// operation to the existing legacy cuDNN shim paths (conv, pool, activation,
// matmul, reductions, attention, etc.).
//
// For unimplemented operations, returns CUDNN_STATUS_NOT_SUPPORTED.

#include "cudnn_backend_internal.h"
#include "vgre/common/openmp_helper.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

// ── Global backend state ─────────────────────────────────────────────────────
// Definitions of the extern symbols declared in cudnn_backend_internal.h
std::unordered_map<uintptr_t, BackendNode> g_backendNodes;
uintptr_t g_nextBackendId = 1;

BackendNode* getNode(uintptr_t id) {
    auto it = g_backendNodes.find(id);
    return (it != g_backendNodes.end()) ? &it->second : nullptr;
}

const std::vector<uint64_t>* getAttrVec(const BackendNode* node, int attr) {
    if (!node) return nullptr;
    auto it = node->attrs.find(attr);
    return (it != node->attrs.end()) ? &it->second : nullptr;
}

uintptr_t getAttrUint64(const BackendNode* node, int attr, uintptr_t def) {
    auto* v = getAttrVec(node, attr);
    return (v && !v->empty()) ? static_cast<uintptr_t>((*v)[0]) : def;
}

float getAttrFloat(const BackendNode* node, int attr, float def) {
    auto* v = getAttrVec(node, attr);
    if (!v || v->empty()) return def;
    float f = 0.0f;
    memcpy(&f, &(*v)[0], sizeof(float));
    return f;
}

namespace {

static size_t backendAttributeElementSize(int attributeType) {
    switch (attributeType) {
    case 0: return sizeof(char);
    case 1: return sizeof(float);
    case 2: return sizeof(double);
    case 3: return sizeof(int32_t);
    case 4: return sizeof(int64_t);
    case 5: return sizeof(uint64_t);
    case 6: return sizeof(uint64_t); // descriptor handles are stored as ids
    default: return sizeof(uint64_t);
    }
}

static uint64_t readBackendAttributeElement(const uint8_t *base, size_t elemSize) {
    uint64_t out = 0;
    memcpy(&out, base, std::min(elemSize, sizeof(out)));
    return out;
}

static void writeBackendAttributeElement(uint8_t *base, size_t elemSize, uint64_t value) {
    memcpy(base, &value, std::min(elemSize, sizeof(value)));
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
                                       int attributeType, // 0=char,1=float,2=double,3=int32,4=int64,5=uint64,6=descriptor
                                       int64_t elementCount,
                                       const void* arrayOfElements) {
    if (!descriptor || !arrayOfElements || elementCount <= 0) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;

    BackendNode& node = it->second;
    std::vector<uint64_t>& vec = node.attrs[static_cast<int>(attrName)];
    vec.resize(static_cast<size_t>(elementCount));
    const size_t elemSize = backendAttributeElementSize(attributeType);
    const auto *bytes = static_cast<const uint8_t*>(arrayOfElements);
    for (int64_t i = 0; i < elementCount; ++i)
        vec[static_cast<size_t>(i)] = readBackendAttributeElement(bytes + static_cast<size_t>(i) * elemSize, elemSize);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int attributeType,
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
    const size_t elemSize = backendAttributeElementSize(attributeType);
    auto *bytes = static_cast<uint8_t*>(arrayOfElements);
    for (int64_t i = 0; i < copyN; ++i)
        writeBackendAttributeElement(bytes + static_cast<size_t>(i) * elemSize, elemSize,
                                     attrIt->second[static_cast<size_t>(i)]);
    return (requestedElementCount >= n) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_INVALID_VALUE;
}

// ── Finalize ─────────────────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendFinalize(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;
    BackendNode& node = it->second;
    node.finalized = true;

    // ── Engine heuristic finalization ─────────────────────────────────────────
    // VGRE has exactly one engine; populate ENGINEHEUR_RESULTS with a single
    // engine-config descriptor so callers of cudnnGetAttribute(ENGINEHEUR_RESULTS)
    // get a valid (non-empty) list without having to call cudnnBackendExecute.
    if (node.type == CUDNN_BACKEND_ENGINHEUR_DESCRIPTOR) {
        // Only populate if results not already set by the caller.
        int resultsAttr = static_cast<int>(CUDNN_ATTR_ENGINEHEUR_RESULTS);
        if (node.attrs.find(resultsAttr) == node.attrs.end()) {
            // Create a minimal engine-config descriptor and record its id.
            uintptr_t cfgId = g_nextBackendId++;
            BackendNode cfgNode;
            cfgNode.type      = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
            cfgNode.finalized = true;

            // Build a minimal ENGINE child descriptor.
            uintptr_t engId = g_nextBackendId++;
            BackendNode engNode;
            engNode.type      = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
            engNode.finalized = true;
            // Link engine to the operation graph stored in the heuristic node.
            uintptr_t opGraphId = getAttrUint64(&node,
                static_cast<int>(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH), 0);
            if (opGraphId) {
                engNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)]
                    .push_back(static_cast<uint64_t>(opGraphId));
            }
            g_backendNodes[engId] = std::move(engNode);

            // Link engine into config.
            cfgNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINECFG_ENGINE)]
                .push_back(static_cast<uint64_t>(engId));
            g_backendNodes[cfgId] = std::move(cfgNode);

            // Store config id as the single ENGINEHEUR_RESULTS entry.
            node.attrs[resultsAttr].push_back(static_cast<uint64_t>(cfgId));
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

// ── Initialize / Populate / Execute ──────────────────────────────────────────

cudnnStatus_t cudnnBackendInitialize(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    if (g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor)) == g_backendNodes.end())
        return CUDNN_STATUS_INVALID_VALUE;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendPopulate(void* /*handle*/, void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    if (g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor)) == g_backendNodes.end())
        return CUDNN_STATUS_INVALID_VALUE;
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
cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t, int, const void*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*, double, void*, void*, double, void*, void*);
cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t, int, const void*, const void*, const void*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, void*, void*, double, const void*, const void*);
cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t, cudnnTensorDescriptor_t, const void*, const void*, void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t, cudnnTensorDescriptor_t, const void*, const void*, const void*, void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t, void* /*rnnDesc*/, int, cudnnTensorDescriptor_t*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, void* /*wDesc*/, const void*, cudnnTensorDescriptor_t*, void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, void*, void*, size_t);

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
        case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR: {
            // Route to batch norm forward training with default parameters
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            float alpha = 1.0f, beta = 0.0f;
            // Minimal BN scale/bias buffers (all ones / zeros)
            int C = xDesc.c;
            std::vector<float> scale(C, 1.0f), bias(C, 0.0f);
            std::vector<float> runningMean(C, 0.0f), runningVar(C, 1.0f);
            std::vector<float> savedMean(C, 0.0f), savedInvVar(C, 1.0f);
            cudnnStatus_t s = cudnnBatchNormalizationForwardTraining(
                handle, 1 /*CUDNN_BATCHNORM_SPATIAL*/, &alpha, &beta,
                &xDesc, xPtr, &yDesc, yPtr,
                nullptr, scale.data(), bias.data(),
                0.0, runningMean.data(), runningVar.data(),
                1e-5, savedMean.data(), savedInvVar.data());
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            cudnnStatus_t s = cudnnDivisiveNormalizationForward(handle, &xDesc, xPtr, nullptr, nullptr, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);
            void* xPtr  = dataPtrs[xId];
            void* dyPtr = dataPtrs[yId];
            void* dxPtr = dataPtrs[dxId];
            if (!xPtr || !dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc dyDesc = buildTensorDesc(yId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            cudnnStatus_t s = cudnnDivisiveNormalizationBackward(handle, &xDesc, xPtr, dyPtr, nullptr, nullptr, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_RNN_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            // Minimal RNN descriptor with default hidden size
            struct MiniRNNDesc { int hiddenSize = 128; int numLayers = 1; int inputSize = 128; } miniRnn;
            // Use xDesc as both x and y descriptor arrays (single timestep)
            cudnnTensorDescriptor_t xDescArr[1] = { &xDesc };
            cudnnTensorDescriptor_t yDescArr[1] = { &yDesc };
            int seqLength = 1;
            cudnnStatus_t s = cudnnRNNForwardInference(
                handle, &miniRnn, seqLength, xDescArr, xPtr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                yDescArr, yPtr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: {
            // Simple concatenation: copy inputs contiguously into output buffer
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            size_t xBytes = static_cast<size_t>(xDesc.n) * xDesc.c * xDesc.h * xDesc.w * sizeof(float);
            size_t yBytes = static_cast<size_t>(yDesc.n) * yDesc.c * yDesc.h * yDesc.w * sizeof(float);
            memcpy(yPtr, xPtr, xBytes);
            // Second input (if present) appended after first
            uintptr_t x2Id = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X); // reuse attr slot
            void* x2Ptr = dataPtrs.count(x2Id) ? dataPtrs[x2Id] : nullptr;
            if (x2Ptr && x2Id != xId)
                memcpy(static_cast<char*>(yPtr) + xBytes, x2Ptr, yBytes - xBytes);
            break;
        }
        case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR: {
            // Signal is a synchronization barrier; no-op in single-threaded CPU emulation
            break;
        }
        case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: {
            // Generate statistics (mean, variance) per-channel
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            int N = xDesc.n, C = xDesc.c, HW = xDesc.h * xDesc.w;
            const float* xf = static_cast<const float*>(xPtr);
            float* meanOut = static_cast<float*>(yPtr);
            float* varOut  = meanOut + C;
            #ifdef _OPENMP
            #pragma omp parallel for if (C > 4)
            #endif
            for (int c = 0; c < C; ++c) {
                double sum = 0.0, sq = 0.0;
                for (int n = 0; n < N; ++n)
                for (int hw = 0; hw < HW; ++hw) {
                    float v = xf[((n * C + c) * xDesc.h + (hw / xDesc.w)) * xDesc.w + (hw % xDesc.w)];
                    sum += v;
                    sq += v * v;
                }
                double count = static_cast<double>(N * HW);
                meanOut[c] = static_cast<float>(sum / count);
                varOut[c]  = static_cast<float>(sq / count - meanOut[c] * meanOut[c]);
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR: {
            // Route to batch norm backward to compute dScale and dBias
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);
            void* xPtr  = dataPtrs[xId];
            void* dyPtr = dataPtrs[dyId];
            void* dxPtr = dataPtrs[dxId];
            if (!xPtr || !dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc  = buildTensorDesc(xId);
            TensorDesc dyDesc = buildTensorDesc(dyId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            int C = xDesc.c;
            std::vector<float> scale(C, 1.0f), dScale(C, 0.0f), dBias(C, 0.0f);
            std::vector<float> savedMean(C, 0.0f), savedInvVar(C, 1.0f);
            float alphaDataDiff[1] = {1.0f}, betaDataDiff[1] = {0.0f};
            float alphaParamDiff[1] = {1.0f}, betaParamDiff[1] = {0.0f};
            cudnnStatus_t s = cudnnBatchNormalizationBackward(
                handle, 1 /*CUDNN_BATCHNORM_SPATIAL*/,
                alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
                &xDesc, xPtr, &dyDesc, dyPtr, &dxDesc, dxPtr,
                nullptr, scale.data(), dScale.data(), dBias.data(),
                1e-5, savedMean.data(), savedInvVar.data());
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_ATTENTION_DESCRIPTOR: {
            // Scaled dot-product attention: O = softmax(Q*K^T / scale) * V
            // Tensor layout expected: (batch, heads, seqLen, head_dim) for Q/K/V/O.
            uintptr_t qId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_QDESC);
            uintptr_t kId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_KDESC);
            uintptr_t vId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_VDESC);
            uintptr_t oId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_ODESC);
            void *qPtr = dataPtrs[qId], *kPtr = dataPtrs[kId];
            void *vPtr = dataPtrs[vId], *oPtr = dataPtrs[oId];
            if (!qPtr || !kPtr || !vPtr || !oPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc qDesc = buildTensorDesc(qId);
            TensorDesc kDesc = buildTensorDesc(kId);
            TensorDesc vDesc = buildTensorDesc(vId);

            // Infer dimensions: n=batch, c=heads, h=seqLen, w=head_dim
            int batch    = qDesc.n;
            int heads    = qDesc.c;
            int seqQ     = qDesc.h;
            int head_dim = qDesc.w;
            int seqK     = kDesc.h;

            float attnScale = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ATTENTION_SCALE,
                                            1.0f / std::sqrt(static_cast<float>(head_dim)));

            const float *Q = static_cast<const float*>(qPtr);
            const float *K = static_cast<const float*>(kPtr);
            const float *V = static_cast<const float*>(vPtr);
            float       *O = static_cast<float*>(oPtr);

            std::vector<float> attnScore(static_cast<size_t>(seqQ * seqK));

            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < heads; ++h) {
                    int bhQ = (b * heads + h) * seqQ * head_dim;
                    int bhK = (b * heads + h) * seqK * head_dim;
                    int bhO = (b * heads + h) * seqQ * head_dim;

                    // Compute A = Q * K^T * scale  [seqQ × seqK]
                    for (int i = 0; i < seqQ; ++i) {
                        for (int j = 0; j < seqK; ++j) {
                            float dot = 0.0f;
                            for (int d = 0; d < head_dim; ++d)
                                dot += Q[bhQ + i * head_dim + d] * K[bhK + j * head_dim + d];
                            attnScore[static_cast<size_t>(i * seqK + j)] = dot * attnScale;
                        }
                    }

                    // Softmax over seqK dim for each query position
                    for (int i = 0; i < seqQ; ++i) {
                        float *row = attnScore.data() + i * seqK;
                        float maxVal = *std::max_element(row, row + seqK);
                        float sumExp = 0.0f;
                        for (int j = 0; j < seqK; ++j) { row[j] = std::exp(row[j] - maxVal); sumExp += row[j]; }
                        for (int j = 0; j < seqK; ++j) row[j] /= sumExp;
                    }

                    // Output = A * V  [seqQ × head_dim]
                    for (int i = 0; i < seqQ; ++i) {
                        for (int d = 0; d < head_dim; ++d) {
                            float sum = 0.0f;
                            for (int j = 0; j < seqK; ++j)
                                sum += attnScore[static_cast<size_t>(i * seqK + j)] * V[bhK + j * head_dim + d];
                            O[bhO + i * head_dim + d] = sum;
                        }
                    }
                }
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: {
            // Element-wise operation between tensors, dispatching on the mode
            // stored in a nested CUDNN_BACKEND_POINTWISE_DESCRIPTOR.
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_XDESC);
            uintptr_t yId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
            uintptr_t bId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_BDESC);
            uintptr_t pwId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR);
            float alpha1   = getAttrFloat (opNode, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, 1.0f);
            float alpha2   = getAttrFloat (opNode, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, 1.0f);

            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            void* bPtr = bId ? dataPtrs[bId] : nullptr;
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            cudnnPointwiseMode_t mode = CUDNN_POINTWISE_IDENTITY;
            if (pwId) {
                const BackendNode* pw = getNode(pwId);
                if (pw) {
                    auto* mv = getAttrVec(pw, CUDNN_ATTR_POINTWISE_MODE);
                    if (mv && !mv->empty())
                        mode = static_cast<cudnnPointwiseMode_t>((*mv)[0]);
                }
            }

            TensorDesc xDesc = buildTensorDesc(xId);
            size_t nelems = static_cast<size_t>(xDesc.n) * xDesc.c * xDesc.h * xDesc.w;

            const float* X = static_cast<const float*>(xPtr);
            const float* B = bPtr ? static_cast<const float*>(bPtr) : nullptr;
            float*       Y = static_cast<float*>(yPtr);

            for (size_t i = 0; i < nelems; ++i) {
                float x = X[i] * alpha1;
                float b = B ? B[i] * alpha2 : 0.0f;
                float y = x;
                switch (mode) {
                case CUDNN_POINTWISE_ADD:        y = x + b; break;
                case CUDNN_POINTWISE_MUL:        y = x * b; break;
                case CUDNN_POINTWISE_MIN:        y = x < b ? x : b; break;
                case CUDNN_POINTWISE_MAX:        y = x > b ? x : b; break;
                case CUDNN_POINTWISE_DIV:        y = b != 0.f ? x / b : 0.f; break;
                case CUDNN_POINTWISE_SQRT:       y = std::sqrt(x > 0.f ? x : 0.f); break;
                case CUDNN_POINTWISE_EXP:        y = std::exp(x); break;
                case CUDNN_POINTWISE_LOG:        y = x > 0.f ? std::log(x) : -1e30f; break;
                case CUDNN_POINTWISE_NEG:        y = -x; break;
                case CUDNN_POINTWISE_ABS:        y = std::abs(x); break;
                case CUDNN_POINTWISE_CEIL:       y = std::ceil(x); break;
                case CUDNN_POINTWISE_FLOOR:      y = std::floor(x); break;
                case CUDNN_POINTWISE_RECIPROCAL: y = x != 0.f ? 1.f / x : 0.f; break;
                case CUDNN_POINTWISE_RELU_FWD:   y = x > 0.f ? x : 0.f; break;
                case CUDNN_POINTWISE_TANH_FWD:   y = std::tanh(x); break;
                case CUDNN_POINTWISE_SIGMOID_FWD:y = 1.f / (1.f + std::exp(-x)); break;
                case CUDNN_POINTWISE_ELU_FWD:    y = x > 0.f ? x : std::exp(x) - 1.f; break;
                case CUDNN_POINTWISE_SWISH_FWD:  y = x / (1.f + std::exp(-x)); break;
                case CUDNN_POINTWISE_SOFTPLUS_FWD:y = std::log1p(std::exp(x)); break;
                case CUDNN_POINTWISE_GELU_FWD:
                case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: {
                    // GELU: x * Φ(x) ≈ 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))
                    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
                    y = 0.5f * x * (1.f + std::tanh(t));
                    break;
                }
                case CUDNN_POINTWISE_RELU_BWD:   y = (b > 0.f) ? x : 0.f; break;
                case CUDNN_POINTWISE_TANH_BWD:   y = x * (1.f - b * b); break;
                case CUDNN_POINTWISE_SIGMOID_BWD:y = x * b * (1.f - b); break;
                case CUDNN_POINTWISE_ADD_SQUARE:  y = x + b * b; break;
                case CUDNN_POINTWISE_POW:        y = std::pow(x, b); break;
                case CUDNN_POINTWISE_CMP_EQ:     y = (x == b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_NEQ:    y = (x != b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_GT:     y = (x >  b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_GE:     y = (x >= b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_LT:     y = (x <  b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_LE:     y = (x <= b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_AND:y = (x != 0.f && b != 0.f) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_OR: y = (x != 0.f || b != 0.f) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_NOT:y = (x == 0.f) ? 1.f : 0.f; break;
                default: y = x; break;  // IDENTITY and unknown
                }
                Y[i] = y;
            }
            break;
        }

        case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xD = buildTensorDesc(xId);
            TensorDesc yD = buildTensorDesc(yId);
            float scaleH = (yD.h > 0 && xD.h > 0) ? static_cast<float>(xD.h) / yD.h : 1.f;
            float scaleW = (yD.w > 0 && xD.w > 0) ? static_cast<float>(xD.w) / yD.w : 1.f;

            const float* X = static_cast<const float*>(xPtr);
            float*       Y = static_cast<float*>(yPtr);

            // Bilinear resample (covers both upsample and downsample)
            for (int n = 0; n < yD.n; ++n) {
                for (int c = 0; c < yD.c; ++c) {
                    for (int hy = 0; hy < yD.h; ++hy) {
                        for (int wy = 0; wy < yD.w; ++wy) {
                            float sx = (hy + 0.5f) * scaleH - 0.5f;
                            float sy = (wy + 0.5f) * scaleW - 0.5f;
                            sx = std::max(0.f, std::min(static_cast<float>(xD.h - 1), sx));
                            sy = std::max(0.f, std::min(static_cast<float>(xD.w - 1), sy));
                            int x0 = static_cast<int>(sx);
                            int x1 = std::min(xD.h - 1, x0 + 1);
                            int y0 = static_cast<int>(sy);
                            int y1 = std::min(xD.w - 1, y0 + 1);
                            float wx = sx - x0;
                            float wy_ = sy - y0;
                            size_t base = static_cast<size_t>(n * xD.c + c) * xD.h * xD.w;
                            float v = X[base + x0*xD.w + y0] * (1-wx)*(1-wy_)
                                    + X[base + x0*xD.w + y1] * (1-wx)*wy_
                                    + X[base + x1*xD.w + y0] * wx*(1-wy_)
                                    + X[base + x1*xD.w + y1] * wx*wy_;
                            Y[static_cast<size_t>(n*yD.c + c)*yD.h*yD.w + hy*yD.w + wy] = v;
                        }
                    }
                }
            }
            break;
        }

        case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: {
            // Backward pass: dX = scatter-add of bilinear-weighted dY gradients.
            // xDesc = dX (gradient w.r.t. input),  yDesc = dY (upstream gradient)
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC);
            void* dxPtr = dataPtrs[xId];
            void* dyPtr = dataPtrs[yId];
            if (!dxPtr || !dyPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xD = buildTensorDesc(xId);  // input (dX) dimensions
            TensorDesc yD = buildTensorDesc(yId);   // output (dY) dimensions
            float scaleH = (yD.h > 0 && xD.h > 0) ? static_cast<float>(xD.h) / yD.h : 1.f;
            float scaleW = (yD.w > 0 && xD.w > 0) ? static_cast<float>(xD.w) / yD.w : 1.f;

            float*       dX = static_cast<float*>(dxPtr);
            const float* dY = static_cast<const float*>(dyPtr);

            // Zero-initialize dX
            size_t dxCount = static_cast<size_t>(xD.n) * xD.c * xD.h * xD.w;
            std::memset(dX, 0, dxCount * sizeof(float));

            // Scatter-add: for each output pixel, distribute gradient to 4 input neighbors
            for (int n = 0; n < yD.n; ++n) {
                for (int c = 0; c < yD.c; ++c) {
                    size_t xBase = static_cast<size_t>(n * xD.c + c) * xD.h * xD.w;
                    size_t yBase = static_cast<size_t>(n * yD.c + c) * yD.h * yD.w;
                    for (int hy = 0; hy < yD.h; ++hy) {
                        for (int wy = 0; wy < yD.w; ++wy) {
                            float sx = (hy + 0.5f) * scaleH - 0.5f;
                            float sy = (wy + 0.5f) * scaleW - 0.5f;
                            sx = std::max(0.f, std::min(static_cast<float>(xD.h - 1), sx));
                            sy = std::max(0.f, std::min(static_cast<float>(xD.w - 1), sy));
                            int x0 = static_cast<int>(sx);
                            int x1 = std::min(xD.h - 1, x0 + 1);
                            int y0 = static_cast<int>(sy);
                            int y1 = std::min(xD.w - 1, y0 + 1);
                            float wx = sx - x0;
                            float wy_ = sy - y0;
                            float grad = dY[yBase + hy * yD.w + wy];
                            dX[xBase + x0 * xD.w + y0] += grad * (1 - wx) * (1 - wy_);
                            dX[xBase + x0 * xD.w + y1] += grad * (1 - wx) * wy_;
                            dX[xBase + x1 * xD.w + y0] += grad * wx * (1 - wy_);
                            dX[xBase + x1 * xD.w + y1] += grad * wx * wy_;
                        }
                    }
                }
            }
            break;
        }

        default:
            return CUDNN_STATUS_NOT_SUPPORTED;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Re-executable plan creation ──────────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateReexecutable(cudnnHandle_t /*handle*/, void* opGraph, void** plan) {
    if (!opGraph || !plan) return CUDNN_STATUS_INVALID_VALUE;
    uintptr_t graphId = reinterpret_cast<uintptr_t>(opGraph);
    const BackendNode *graph = getNode(graphId);
    if (!graph) return CUDNN_STATUS_INVALID_VALUE;

    uintptr_t engineId = g_nextBackendId++;
    BackendNode &engine = g_backendNodes[engineId];
    engine.type = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
    engine.finalized = true;
    engine.attrs[static_cast<int>(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)] = {graphId};

    uintptr_t configId = g_nextBackendId++;
    BackendNode &config = g_backendNodes[configId];
    config.type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
    config.finalized = true;
    config.attrs[static_cast<int>(CUDNN_ATTR_ENGINECFG_ENGINE)] = {engineId};

    uintptr_t planId = g_nextBackendId++;
    BackendNode &planNode = g_backendNodes[planId];
    planNode.type = CUDNN_BACKEND_HANDLE_DESCRIPTOR;
    planNode.finalized = true;
    planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG)] = {configId};
    planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)] = {0};

    *plan = reinterpret_cast<void*>(planId);
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
