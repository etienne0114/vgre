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
    CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 132
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

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t /*handle*/, void* /*plan*/, void* /*variantPack*/) {
    // In a full implementation this would:
    // 1. Walk the operation graph from the plan
    // 2. Map each operation to existing legacy cuDNN shim calls
    // 3. Execute in topological order
    //
    // For the minimal stub, we return NOT_SUPPORTED because we don't yet
    // wire the descriptor graph to the legacy paths.  This is a deliberate
    // choice: the Backend API is huge; full wiring requires substantial
    // descriptor-graph traversal and mapping code.
    return CUDNN_STATUS_NOT_SUPPORTED;
}

// ── Engine / Plan / Heuristics stubs ─────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateReexecutable(cudnnHandle_t /*handle*/, void* /*opGraph*/, void** /*plan*/) {
    // Re-executable plans are not supported in CPU emulation
    return CUDNN_STATUS_NOT_SUPPORTED;
}

} // extern "C"
