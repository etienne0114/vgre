// cuDNN v9 Graph API — higher-level builder on top of backend descriptor infrastructure.
// Internally reuses g_backendNodes / g_nextBackendId from cudnn_backend_api.cpp.

#include "cudnn_backend_internal.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

// ── Graph object ──────────────────────────────────────────────────────────────

struct CudnnGraph {
    std::vector<uintptr_t> opIds;   // backend op descriptor IDs in topo order
    bool built = false;
};

// ── Forward declarations of functions defined in cudnn_backend_api.cpp ────────
// (needed for inline dispatch in cudnnGraphExecute)
extern "C" {
    cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, void* plan, void* variantPack);
}

// ── Public Graph API ──────────────────────────────────────────────────────────

extern "C" {

cudnnStatus_t cudnnGraphCreate(CudnnGraph** graph, cudnnHandle_t /*handle*/) {
    if (!graph) return CUDNN_STATUS_INVALID_VALUE;
    *graph = new CudnnGraph();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGraphDestroy(CudnnGraph* graph) {
    delete graph;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGraphAddNode(CudnnGraph* graph, void* opDescriptor) {
    if (!graph || !opDescriptor) return CUDNN_STATUS_INVALID_VALUE;
    uintptr_t id = reinterpret_cast<uintptr_t>(opDescriptor);
    graph->opIds.push_back(id);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGraphBuildAndCheck(CudnnGraph* graph, cudnnHandle_t /*handle*/) {
    if (!graph) return CUDNN_STATUS_INVALID_VALUE;
    // Validate that each stored opId exists in g_backendNodes
    for (uintptr_t id : graph->opIds) {
        if (g_backendNodes.find(id) == g_backendNodes.end())
            return CUDNN_STATUS_INVALID_VALUE;
    }
    graph->built = true;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGraphExecute(cudnnHandle_t handle, CudnnGraph* graph, void* variantPack) {
    if (!graph || !graph->built) return CUDNN_STATUS_INVALID_VALUE;
    if (graph->opIds.empty()) return CUDNN_STATUS_SUCCESS;

    // Create a temporary OPERATIONSET node referencing graph->opIds,
    // then build a minimal plan pointing to it, and call cudnnBackendExecute.

    // 1. Create operation-set node
    uintptr_t opSetId = g_nextBackendId++;
    {
        BackendNode& opSetNode = g_backendNodes[opSetId];
        opSetNode.type = CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR;
        opSetNode.finalized = true;
        std::vector<uint64_t>& ops = opSetNode.attrs[static_cast<int>(CUDNN_ATTR_OPERATIONSET_OPS)];
        ops.reserve(graph->opIds.size());
        for (uintptr_t id : graph->opIds)
            ops.push_back(static_cast<uint64_t>(id));
    }

    // 2. Create engine node pointing to opSet
    uintptr_t engineId = g_nextBackendId++;
    {
        BackendNode& engNode = g_backendNodes[engineId];
        engNode.type = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
        engNode.finalized = true;
        engNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)]
            .push_back(static_cast<uint64_t>(opSetId));
    }

    // 3. Create engine-config node
    uintptr_t cfgId = g_nextBackendId++;
    {
        BackendNode& cfgNode = g_backendNodes[cfgId];
        cfgNode.type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
        cfgNode.finalized = true;
        cfgNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINECFG_ENGINE)]
            .push_back(static_cast<uint64_t>(engineId));
    }

    // 4. Create plan node
    uintptr_t planId = g_nextBackendId++;
    {
        BackendNode& planNode = g_backendNodes[planId];
        planNode.type = CUDNN_BACKEND_HANDLE_DESCRIPTOR;
        planNode.finalized = true;
        planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG)]
            .push_back(static_cast<uint64_t>(cfgId));
        planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)]
            .push_back(0ULL);
    }

    // 5. Execute via the existing backend execute path
    cudnnStatus_t status = cudnnBackendExecute(
        handle,
        reinterpret_cast<void*>(planId),
        variantPack);

    // 6. Clean up temporary nodes (plan, config, engine, opset)
    g_backendNodes.erase(planId);
    g_backendNodes.erase(cfgId);
    g_backendNodes.erase(engineId);
    g_backendNodes.erase(opSetId);

    return status;
}

cudnnStatus_t cudnnGraphGetAttribute(CudnnGraph* graph, int attr, void* value, size_t* size) {
    if (!graph || !value || !size) return CUDNN_STATUS_INVALID_VALUE;
    switch (attr) {
    case 0: {
        // attr=0: return number of ops as int64_t
        int64_t count = static_cast<int64_t>(graph->opIds.size());
        memcpy(value, &count, sizeof(int64_t));
        *size = sizeof(int64_t);
        return CUDNN_STATUS_SUCCESS;
    }
    case 1: {
        // attr=1: return built status as int32_t
        int32_t built = graph->built ? 1 : 0;
        memcpy(value, &built, sizeof(int32_t));
        *size = sizeof(int32_t);
        return CUDNN_STATUS_SUCCESS;
    }
    default:
        return CUDNN_STATUS_INVALID_VALUE;
    }
}

} // extern "C"
