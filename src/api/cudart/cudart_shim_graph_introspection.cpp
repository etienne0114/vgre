/**
 * VGRE CUDART Shim — Graph Introspection (P1.17)
 *
 * Read-only topology and parameter queries:
 *   cudaGraphGetNodes, cudaGraphGetRootNodes, cudaGraphGetEdges
 *   cudaGraphNodeGetType, cudaGraphNodeGetDependencies,
 *   cudaGraphNodeGetDependentNodes
 *   cudaGraphKernelNodeGetParams, cudaGraphMemsetNodeGetParams,
 *   cudaGraphHostNodeGetParams, cudaGraphMemcpyNodeGetParams
 *   cudaGraphNodeFindInClone, cudaGraphDebugDotPrint
 *   cudaGraphKernelNodeCopyAttributes (no-op in CPU model)
 *
 * All queries return copies — callers never hold references into live graph data.
 */

#include "cudart_graph_internal.h"

// Exposed from cudart_shim.cpp
extern "C" const void *vgre_lookup_host_function_by_name(const char *name);

extern "C" {

// ── Graph topology queries ────────────────────────────────────────────────────

cudaError_t cudaGraphGetNodes(
        cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
    if (!numNodes) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<vgre::core::GraphNode> raw;
    auto r = RE().graphGetNodes(static_cast<vgre::GraphId>(graph), raw);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    if (nodes && *numNodes >= raw.size())
        for (size_t i = 0; i < raw.size(); ++i)
            nodes[i] = raw[i].nodeId;
    *numNodes = raw.size();
    return cudaSuccess;
}

cudaError_t cudaGraphGetRootNodes(
        cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes) {
    if (!pNumRootNodes) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> roots;
    auto r = RE().graphGetRootNodes(static_cast<vgre::GraphId>(graph), roots);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    if (pRootNodes && *pNumRootNodes >= roots.size())
        for (size_t i = 0; i < roots.size(); ++i)
            pRootNodes[i] = roots[i];
    *pNumRootNodes = roots.size();
    return cudaSuccess;
}

cudaError_t cudaGraphGetEdges(
        cudaGraph_t graph,
        cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges) {
    if (!numEdges) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> froms, tos;
    size_t count = 0;
    auto r = RE().graphGetEdges(
        static_cast<vgre::GraphId>(graph), froms, tos, count);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    if (from && to && *numEdges >= count)
        for (size_t i = 0; i < count; ++i) { from[i] = froms[i]; to[i] = tos[i]; }
    *numEdges = count;
    return cudaSuccess;
}

// ── Node topology queries ─────────────────────────────────────────────────────

cudaError_t cudaGraphNodeGetType(
        cudaGraphNode_t node, cudaGraph_t graph, cudaGraphNodeType *pType) {
    if (!pType) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    vgre::core::GraphNodeType t;
    auto r = RE().graphNodeGetType(static_cast<vgre::GraphId>(graph), node, t);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    *pType = toPublicNodeType(t);
    return cudaSuccess;
}

cudaError_t cudaGraphNodeGetDependencies(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaGraphNode_t *pDependencies, size_t *pNumDependencies) {
    if (!pNumDependencies) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    size_t count = 0;
    auto r = RE().graphNodeGetDependencies(
        static_cast<vgre::GraphId>(graph), node, deps, count);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    if (pDependencies && *pNumDependencies >= count)
        for (size_t i = 0; i < count; ++i)
            pDependencies[i] = deps[i];
    *pNumDependencies = count;
    return cudaSuccess;
}

cudaError_t cudaGraphNodeGetDependentNodes(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes) {
    if (!pNumDependentNodes) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> dependents;
    size_t count = 0;
    auto r = RE().graphNodeGetDependentNodes(
        static_cast<vgre::GraphId>(graph), node, dependents, count);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    if (pDependentNodes && *pNumDependentNodes >= count)
        for (size_t i = 0; i < count; ++i)
            pDependentNodes[i] = dependents[i];
    *pNumDependentNodes = count;
    return cudaSuccess;
}

// ── Node parameter introspection ──────────────────────────────────────────────

cudaError_t cudaGraphKernelNodeGetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaKernelNodeParams *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    vgre::KernelId kid = 0;
    std::string name;
    vgre::dim3 grid, block;
    std::vector<std::vector<uint8_t>> args;
    auto r = RE().graphKernelNodeGetParams(
        static_cast<vgre::GraphId>(graph), node, kid, name, grid, block, args);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    // Reconstruct func pointer from kernel name via reverse registry mapping.
    pNodeParams->func           = const_cast<void*>(vgre_lookup_host_function_by_name(name.c_str()));
    pNodeParams->gridDim.x      = grid.x;
    pNodeParams->gridDim.y      = grid.y;
    pNodeParams->gridDim.z      = grid.z;
    pNodeParams->blockDim.x     = block.x;
    pNodeParams->blockDim.y     = block.y;
    pNodeParams->blockDim.z     = block.z;
    pNodeParams->sharedMemBytes = 0;
    pNodeParams->kernelParams   = nullptr;
    pNodeParams->extra          = nullptr;
    return cudaSuccess;
}

cudaError_t cudaGraphMemsetNodeGetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaMemsetParams *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    void *dst = nullptr; int val = 0;
    size_t pitch = 0, width = 0, height = 0, depth = 0;
    auto r = RE().graphMemsetNodeGetParams(
        static_cast<vgre::GraphId>(graph), node, dst, val, pitch, width, height, depth);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    pNodeParams->dst    = dst;
    pNodeParams->pitch  = pitch;
    pNodeParams->value  = static_cast<unsigned int>(val & 0xFF);
    pNodeParams->width  = width;
    pNodeParams->height = height;
    return cudaSuccess;
}

cudaError_t cudaGraphHostNodeGetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaHostNodeParams *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    void (*fn)(void *) = nullptr;
    void *ud = nullptr;
    auto r = RE().graphHostNodeGetParams(
        static_cast<vgre::GraphId>(graph), node, &fn, ud);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    pNodeParams->fn       = fn;
    pNodeParams->userData = ud;
    return cudaSuccess;
}

cudaError_t cudaGraphMemcpyNodeGetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        cudaMemcpy3DParms *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    void *dst = nullptr, *src = nullptr;
    size_t count = 0; int kind = 0;
    auto r = RE().graphMemcpyNodeGetParams(
        static_cast<vgre::GraphId>(graph), node, dst, src, count, kind);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    std::memset(pNodeParams, 0, sizeof(*pNodeParams));
    pNodeParams->dstPtr.ptr   = dst;
    pNodeParams->srcPtr.ptr   = src;
    pNodeParams->extent.width = count;
    pNodeParams->extent.height = 1;
    pNodeParams->extent.depth  = 1;
    pNodeParams->kind          = static_cast<cudaMemcpyKind_t>(kind);
    return cudaSuccess;
}

// ── Clone and debug utilities ─────────────────────────────────────────────────

cudaError_t cudaGraphNodeFindInClone(
        cudaGraphNode_t *pNode, cudaGraphNode_t originalNode,
        cudaGraph_t clonedGraph) {
    if (!pNode) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    uint64_t clonedNodeId = 0;
    auto r = RE().graphNodeFindInClone(
        0, originalNode, static_cast<vgre::GraphId>(clonedGraph), clonedNodeId);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
    *pNode = clonedNodeId;
    return cudaSuccess;
}

cudaError_t cudaGraphDebugDotPrint(
        cudaGraph_t graph, const char *path, unsigned int flags) {
    if (!RE().isInitialized()) return cudaErrorNotInitialized;
    return RE().graphDebugDotPrint(
        static_cast<vgre::GraphId>(graph), path, flags)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// Kernel-node attributes are hardware-specific; the CPU model has no equivalent.
cudaError_t cudaGraphKernelNodeCopyAttributes(
        cudaGraphNode_t dst, cudaGraph_t dstGraph,
        cudaGraphNode_t src, cudaGraph_t srcGraph) {
    (void)dst; (void)dstGraph; (void)src; (void)srcGraph;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;
    return cudaSuccess;
}

} // extern "C"
