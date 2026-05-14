/**
 * VGRE CUDART Shim — Graph Node Creation (P1.10 – P1.16)
 *
 * P1.10  cudaGraphAddKernelNode        — resolves host stub → KernelId
 * P1.11  cudaGraphAddMemsetNode        — pitched 2D fill node
 * P1.12  cudaGraphAddHostNode          — CPU callback node
 * P1.13  cudaGraphAddChildGraphNode    — nested sub-graph node
 * P1.14  cudaGraphAddEmptyNode         — dependency placeholder
 * P1.15  cudaGraphAddEventRecordNode / EventWaitNode
 * P1.16  cudaGraphAddMemAllocNode / MemFreeNode
 *
 * All node creation is delegated to RuntimeEngine which holds the sole
 * authoritative reference to GraphManager.  The kernel lookup path uses
 * vgre_lookup_kernel_name/source (exported from cudart_shim.cpp, same .so).
 */

#include "cudart_graph_internal.h"

extern "C" {

// ── P1.10: cudaGraphAddKernelNode ─────────────────────────────────────────────

cudaError_t cudaGraphAddKernelNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        const cudaKernelNodeParams *pNodeParams) {
    if (!pGraphNode || !pNodeParams || !pNodeParams->func)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::string kname = vgre_lookup_kernel_name(pNodeParams->func);
    std::string ksrc  = vgre_lookup_kernel_source(pNodeParams->func);
    if (kname.empty() || ksrc.empty()) {
        VGRE_LOG_ERROR("CUDART", "cudaGraphAddKernelNode: unknown host function");
        return cudaErrorInvalidDeviceFunction;
    }

    vgre::KernelId kid = 0;
    if (RE().registerKernel(kname, ksrc, kid) != vgre::VGREResult::SUCCESS || kid == 0)
        return cudaErrorLaunchFailure;

    std::vector<vgre::ArgType> argTypes;
    RE().getKernelArgTypes(kid, argTypes);

    vgre::dim3 grid(pNodeParams->gridDim.x, pNodeParams->gridDim.y, pNodeParams->gridDim.z);
    vgre::dim3 block(pNodeParams->blockDim.x, pNodeParams->blockDim.y, pNodeParams->blockDim.z);
    if (grid.x == 0)  grid.x  = 1;
    if (block.x == 0) block.x = 1;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t outNodeId = 0;
    auto r = RE().graphAddKernelNode(
        static_cast<vgre::GraphId>(graph), kid, kname,
        grid, block, pNodeParams->kernelParams, argTypes, deps, outNodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = outNodeId;
    return cudaSuccess;
}

// ── P1.11: cudaGraphAddMemsetNode ─────────────────────────────────────────────

cudaError_t cudaGraphAddMemsetNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        const cudaMemsetParams *pMemsetParams) {
    if (!pGraphNode || !pMemsetParams || !pMemsetParams->dst)
        return cudaErrorInvalidValue;
    if (pMemsetParams->width == 0 || pMemsetParams->height == 0 ||
        pMemsetParams->pitch < pMemsetParams->width)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddMemsetNode(
        static_cast<vgre::GraphId>(graph),
        pMemsetParams->dst,
        static_cast<int>(pMemsetParams->value & 0xFF),
        pMemsetParams->pitch,
        pMemsetParams->width,
        pMemsetParams->height,
        1, deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

// ── P1.12: cudaGraphAddHostNode ───────────────────────────────────────────────

cudaError_t cudaGraphAddHostNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        const cudaHostNodeParams *pNodeParams) {
    if (!pGraphNode || !pNodeParams || !pNodeParams->fn)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddHostNode(
        static_cast<vgre::GraphId>(graph),
        pNodeParams->fn, pNodeParams->userData, deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

// ── P1.13: cudaGraphAddChildGraphNode ─────────────────────────────────────────

cudaError_t cudaGraphAddChildGraphNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        cudaGraph_t childGraph) {
    if (!pGraphNode || graph == 0 || childGraph == 0)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddChildGraphNode(
        static_cast<vgre::GraphId>(graph),
        static_cast<vgre::GraphId>(childGraph), deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

// ── P1.14: cudaGraphAddEmptyNode ──────────────────────────────────────────────

cudaError_t cudaGraphAddEmptyNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies) {
    if (!pGraphNode || graph == 0)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddEmptyNode(
        static_cast<vgre::GraphId>(graph), deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

// ── P1.15: cudaGraphAddEventRecordNode / EventWaitNode ────────────────────────

cudaError_t cudaGraphAddEventRecordNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        cudaEvent_t event) {
    if (!pGraphNode || graph == 0 || !event)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddEventRecordNode(
        static_cast<vgre::GraphId>(graph), static_cast<void *>(event), deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

cudaError_t cudaGraphAddEventWaitNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        cudaEvent_t event) {
    if (!pGraphNode || graph == 0 || !event)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddEventWaitNode(
        static_cast<vgre::GraphId>(graph), static_cast<void *>(event), deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

// ── P1.16: cudaGraphAddMemAllocNode / MemFreeNode ─────────────────────────────

cudaError_t cudaGraphAddMemAllocNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        cudaMemAllocNodeParams *pNodeParams) {
    if (!pGraphNode || graph == 0 || !pNodeParams || pNodeParams->bytesize == 0)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddMemAllocNode(
        static_cast<vgre::GraphId>(graph),
        pNodeParams->bytesize, &pNodeParams->dptr, deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return (r == vgre::VGREResult::ERR_OUT_OF_MEMORY)
               ? cudaErrorMemoryAllocation : cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

cudaError_t cudaGraphAddMemFreeNode(
        cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
        const cudaGraphNode_t *pDependencies, size_t numDependencies,
        void *dptr) {
    if (!pGraphNode || graph == 0 || !dptr)
        return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::vector<uint64_t> deps;
    if (pDependencies && numDependencies > 0)
        deps.assign(pDependencies, pDependencies + numDependencies);

    uint64_t nodeId = 0;
    auto r = RE().graphAddMemFreeNode(
        static_cast<vgre::GraphId>(graph), dptr, deps, nodeId);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *pGraphNode = nodeId;
    return cudaSuccess;
}

} // extern "C"
