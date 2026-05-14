/**
 * VGRE CUDART Shim — Graph Exec Mutation & Instantiation (P1.18)
 *
 * Template graph node param setters:
 *   cudaGraphKernelNodeSetParams, cudaGraphMemsetNodeSetParams,
 *   cudaGraphHostNodeSetParams
 *
 * Executable graph (exec) mutation — operates on the deep-cloned working copy
 * that GraphManager::instantiate() creates so the template graph is unchanged:
 *   cudaGraphExecKernelNodeSetParams, cudaGraphExecMemcpyNodeSetParams,
 *   cudaGraphExecMemsetNodeSetParams, cudaGraphExecHostNodeSetParams,
 *   cudaGraphExecChildGraphNodeSetParams,
 *   cudaGraphExecEventRecordNodeSetEvent, cudaGraphExecEventWaitNodeSetEvent
 *
 * Exec-level state:
 *   cudaGraphNodeSetEnabled / GetEnabled, cudaGraphExecGetFlags
 *   cudaGraphExecNodeSetParams (generic type-dispatch)
 *
 * Instantiation and upload:
 *   cudaGraphInstantiateWithFlags, cudaGraphUpload
 */

#include "cudart_graph_internal.h"

extern "C" {

// ── Template-graph node param setters ────────────────────────────────────────

cudaError_t cudaGraphKernelNodeSetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        const cudaKernelNodeParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->func) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::string kname = vgre_lookup_kernel_name(pNodeParams->func);
    std::string ksrc  = vgre_lookup_kernel_source(pNodeParams->func);
    if (kname.empty()) return cudaErrorInvalidDeviceFunction;

    vgre::KernelId kid = 0;
    RE().registerKernel(kname, ksrc, kid);
    std::vector<vgre::ArgType> argTypes;
    RE().getKernelArgTypes(kid, argTypes);

    return RE().graphKernelNodeSetParams(
        static_cast<vgre::GraphId>(graph), node,
        pNodeParams->kernelParams, argTypes)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphMemsetNodeSetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        const cudaMemsetParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->dst) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphMemsetNodeSetParams(
        static_cast<vgre::GraphId>(graph), node,
        pNodeParams->dst,
        static_cast<int>(pNodeParams->value & 0xFF),
        pNodeParams->pitch, pNodeParams->width, pNodeParams->height, 1)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphHostNodeSetParams(
        cudaGraphNode_t node, cudaGraph_t graph,
        const cudaHostNodeParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->fn) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphHostNodeSetParams(
        static_cast<vgre::GraphId>(graph), node,
        pNodeParams->fn, pNodeParams->userData)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// ── Exec-level kernel mutation ────────────────────────────────────────────────

cudaError_t cudaGraphExecKernelNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const cudaKernelNodeParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->func) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    std::string kname = vgre_lookup_kernel_name(pNodeParams->func);
    std::string ksrc  = vgre_lookup_kernel_source(pNodeParams->func);
    if (kname.empty()) return cudaErrorInvalidDeviceFunction;

    vgre::KernelId kid = 0;
    RE().registerKernel(kname, ksrc, kid);
    std::vector<vgre::ArgType> argTypes;
    RE().getKernelArgTypes(kid, argTypes);

    return RE().graphExecKernelNodeSetParams(
        static_cast<vgre::GraphExecId>(graphExec), node,
        pNodeParams->kernelParams, argTypes)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const cudaMemcpy3DParms *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    size_t count = pNodeParams->extent.width
                   * std::max<size_t>(1, pNodeParams->extent.height)
                   * std::max<size_t>(1, pNodeParams->extent.depth);
    return RE().graphExecMemcpyNodeSetParams(
        static_cast<vgre::GraphExecId>(graphExec), node,
        pNodeParams->dstPtr.ptr,
        pNodeParams->srcPtr.ptr,
        count,
        static_cast<int>(pNodeParams->kind))
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecMemsetNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const cudaMemsetParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->dst) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphExecMemsetNodeSetParams(
        static_cast<vgre::GraphExecId>(graphExec), node,
        pNodeParams->dst,
        static_cast<int>(pNodeParams->value & 0xFF),
        pNodeParams->pitch, pNodeParams->width, pNodeParams->height, 1)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecHostNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const cudaHostNodeParams *pNodeParams) {
    if (!pNodeParams || !pNodeParams->fn) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphExecHostNodeSetParams(
        static_cast<vgre::GraphExecId>(graphExec), node,
        pNodeParams->fn, pNodeParams->userData)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        cudaGraph_t childGraph) {
    if (graphExec == 0 || childGraph == 0) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphExecChildGraphNodeSetParams(
        static_cast<vgre::GraphExecId>(graphExec), node,
        static_cast<vgre::GraphId>(childGraph))
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(
        cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphExecEventRecordNodeSetEvent(
        static_cast<vgre::GraphExecId>(graphExec), node,
        static_cast<void *>(event))
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(
        cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphExecEventWaitNodeSetEvent(
        static_cast<vgre::GraphExecId>(graphExec), node,
        static_cast<void *>(event))
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// ── Exec node enable/disable and flags ───────────────────────────────────────

cudaError_t cudaGraphNodeSetEnabled(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        unsigned int isEnabled) {
    if (!RE().isInitialized()) return cudaErrorNotInitialized;
    return RE().graphNodeSetEnabled(
        static_cast<vgre::GraphExecId>(graphExec), node, isEnabled != 0)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphNodeGetEnabled(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        unsigned int *isEnabled) {
    if (!isEnabled) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    bool en = true;
    auto r = RE().graphNodeGetEnabled(
        static_cast<vgre::GraphExecId>(graphExec), node, en);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
    *isEnabled = en ? 1u : 0u;
    return cudaSuccess;
}

cudaError_t cudaGraphExecGetFlags(
        cudaGraphExec_t graphExec, unsigned int *flags) {
    if (!flags) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;
    return RE().graphExecGetFlags(
        static_cast<vgre::GraphExecId>(graphExec), *flags)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// ── cudaGraphInstantiateWithFlags ─────────────────────────────────────────────

cudaError_t cudaGraphInstantiateWithFlags(
        cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
        unsigned long long flags) {
    if (!pGraphExec) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    vgre::GraphExecId execId = 0;
    auto r = RE().graphInstantiate(static_cast<vgre::GraphId>(graph), execId);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    RE().graphExecSetFlags(execId, static_cast<unsigned int>(flags));
    *pGraphExec = static_cast<cudaGraphExec_t>(execId);
    return cudaSuccess;
}

// cudaGraphUpload: no DMA transfer needed in CPU emulation — graph is already
// resident in host memory.  Return SUCCESS so call sequences that upload before
// launch work unmodified.
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    (void)graphExec; (void)stream;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;
    return cudaSuccess;
}

// ── Missing node param structs (defined here for generic dispatcher) ─────────
struct cudaEventRecordNodeParams { cudaEvent_t event; };
struct cudaEventWaitNodeParams  { cudaEvent_t event; };

// Forward declarations for external semaphore node params (defined in cudart_shim.cpp)
struct cudaExternalSemaphoreSignalNodeParams;
struct cudaExternalSemaphoreWaitNodeParams;
extern "C" cudaError_t vgreGraphExtSemGetSignalNodeParams(
        uint64_t node, cudaExternalSemaphoreSignalNodeParams *params_out);
extern "C" cudaError_t vgreGraphExtSemGetWaitNodeParams(
        uint64_t node, cudaExternalSemaphoreWaitNodeParams *params_out);

// ── cudaGraphExecNodeSetParams (generic dispatcher) ───────────────────────────

cudaError_t cudaGraphExecNodeSetParams(
        cudaGraphExec_t graphExec, cudaGraphNode_t node,
        const void *pNodeParams) {
    if (!pNodeParams) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    vgre::core::GraphNodeType nodeType;
    auto r = RE().graphExecGetNodeType(
        static_cast<vgre::GraphExecId>(graphExec), node, nodeType);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    switch (nodeType) {
    case vgre::core::GraphNodeType::KERNEL:
        return cudaGraphExecKernelNodeSetParams(
            graphExec, node,
            static_cast<const cudaKernelNodeParams *>(pNodeParams));
    case vgre::core::GraphNodeType::MEMCPY:
        return cudaGraphExecMemcpyNodeSetParams(
            graphExec, node,
            static_cast<const cudaMemcpy3DParms *>(pNodeParams));
    case vgre::core::GraphNodeType::MEMSET:
        return cudaGraphExecMemsetNodeSetParams(
            graphExec, node,
            static_cast<const cudaMemsetParams *>(pNodeParams));
    case vgre::core::GraphNodeType::HOST:
        return cudaGraphExecHostNodeSetParams(
            graphExec, node,
            static_cast<const cudaHostNodeParams *>(pNodeParams));
    case vgre::core::GraphNodeType::CHILD:
        return cudaGraphExecChildGraphNodeSetParams(
            graphExec, node,
            *static_cast<const cudaGraph_t *>(pNodeParams));
    case vgre::core::GraphNodeType::EVENT_RECORD: {
        auto *pr = static_cast<const cudaEventRecordNodeParams *>(pNodeParams);
        return cudaGraphExecEventRecordNodeSetEvent(graphExec, node, pr->event);
    }
    case vgre::core::GraphNodeType::EVENT_WAIT: {
        auto *pw = static_cast<const cudaEventWaitNodeParams *>(pNodeParams);
        return cudaGraphExecEventWaitNodeSetEvent(graphExec, node, pw->event);
    }
    case vgre::core::GraphNodeType::EMPTY:
        return cudaSuccess;
    default:
        return cudaErrorNotSupported;
    }
}

} // extern "C"
