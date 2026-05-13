/**
 * VGRE RuntimeEngine — Graph Extended APIs
 *
 * Delegation layer: all new graph methods on RuntimeEngine call the
 * corresponding GraphManager method under the engine's recursive mutex.
 * This keeps RuntimeEngine thin and GraphManager the single authoritative
 * source of graph state.
 */

#include "vgre/core/runtime_engine.h"
#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"

#include <vector>

namespace vgre {
namespace core {

// ── Guard macro ───────────────────────────────────────────────────────────────
#define GRAPH_GUARD()                                        \
    std::lock_guard<std::recursive_mutex> _lock_(mutex_);    \
    if (!initialized_ || !graphManager_)                     \
        return VGREResult::ERR_NOT_INITIALIZED;

// ── New node types ────────────────────────────────────────────────────────────

VGREResult RuntimeEngine::graphAddMemsetNode(GraphId graph, void *dst,
                                              int value, size_t pitch,
                                              size_t width, size_t height,
                                              size_t depth,
                                              const std::vector<uint64_t> &deps,
                                              uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addMemsetNode(
        graph, dst, value, pitch, width, height, depth, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddHostNode(GraphId graph,
                                            void (*fn)(void *), void *userData,
                                            const std::vector<uint64_t> &deps,
                                            uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addHostNode(graph, fn, userData, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddChildGraphNode(GraphId graph,
                                                  GraphId childGraph,
                                                  const std::vector<uint64_t> &deps,
                                                  uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addChildGraphNode(graph, childGraph, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddEmptyNode(GraphId graph,
                                             const std::vector<uint64_t> &deps,
                                             uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addEmptyNode(graph, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddEventRecordNode(
        GraphId graph, void *event, const std::vector<uint64_t> &deps,
        uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addEventRecordNode(graph, event, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddEventWaitNode(
        GraphId graph, void *event, const std::vector<uint64_t> &deps,
        uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addEventWaitNode(graph, event, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddMemAllocNode(GraphId graph, size_t bytesize,
                                                void **outDevPtr,
                                                const std::vector<uint64_t> &deps,
                                                uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addMemAllocNode(
        graph, bytesize, outDevPtr, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddMemFreeNode(
        GraphId graph, void *devPtr, const std::vector<uint64_t> &deps,
        uint64_t &outNodeId) {
    GRAPH_GUARD();
    return graphManager_->addMemFreeNode(graph, devPtr, deps, outNodeId);
}

// ── Introspection ─────────────────────────────────────────────────────────────

VGREResult RuntimeEngine::graphGetNodes(GraphId graph,
                                         std::vector<GraphNode> &outNodes) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getGraphNodes(graph, outNodes);
}

VGREResult RuntimeEngine::graphGetRootNodes(
        GraphId graph, std::vector<uint64_t> &outRoots) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getGraphRootNodes(graph, outRoots);
}

VGREResult RuntimeEngine::graphGetEdges(GraphId graph,
                                         std::vector<uint64_t> &fromNodes,
                                         std::vector<uint64_t> &toNodes,
                                         size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getGraphEdges(graph, fromNodes, toNodes, outCount);
}

VGREResult RuntimeEngine::graphNodeGetType(GraphId graph, uint64_t nodeId,
                                            GraphNodeType &outType) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getNodeType(graph, nodeId, outType);
}

VGREResult RuntimeEngine::graphNodeGetDependencies(
        GraphId graph, uint64_t nodeId,
        std::vector<uint64_t> &outDeps, size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getNodeDependencies(graph, nodeId, outDeps, outCount);
}

VGREResult RuntimeEngine::graphNodeGetDependentNodes(
        GraphId graph, uint64_t nodeId,
        std::vector<uint64_t> &outDependents, size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getNodeDependentNodes(
        graph, nodeId, outDependents, outCount);
}

VGREResult RuntimeEngine::graphKernelNodeGetParams(
        GraphId graph, uint64_t nodeId,
        KernelId &outKid, std::string &outName,
        dim3 &outGrid, dim3 &outBlock,
        std::vector<std::vector<uint8_t>> &outArgs) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getKernelNodeParams(
        graph, nodeId, outKid, outName, outGrid, outBlock, outArgs);
}

VGREResult RuntimeEngine::graphMemcpyNodeGetParams(
        GraphId graph, uint64_t nodeId,
        void *&outDst, void *&outSrc,
        size_t &outCount, int &outKind) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getMemcpyNodeParams(
        graph, nodeId, outDst, outSrc, outCount, outKind);
}

VGREResult RuntimeEngine::graphMemsetNodeGetParams(
        GraphId graph, uint64_t nodeId,
        void *&outDst, int &outValue,
        size_t &outPitch, size_t &outWidth,
        size_t &outHeight, size_t &outDepth) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getMemsetNodeParams(
        graph, nodeId, outDst, outValue, outPitch, outWidth, outHeight, outDepth);
}

VGREResult RuntimeEngine::graphHostNodeGetParams(
        GraphId graph, uint64_t nodeId,
        void (**outFn)(void *), void *&outUserData) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getHostNodeParams(graph, nodeId, outFn, outUserData);
}

// ── Node param update (on template graph) ─────────────────────────────────────

VGREResult RuntimeEngine::graphKernelNodeSetParams(
        GraphId graph, uint64_t nodeId, void **args,
        const std::vector<ArgType> &argTypes) {
    GRAPH_GUARD();
    return graphManager_->updateKernelNodeArgs(graph, nodeId, args, argTypes);
}

VGREResult RuntimeEngine::graphMemsetNodeSetParams(
        GraphId graph, uint64_t nodeId, void *dst, int value,
        size_t pitch, size_t width, size_t height, size_t depth) {
    GRAPH_GUARD();
    return graphManager_->updateMemsetNodeInGraph(
        graph, nodeId, dst, value, pitch, width, height, depth);
}

VGREResult RuntimeEngine::graphHostNodeSetParams(
        GraphId graph, uint64_t nodeId, void (*fn)(void *), void *userData) {
    GRAPH_GUARD();
    return graphManager_->updateHostNodeInGraph(graph, nodeId, fn, userData);
}

// ── Exec-level mutations ──────────────────────────────────────────────────────

VGREResult RuntimeEngine::graphExecKernelNodeSetParams(
        GraphExecId execId, uint64_t nodeId, void **args,
        const std::vector<ArgType> &argTypes) {
    GRAPH_GUARD();
    return graphManager_->execKernelNodeSetParams(execId, nodeId, args, argTypes);
}

VGREResult RuntimeEngine::graphExecMemcpyNodeSetParams(
        GraphExecId execId, uint64_t nodeId,
        void *dst, void *src, size_t count, int kind) {
    GRAPH_GUARD();
    return graphManager_->execMemcpyNodeSetParams(
        execId, nodeId, dst, src, count, kind);
}

VGREResult RuntimeEngine::graphExecMemsetNodeSetParams(
        GraphExecId execId, uint64_t nodeId, void *dst, int value,
        size_t pitch, size_t width, size_t height, size_t depth) {
    GRAPH_GUARD();
    return graphManager_->execMemsetNodeSetParams(
        execId, nodeId, dst, value, pitch, width, height, depth);
}

VGREResult RuntimeEngine::graphExecHostNodeSetParams(
        GraphExecId execId, uint64_t nodeId,
        void (*fn)(void *), void *userData) {
    GRAPH_GUARD();
    return graphManager_->execHostNodeSetParams(execId, nodeId, fn, userData);
}

VGREResult RuntimeEngine::graphExecChildGraphNodeSetParams(
        GraphExecId execId, uint64_t nodeId, GraphId newChildGraph) {
    GRAPH_GUARD();
    return graphManager_->execChildGraphNodeSetParams(
        execId, nodeId, newChildGraph);
}

VGREResult RuntimeEngine::graphExecEventRecordNodeSetEvent(
        GraphExecId execId, uint64_t nodeId, void *event) {
    GRAPH_GUARD();
    return graphManager_->execEventRecordNodeSetEvent(execId, nodeId, event);
}

VGREResult RuntimeEngine::graphExecEventWaitNodeSetEvent(
        GraphExecId execId, uint64_t nodeId, void *event) {
    GRAPH_GUARD();
    return graphManager_->execEventWaitNodeSetEvent(execId, nodeId, event);
}

VGREResult RuntimeEngine::graphNodeSetEnabled(GraphExecId execId,
                                               uint64_t nodeId, bool enabled) {
    GRAPH_GUARD();
    return graphManager_->nodeSetEnabled(execId, nodeId, enabled);
}

VGREResult RuntimeEngine::graphNodeGetEnabled(GraphExecId execId,
                                               uint64_t nodeId,
                                               bool &outEnabled) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->nodeGetEnabled(execId, nodeId, outEnabled);
}

VGREResult RuntimeEngine::graphExecGetFlags(GraphExecId execId,
                                             unsigned int &outFlags) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->getExecFlags(execId, outFlags);
}

VGREResult RuntimeEngine::graphExecSetFlags(GraphExecId execId,
                                             unsigned int flags) {
    GRAPH_GUARD();
    return graphManager_->setExecFlags(execId, flags);
}

VGREResult RuntimeEngine::graphExecGetNodeType(GraphExecId execId,
                                                uint64_t nodeId,
                                                GraphNodeType &outType) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    // GraphManager's findExecNode is public; use it via getExecNodeType.
    return graphManager_->getExecNodeType(execId, nodeId, outType);
}

// ── Dependency editing ────────────────────────────────────────────────────────

VGREResult RuntimeEngine::graphAddDependencies(GraphId graph,
                                                const uint64_t *fromNodes,
                                                const uint64_t *toNodes,
                                                size_t numDeps) {
    GRAPH_GUARD();
    return graphManager_->addDependencies(graph, fromNodes, toNodes, numDeps);
}

VGREResult RuntimeEngine::graphRemoveDependencies(GraphId graph,
                                                   const uint64_t *fromNodes,
                                                   const uint64_t *toNodes,
                                                   size_t numDeps) {
    GRAPH_GUARD();
    return graphManager_->removeDependencies(graph, fromNodes, toNodes, numDeps);
}

VGREResult RuntimeEngine::graphNodeFindInClone(GraphId originalGraph,
                                                uint64_t originalNodeId,
                                                GraphId clonedGraph,
                                                uint64_t &outClonedNodeId) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->findNodeInClone(
        originalGraph, originalNodeId, clonedGraph, outClonedNodeId);
}

VGREResult RuntimeEngine::graphDebugDotPrint(GraphId graph, const char *path,
                                              unsigned int flags) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_) return VGREResult::ERR_NOT_INITIALIZED;
    return graphManager_->debugDotPrint(graph, path, flags);
}

#undef GRAPH_GUARD

} // namespace core
} // namespace vgre
