/**
 * VGRE Graph Manager — Executable Graph Mutation APIs
 *
 * Implements in-place parameter updates on a launched executable graph
 * (GraphExec) without reinstantiating it.  All mutations operate on
 * GraphExec::sourceGraph's working copy so subsequent calls to
 * RuntimeEngine::graphLaunch pick up the updated parameters.
 *
 * Design: GraphExec::sourceGraph is a shared_ptr<Graph> created by
 * instantiate() as a deep-clone of the template graph.  Mutations here
 * write directly into that cloned Graph so the template is unaffected.
 */

#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"

#include <cstring>

namespace vgre {
namespace core {

// ── Helper: locate a mutable node in an exec's working graph ─────────────────

GraphNode *GraphManager::findExecNode(GraphExecId execId, uint64_t nodeId) {
    auto eit = executables_.find(execId);
    if (eit == executables_.end() || !eit->second) return nullptr;
    auto &graph = eit->second->sourceGraph;
    if (!graph) return nullptr;
    auto nit = graph->nodeIndex.find(nodeId);
    if (nit == graph->nodeIndex.end()) return nullptr;
    return &graph->nodes[nit->second];
}

// ── Exec kernel node parameter update ────────────────────────────────────────

VGREResult GraphManager::execKernelNodeSetParams(
        GraphExecId execId, uint64_t nodeId,
        void **args, const std::vector<ArgType> &argTypes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::KERNEL)
        return VGREResult::ERR_INVALID_VALUE;

    n->capturedArgs.clear();
    n->capturedWritePtrs.clear();
    n->capturedReadPtrs.clear();

    for (size_t i = 0; i < argTypes.size(); ++i) {
        size_t sz = 0;
        switch (argTypes[i]) {
        case ArgType::INT32: case ArgType::UINT32: case ArgType::FLOAT32:
            sz = 4; break;
        case ArgType::POINTER: case ArgType::INT64: case ArgType::UINT64:
        case ArgType::FLOAT64: case ArgType::STRUCT: default:
            sz = 8; break;
        }

        // For STRUCT, attempt size from KernelIR.
        if (argTypes[i] == ArgType::STRUCT) {
            const auto *ir = RuntimeEngine::instance().getKernelIR(n->kernelId);
            if (ir && i < ir->argSizes.size() && ir->argSizes[i] > 0)
                sz = ir->argSizes[i];
        }

        std::vector<uint8_t> buf(sz, 0);
        if (args && args[i] && sz > 0)
            memcpy(buf.data(), args[i], sz);

        if (argTypes[i] == ArgType::POINTER && buf.size() >= sizeof(void *)) {
            void *p = nullptr;
            memcpy(&p, buf.data(), sizeof(void *));
            if (p) {
                n->capturedWritePtrs.push_back(p);
                n->capturedReadPtrs.push_back(p);
            }
        }
        n->capturedArgs.push_back(std::move(buf));
    }

    VGRE_LOG_INFO("GraphManager",
                  "execKernelNodeSetParams: exec=" + std::to_string(execId) +
                  " node=" + std::to_string(nodeId));
    return VGREResult::SUCCESS;
}

// ── Exec memcpy node parameter update ────────────────────────────────────────

VGREResult GraphManager::execMemcpyNodeSetParams(GraphExecId execId,
                                                  uint64_t nodeId,
                                                  void *dst, void *src,
                                                  size_t count, int kind) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::MEMCPY)
        return VGREResult::ERR_INVALID_VALUE;
    if (!dst || !src || count == 0)
        return VGREResult::ERR_INVALID_VALUE;

    n->dst   = dst;
    n->src   = src;
    n->count = count;
    n->kind  = kind;
    return VGREResult::SUCCESS;
}

// ── Exec memset node parameter update ────────────────────────────────────────

VGREResult GraphManager::execMemsetNodeSetParams(GraphExecId execId,
                                                  uint64_t nodeId,
                                                  void *dst, int value,
                                                  size_t pitch, size_t width,
                                                  size_t height, size_t depth) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::MEMSET)
        return VGREResult::ERR_INVALID_VALUE;
    if (!dst || width == 0 || height == 0 || depth == 0 || pitch < width)
        return VGREResult::ERR_INVALID_VALUE;

    n->dst          = dst;
    n->memsetValue  = value;
    n->memsetPitch  = pitch;
    n->memsetWidth  = width;
    n->memsetHeight = height;
    n->memsetDepth  = depth;
    return VGREResult::SUCCESS;
}

// ── Exec host node parameter update ──────────────────────────────────────────

VGREResult GraphManager::execHostNodeSetParams(GraphExecId execId,
                                               uint64_t nodeId,
                                               void (*fn)(void *),
                                               void *userData) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::HOST)
        return VGREResult::ERR_INVALID_VALUE;
    if (!fn)
        return VGREResult::ERR_INVALID_VALUE;

    n->hostFn       = fn;
    n->hostUserData = userData;
    return VGREResult::SUCCESS;
}

// ── Exec child graph node parameter update ────────────────────────────────────

VGREResult GraphManager::execChildGraphNodeSetParams(GraphExecId execId,
                                                     uint64_t nodeId,
                                                     GraphId newChildGraph) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::CHILD)
        return VGREResult::ERR_INVALID_VALUE;
    if (graphs_.find(newChildGraph) == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    n->bodyGraphId = newChildGraph;
    return VGREResult::SUCCESS;
}

// ── Exec event record/wait node event update ──────────────────────────────────

VGREResult GraphManager::execEventRecordNodeSetEvent(GraphExecId execId,
                                                      uint64_t nodeId,
                                                      void *event) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::EVENT_RECORD)
        return VGREResult::ERR_INVALID_VALUE;
    if (!event)
        return VGREResult::ERR_INVALID_VALUE;

    n->eventHandle = event;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::execEventWaitNodeSetEvent(GraphExecId execId,
                                                    uint64_t nodeId,
                                                    void *event) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    GraphNode *n = findExecNode(execId, nodeId);
    if (!n || n->type != GraphNodeType::EVENT_WAIT)
        return VGREResult::ERR_INVALID_VALUE;
    if (!event)
        return VGREResult::ERR_INVALID_VALUE;

    n->eventHandle = event;
    return VGREResult::SUCCESS;
}

// ── Exec node enable/disable ──────────────────────────────────────────────────

VGREResult GraphManager::nodeSetEnabled(GraphExecId execId, uint64_t nodeId,
                                         bool enabled) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto eit = executables_.find(execId);
    if (eit == executables_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto &graph = eit->second->sourceGraph;
    if (!graph || graph->nodeIndex.find(nodeId) == graph->nodeIndex.end())
        return VGREResult::ERR_INVALID_VALUE;

    eit->second->nodeEnabled[nodeId] = enabled;
    VGRE_LOG_INFO("GraphManager",
                  "nodeSetEnabled: exec=" + std::to_string(execId) +
                  " node=" + std::to_string(nodeId) +
                  " enabled=" + (enabled ? "true" : "false"));
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::nodeGetEnabled(GraphExecId execId, uint64_t nodeId,
                                         bool &outEnabled) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto eit = executables_.find(execId);
    if (eit == executables_.end()) return VGREResult::ERR_INVALID_VALUE;

    auto nit = eit->second->nodeEnabled.find(nodeId);
    outEnabled = (nit == eit->second->nodeEnabled.end()) ? true : nit->second;
    return VGREResult::SUCCESS;
}

// ── Exec flags introspection ──────────────────────────────────────────────────

VGREResult GraphManager::getExecFlags(GraphExecId execId,
                                       unsigned int &outFlags) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto eit = executables_.find(execId);
    if (eit == executables_.end()) return VGREResult::ERR_INVALID_VALUE;
    outFlags = eit->second->flags;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::setExecFlags(GraphExecId execId, unsigned int flags) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto eit = executables_.find(execId);
    if (eit == executables_.end()) return VGREResult::ERR_INVALID_VALUE;
    eit->second->flags = flags;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getExecNodeType(GraphExecId execId, uint64_t nodeId,
                                          GraphNodeType &outType) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto eit = executables_.find(execId);
    if (eit == executables_.end()) return VGREResult::ERR_INVALID_VALUE;
    const auto &graph = eit->second->sourceGraph;
    if (!graph) return VGREResult::ERR_INVALID_VALUE;
    auto nit = graph->nodeIndex.find(nodeId);
    if (nit == graph->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    outType = graph->nodes[nit->second].type;
    return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
