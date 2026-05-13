/**
 * VGRE Graph Manager — Extended Node Types
 *
 * Implements MEMSET, HOST, CHILD, EMPTY, EVENT_RECORD, EVENT_WAIT,
 * MEMALLOC, and MEMFREE graph nodes as required by the CUDA 12.x runtime API.
 * All allocations for MEMALLOC nodes use MemoryManager so pointer provenance
 * is tracked for UVM, pointer-attribute queries, and OOM reporting.
 */

#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/event.h"
#include "vgre/common/logger.h"

#include <cstdlib>
#include <cstring>

namespace vgre {
namespace core {

// ── MEMSET node ─────────────────────────────────────────────────────────────

VGREResult GraphManager::addMemsetNode(GraphId id, void *dst, int value,
                                       size_t pitch, size_t width,
                                       size_t height, size_t depth,
                                       const std::vector<uint64_t> &deps,
                                       uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!dst || width == 0 || height == 0 || depth == 0)
        return VGREResult::ERR_INVALID_VALUE;
    if (pitch < width)
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type     = GraphNodeType::MEMSET;
    node.nodeId   = it->second->nextNodeId++;
    node.deps     = deps;
    node.dst      = dst;
    node.memsetValue  = value;
    node.memsetPitch  = pitch;
    node.memsetWidth  = width;
    node.memsetHeight = height;
    node.memsetDepth  = depth;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added MEMSET node " + std::to_string(outNodeId) +
                  " dst=" + std::to_string(reinterpret_cast<uintptr_t>(dst)) +
                  " " + std::to_string(width) + "x" + std::to_string(height) +
                  "x" + std::to_string(depth) + " pitch=" + std::to_string(pitch));
    return VGREResult::SUCCESS;
}

// ── HOST node ────────────────────────────────────────────────────────────────

VGREResult GraphManager::addHostNode(GraphId id, void (*fn)(void *),
                                     void *userData,
                                     const std::vector<uint64_t> &deps,
                                     uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!fn)
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type         = GraphNodeType::HOST;
    node.nodeId       = it->second->nextNodeId++;
    node.deps         = deps;
    node.hostFn       = fn;
    node.hostUserData = userData;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added HOST node " + std::to_string(outNodeId));
    return VGREResult::SUCCESS;
}

// ── CHILD graph node ─────────────────────────────────────────────────────────

VGREResult GraphManager::addChildGraphNode(GraphId id, GraphId childGraphId,
                                           const std::vector<uint64_t> &deps,
                                           uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (childGraphId == 0 || graphs_.find(childGraphId) == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (childGraphId == id)
        return VGREResult::ERR_INVALID_VALUE;  // self-reference is a cycle

    GraphNode node;
    node.type        = GraphNodeType::CHILD;
    node.nodeId      = it->second->nextNodeId++;
    node.deps        = deps;
    node.bodyGraphId = childGraphId;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added CHILD node " + std::to_string(outNodeId) +
                  " child=" + std::to_string(childGraphId));
    return VGREResult::SUCCESS;
}

// ── EMPTY node ───────────────────────────────────────────────────────────────

VGREResult GraphManager::addEmptyNode(GraphId id,
                                      const std::vector<uint64_t> &deps,
                                      uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type   = GraphNodeType::EMPTY;
    node.nodeId = it->second->nextNodeId++;
    node.deps   = deps;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added EMPTY node " + std::to_string(outNodeId));
    return VGREResult::SUCCESS;
}

// ── EVENT_RECORD node ─────────────────────────────────────────────────────────

VGREResult GraphManager::addEventRecordNode(GraphId id, void *event,
                                            const std::vector<uint64_t> &deps,
                                            uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!event)
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type        = GraphNodeType::EVENT_RECORD;
    node.nodeId      = it->second->nextNodeId++;
    node.deps        = deps;
    node.eventHandle = event;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added EVENT_RECORD node " + std::to_string(outNodeId));
    return VGREResult::SUCCESS;
}

// ── EVENT_WAIT node ───────────────────────────────────────────────────────────

VGREResult GraphManager::addEventWaitNode(GraphId id, void *event,
                                          const std::vector<uint64_t> &deps,
                                          uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!event)
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type        = GraphNodeType::EVENT_WAIT;
    node.nodeId      = it->second->nextNodeId++;
    node.deps        = deps;
    node.eventHandle = event;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added EVENT_WAIT node " + std::to_string(outNodeId));
    return VGREResult::SUCCESS;
}

// ── MEMALLOC node ─────────────────────────────────────────────────────────────
//
// Pre-allocates backing memory via MemoryManager at node-add time so that
// the returned pointer is valid immediately (CUDA semantics permit reading
// the output dptr from the node params right after the call).  The pointer
// is recorded in memAllocRegistry_ for later release by MEMFREE.

VGREResult GraphManager::addMemAllocNode(GraphId id, size_t bytesize,
                                         void **outDevPtr,
                                         const std::vector<uint64_t> &deps,
                                         uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (bytesize == 0 || !outDevPtr)
        return VGREResult::ERR_INVALID_VALUE;

    // Pre-allocate through MemoryManager for full UVM / pointer-attribute tracking.
    void *allocHandle = nullptr;
    auto r = MemoryManager::instance().allocate(bytesize, allocHandle);
    if (r != VGREResult::SUCCESS)
        return r;

    *outDevPtr = allocHandle;
    memAllocRegistry_[allocHandle] = bytesize;

    GraphNode node;
    node.type           = GraphNodeType::MEMALLOC;
    node.nodeId         = it->second->nextNodeId++;
    node.deps           = deps;
    node.memAllocSize   = bytesize;
    node.memAllocPtr    = allocHandle;
    node.memAllocOutPtr = outDevPtr;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added MEMALLOC node " + std::to_string(outNodeId) +
                  " size=" + std::to_string(bytesize) +
                  " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(allocHandle)));
    return VGREResult::SUCCESS;
}

// ── MEMFREE node ──────────────────────────────────────────────────────────────

VGREResult GraphManager::addMemFreeNode(GraphId id, void *devPtr,
                                        const std::vector<uint64_t> &deps,
                                        uint64_t &outNodeId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!devPtr)
        return VGREResult::ERR_INVALID_VALUE;
    if (memAllocRegistry_.find(devPtr) == memAllocRegistry_.end())
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode node;
    node.type   = GraphNodeType::MEMFREE;
    node.nodeId = it->second->nextNodeId++;
    node.deps   = deps;
    node.dst    = devPtr;

    outNodeId = node.nodeId;
    it->second->nodeIndex[outNodeId] = it->second->nodes.size();
    it->second->nodes.push_back(std::move(node));

    VGRE_LOG_INFO("GraphManager",
                  "Added MEMFREE node " + std::to_string(outNodeId) +
                  " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(devPtr)));
    return VGREResult::SUCCESS;
}

// ── Template-graph node param updates ────────────────────────────────────────

VGREResult GraphManager::updateMemsetNodeInGraph(GraphId id, uint64_t nodeId,
                                                  void *dst, int value,
                                                  size_t pitch, size_t width,
                                                  size_t height, size_t depth) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    GraphNode &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::MEMSET) return VGREResult::ERR_INVALID_VALUE;
    if (!dst || width == 0 || height == 0 || depth == 0 || pitch < width)
        return VGREResult::ERR_INVALID_VALUE;

    n.dst          = dst;
    n.memsetValue  = value;
    n.memsetPitch  = pitch;
    n.memsetWidth  = width;
    n.memsetHeight = height;
    n.memsetDepth  = depth;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::updateHostNodeInGraph(GraphId id, uint64_t nodeId,
                                               void (*fn)(void *),
                                               void *userData) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    GraphNode &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::HOST) return VGREResult::ERR_INVALID_VALUE;
    if (!fn) return VGREResult::ERR_INVALID_VALUE;

    n.hostFn       = fn;
    n.hostUserData = userData;
    return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
