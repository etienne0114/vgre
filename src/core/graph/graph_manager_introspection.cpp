/**
 * VGRE Graph Manager — Node Introspection APIs
 *
 * Implements read-only graph topology queries:
 *   - getGraphNodeCount, getGraphRootNodes, getGraphEdges
 *   - getNodeType, getNodeDependencies, getNodeDependentNodes
 *   - findNodeInClone, debugDotPrint
 *   - Per-node parameter accessors (getKernelNodeParams, etc.)
 *
 * All queries run under the manager's recursive mutex and return copies
 * so callers never hold references into the live graph data structures.
 */

#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"

#include <cstdio>
#include <sstream>
#include <unordered_set>

namespace vgre {
namespace core {

// ── Graph-level topology ──────────────────────────────────────────────────────

VGREResult GraphManager::getGraphNodeCount(GraphId id, size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    outCount = it->second->nodes.size();
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getGraphRootNodes(GraphId id,
                                           std::vector<uint64_t> &outNodeIds) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    // Collect all node IDs that appear as someone else's dependency.
    std::unordered_set<uint64_t> hasPredecessor;
    for (const auto &node : it->second->nodes)
        for (auto dep : node.deps)
            hasPredecessor.insert(dep);

    outNodeIds.clear();
    for (const auto &node : it->second->nodes)
        if (hasPredecessor.find(node.nodeId) == hasPredecessor.end())
            outNodeIds.push_back(node.nodeId);

    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getGraphEdges(GraphId id,
                                       std::vector<uint64_t> &fromNodes,
                                       std::vector<uint64_t> &toNodes,
                                       size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    fromNodes.clear();
    toNodes.clear();
    // Each edge is (dependency -> dependent): node.deps[i] -> node.nodeId.
    for (const auto &node : it->second->nodes) {
        for (auto dep : node.deps) {
            fromNodes.push_back(dep);
            toNodes.push_back(node.nodeId);
        }
    }
    outCount = fromNodes.size();
    return VGREResult::SUCCESS;
}

// ── Node-level topology ───────────────────────────────────────────────────────

VGREResult GraphManager::getNodeType(GraphId id, uint64_t nodeId,
                                     GraphNodeType &outType) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end())
        return VGREResult::ERR_INVALID_VALUE;

    outType = git->second->nodes[nit->second].type;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getNodeDependencies(GraphId id, uint64_t nodeId,
                                             std::vector<uint64_t> &outDeps,
                                             size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end())
        return VGREResult::ERR_INVALID_VALUE;

    outDeps  = git->second->nodes[nit->second].deps;
    outCount = outDeps.size();
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getNodeDependentNodes(GraphId id, uint64_t nodeId,
                                               std::vector<uint64_t> &outDependents,
                                               size_t &outCount) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (git->second->nodeIndex.find(nodeId) == git->second->nodeIndex.end())
        return VGREResult::ERR_INVALID_VALUE;

    outDependents.clear();
    for (const auto &node : git->second->nodes) {
        for (auto dep : node.deps) {
            if (dep == nodeId) {
                outDependents.push_back(node.nodeId);
                break;
            }
        }
    }
    outCount = outDependents.size();
    return VGREResult::SUCCESS;
}

// ── Clone tracking ────────────────────────────────────────────────────────────

VGREResult GraphManager::findNodeInClone(GraphId originalGraph,
                                         uint64_t originalNodeId,
                                         GraphId clonedGraph,
                                         uint64_t &outClonedNodeId) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (graphs_.find(clonedGraph) == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    // If originalGraph is 0, look it up from cloneOriginMap_.
    auto oit = cloneOriginMap_.find(clonedGraph);
    if (oit == cloneOriginMap_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (originalGraph == 0)
        originalGraph = oit->second;
    else if (oit->second != originalGraph)
        return VGREResult::ERR_INVALID_VALUE;

    if (graphs_.find(originalGraph) == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    // Look up the node mapping.
    auto mit = cloneNodeMap_.find(originalGraph);
    if (mit == cloneNodeMap_.end())
        return VGREResult::ERR_INVALID_VALUE;

    auto nit = mit->second.find(originalNodeId);
    if (nit == mit->second.end())
        return VGREResult::ERR_INVALID_VALUE;

    outClonedNodeId = nit->second;
    return VGREResult::SUCCESS;
}

// ── DOT-format debug print ────────────────────────────────────────────────────

static const char *nodeTypeName(GraphNodeType t) {
    switch (t) {
    case GraphNodeType::KERNEL:       return "KERNEL";
    case GraphNodeType::MEMCPY:       return "MEMCPY";
    case GraphNodeType::CONDITIONAL:  return "CONDITIONAL";
    case GraphNodeType::MEMSET:       return "MEMSET";
    case GraphNodeType::HOST:         return "HOST";
    case GraphNodeType::CHILD:        return "CHILD";
    case GraphNodeType::EMPTY:        return "EMPTY";
    case GraphNodeType::EVENT_RECORD: return "EVENT_RECORD";
    case GraphNodeType::EVENT_WAIT:   return "EVENT_WAIT";
    case GraphNodeType::MEMALLOC:     return "MEMALLOC";
    case GraphNodeType::MEMFREE:      return "MEMFREE";
    default:                          return "UNKNOWN";
    }
}

VGREResult GraphManager::debugDotPrint(GraphId id, const char *path,
                                       unsigned int /*flags*/) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    std::ostringstream dot;
    dot << "digraph G {\n  rankdir=LR;\n";
    for (const auto &node : it->second->nodes) {
        std::string label = std::string(nodeTypeName(node.type));
        if (node.type == GraphNodeType::KERNEL && !node.kernelName.empty())
            label += "\\n" + node.kernelName;
        dot << "  node" << node.nodeId
            << " [label=\"[" << node.nodeId << "] " << label << "\"];\n";
        for (auto dep : node.deps)
            dot << "  node" << dep << " -> node" << node.nodeId << ";\n";
    }
    dot << "}\n";

    if (path && path[0]) {
        FILE *f = std::fopen(path, "w");
        if (!f) {
            VGRE_LOG_ERROR("GraphManager",
                           "debugDotPrint: cannot open " + std::string(path));
            return VGREResult::ERR_INVALID_VALUE;
        }
        std::fputs(dot.str().c_str(), f);
        std::fclose(f);
    } else {
        VGRE_LOG_INFO("GraphManager", "Graph " + std::to_string(id) +
                      " DOT:\n" + dot.str());
    }
    return VGREResult::SUCCESS;
}

// ── Node parameter accessors ──────────────────────────────────────────────────

VGREResult GraphManager::getKernelNodeParams(
        GraphId id, uint64_t nodeId,
        KernelId &outKernelId, std::string &outName,
        dim3 &outGrid, dim3 &outBlock,
        std::vector<std::vector<uint8_t>> &outArgs) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    const auto &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::KERNEL) return VGREResult::ERR_INVALID_VALUE;

    outKernelId = n.kernelId;
    outName     = n.kernelName;
    outGrid     = n.gridDim;
    outBlock    = n.blockDim;
    outArgs     = n.capturedArgs;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getMemcpyNodeParams(GraphId id, uint64_t nodeId,
                                              void *&outDst, void *&outSrc,
                                              size_t &outCount,
                                              int &outKind) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    const auto &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::MEMCPY) return VGREResult::ERR_INVALID_VALUE;

    outDst   = n.dst;
    outSrc   = n.src;
    outCount = n.count;
    outKind  = n.kind;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getMemsetNodeParams(GraphId id, uint64_t nodeId,
                                              void *&outDst, int &outValue,
                                              size_t &outPitch, size_t &outWidth,
                                              size_t &outHeight,
                                              size_t &outDepth) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    const auto &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::MEMSET) return VGREResult::ERR_INVALID_VALUE;

    outDst    = n.dst;
    outValue  = n.memsetValue;
    outPitch  = n.memsetPitch;
    outWidth  = n.memsetWidth;
    outHeight = n.memsetHeight;
    outDepth  = n.memsetDepth;
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::getHostNodeParams(GraphId id, uint64_t nodeId,
                                            void (**outFn)(void *),
                                            void *&outUserData) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end()) return VGREResult::ERR_INVALID_VALUE;
    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end()) return VGREResult::ERR_INVALID_VALUE;
    const auto &n = git->second->nodes[nit->second];
    if (n.type != GraphNodeType::HOST) return VGREResult::ERR_INVALID_VALUE;

    if (outFn) *outFn = n.hostFn;
    outUserData = n.hostUserData;
    return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
