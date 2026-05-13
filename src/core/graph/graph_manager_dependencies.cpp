/**
 * VGRE Graph Manager — Dependency Editing APIs
 *
 * Implements cudaGraphAddDependencies and cudaGraphRemoveDependencies by
 * manipulating GraphNode::deps vectors under the manager's mutex.
 * Both APIs process arrays of (from, to) edge descriptors and validate
 * that all referenced nodes exist before committing any changes (all-or-
 * nothing semantics to match CUDA's specification).
 */

#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>

namespace vgre {
namespace core {

// ── Add bulk dependencies ─────────────────────────────────────────────────────

VGREResult GraphManager::removeDependency(GraphId id, uint64_t nodeId,
                                          uint64_t dependsOn) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;

    auto nit = git->second->nodeIndex.find(nodeId);
    if (nit == git->second->nodeIndex.end())
        return VGREResult::ERR_INVALID_VALUE;

    GraphNode &node = git->second->nodes[nit->second];
    auto &deps = node.deps;
    auto it = std::find(deps.begin(), deps.end(), dependsOn);
    if (it == deps.end())
        return VGREResult::ERR_INVALID_VALUE;

    deps.erase(it);
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::addDependencies(GraphId id,
                                          const uint64_t *fromNodes,
                                          const uint64_t *toNodes,
                                          size_t numDeps) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!fromNodes || !toNodes || numDeps == 0)
        return VGREResult::ERR_INVALID_VALUE;

    // Validate all referenced node IDs before mutating anything.
    for (size_t i = 0; i < numDeps; ++i) {
        if (git->second->nodeIndex.find(fromNodes[i]) == git->second->nodeIndex.end() ||
            git->second->nodeIndex.find(toNodes[i])   == git->second->nodeIndex.end())
            return VGREResult::ERR_INVALID_VALUE;
        if (fromNodes[i] == toNodes[i])
            return VGREResult::ERR_INVALID_VALUE;  // self-loop
    }

    // Commit edges: for each (from -> to), add fromNodes[i] to toNodes[i].deps.
    for (size_t i = 0; i < numDeps; ++i) {
        auto &toNode = git->second->nodes[git->second->nodeIndex[toNodes[i]]];
        bool already = false;
        for (auto d : toNode.deps)
            if (d == fromNodes[i]) { already = true; break; }
        if (!already)
            toNode.deps.push_back(fromNodes[i]);
    }

    VGRE_LOG_INFO("GraphManager",
                  "addDependencies: graph=" + std::to_string(id) +
                  " added " + std::to_string(numDeps) + " edges");
    return VGREResult::SUCCESS;
}

VGREResult GraphManager::removeDependencies(GraphId id,
                                             const uint64_t *fromNodes,
                                             const uint64_t *toNodes,
                                             size_t numDeps) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto git = graphs_.find(id);
    if (git == graphs_.end())
        return VGREResult::ERR_INVALID_VALUE;
    if (!fromNodes || !toNodes || numDeps == 0)
        return VGREResult::ERR_INVALID_VALUE;

    // Validate all referenced node IDs and confirm all edges exist.
    for (size_t i = 0; i < numDeps; ++i) {
        auto toIt = git->second->nodeIndex.find(toNodes[i]);
        if (toIt == git->second->nodeIndex.end())
            return VGREResult::ERR_INVALID_VALUE;
        if (git->second->nodeIndex.find(fromNodes[i]) == git->second->nodeIndex.end())
            return VGREResult::ERR_INVALID_VALUE;

        const auto &deps = git->second->nodes[toIt->second].deps;
        bool found = false;
        for (auto d : deps) if (d == fromNodes[i]) { found = true; break; }
        if (!found)
            return VGREResult::ERR_INVALID_VALUE;
    }

    // Commit removals.
    for (size_t i = 0; i < numDeps; ++i) {
        auto &deps = git->second->nodes[git->second->nodeIndex[toNodes[i]]].deps;
        deps.erase(std::remove(deps.begin(), deps.end(), fromNodes[i]), deps.end());
    }

    VGRE_LOG_INFO("GraphManager",
                  "removeDependencies: graph=" + std::to_string(id) +
                  " removed " + std::to_string(numDeps) + " edges");
    return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
