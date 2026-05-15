#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/graph_optimizer.h"
#include <algorithm>
#include <cstring>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace vgre {
namespace core {


GraphManager::GraphManager() = default;
GraphManager::~GraphManager() = default;

// Thread-local capture-stream context: set by the RuntimeEngine before calling
// addKernelNodeWithDepsOut so the stream ID is stored in the GraphNode without
// requiring an API change to every addKernelNode overload.
thread_local StreamId g_capture_stream_id = 0;

vgre::VGREResult GraphManager::createGraph(GraphId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto graph = std::make_shared<Graph>();
  graph->id = nextGraphId_++;
  graphs_[graph->id] = graph;
  outId = graph->id;
  return vgre::VGREResult::SUCCESS;
}

bool GraphManager::graphExists(GraphId id) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return graphs_.count(id) > 0;
}

vgre::VGREResult GraphManager::destroyGraph(GraphId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  graphs_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::cloneGraph(GraphId srcId, GraphId &outCloneId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(srcId);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;
  // Deep-copy the graph. GraphNode has value semantics (capturedArgs/deps are
  // std::vector, so the default copy constructor performs a deep copy).
  auto clone = std::make_shared<Graph>();
  clone->id = nextGraphId_++;
  clone->nextNodeId = it->second->nextNodeId;
  clone->nodes = it->second->nodes;
  // Rebuild node index map for the cloned graph
  for (size_t i = 0; i < clone->nodes.size(); ++i) {
    clone->nodeIndex[clone->nodes[i].nodeId] = i;
  }
  graphs_[clone->id] = clone;
  outCloneId = clone->id;

  // Record node ID mapping for cudaGraphNodeFindInClone.
  // In a structural clone the node IDs are identical (same nodeId values).
  auto &nodeMap = cloneNodeMap_[srcId];
  nodeMap.clear();
  for (const auto &node : clone->nodes)
    nodeMap[node.nodeId] = node.nodeId;
  cloneOriginMap_[clone->id] = srcId;

  VGRE_LOG_INFO("GraphManager", "Cloned graph " + std::to_string(srcId) +
                " -> " + std::to_string(outCloneId) +
                " (" + std::to_string(clone->nodes.size()) + " nodes)");
  return vgre::VGREResult::SUCCESS;
}


// ── validateGraph ──────────────────────────────────────────────────────────
// Semantic checks beyond cycle detection:
//   • Non-empty graph
//   • All dependency IDs reference nodes that exist in the graph
//   • KERNEL nodes: gridDim and blockDim are all ≥ 1
//   • MEMCPY nodes: count > 0, dst and src are non-null
vgre::VGREResult GraphManager::validateGraph(GraphId id) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  const auto &g = *it->second;
  if (g.nodes.empty()) {
    VGRE_LOG_ERROR("GraphManager",
        "validateGraph(" + std::to_string(id) + "): graph has no nodes");
    return vgre::VGREResult::ERR_INVALID_VALUE;
  }

  // Build nodeId set for O(1) dep lookup
  std::unordered_map<uint64_t, size_t> nodeIndex;
  for (size_t i = 0; i < g.nodes.size(); ++i)
    nodeIndex[g.nodes[i].nodeId] = i;

  for (const auto &n : g.nodes) {
    // Validate dependency references
    for (uint64_t dep : n.deps) {
      if (nodeIndex.find(dep) == nodeIndex.end()) {
        VGRE_LOG_ERROR("GraphManager",
            "validateGraph(" + std::to_string(id) + "): node " +
            std::to_string(n.nodeId) + " has dangling dependency " +
            std::to_string(dep));
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
    }

    if (n.type == GraphNodeType::KERNEL) {
      // Kernel nodes must have valid launch dimensions
      if (n.gridDim.x == 0 || n.gridDim.y == 0 || n.gridDim.z == 0) {
        VGRE_LOG_ERROR("GraphManager",
            "validateGraph(" + std::to_string(id) + "): kernel node " +
            std::to_string(n.nodeId) + " has zero gridDim");
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
      if (n.blockDim.x == 0 || n.blockDim.y == 0 || n.blockDim.z == 0) {
        VGRE_LOG_ERROR("GraphManager",
            "validateGraph(" + std::to_string(id) + "): kernel node " +
            std::to_string(n.nodeId) + " has zero blockDim");
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
    } else if (n.type == GraphNodeType::MEMCPY) {
      if (n.count == 0) {
        VGRE_LOG_ERROR("GraphManager",
            "validateGraph(" + std::to_string(id) + "): memcpy node " +
            std::to_string(n.nodeId) + " has count=0");
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
      if (!n.dst || !n.src) {
        VGRE_LOG_ERROR("GraphManager",
            "validateGraph(" + std::to_string(id) + "): memcpy node " +
            std::to_string(n.nodeId) + " has null dst or src");
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
    }
  }

  VGRE_LOG_DEBUG("GraphManager",
      "validateGraph(" + std::to_string(id) + "): OK (" +
      std::to_string(g.nodes.size()) + " nodes)");
  return vgre::VGREResult::SUCCESS;
}

// ── getExecProfile ─────────────────────────────────────────────────────────
vgre::VGREResult GraphManager::getExecProfile(GraphExecId id,
                                               double &lastExecMs,
                                               double &avgExecMs,
                                               uint64_t &launchCount) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = executables_.find(id);
  if (it == executables_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  const auto &p = it->second->profile;
  lastExecMs  = p.last_exec_ms;
  avgExecMs   = p.avg_exec_ms();
  launchCount = p.launch_count;
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::launch(GraphExecId execId, StreamId stream) {
  std::shared_ptr<GraphExec> exec;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = executables_.find(execId);
    if (it == executables_.end())
      return vgre::VGREResult::ERR_INVALID_VALUE;
    exec = it->second;
  }

  auto &engine = RuntimeEngine::instance();
  VGRE_LOG_INFO("GraphManager", "Dispatching real native DAG from Executable " +
                                     std::to_string(execId) + " on stream " +
                                     std::to_string(stream));

  // Filter disabled nodes; rebuild dep list to exclude any edges leading to
  // disabled nodes so the topology remains consistent for the dispatcher.
  std::vector<GraphNode> activeNodes;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::unordered_set<uint64_t> disabledIds;
    for (const auto &kv : exec->nodeEnabled)
      if (!kv.second) disabledIds.insert(kv.first);

    if (disabledIds.empty()) {
      activeNodes = exec->sourceGraph->nodes;
    } else {
      activeNodes.reserve(exec->sourceGraph->nodes.size());
      for (const auto &node : exec->sourceGraph->nodes) {
        if (disabledIds.count(node.nodeId)) continue;
        GraphNode n = node;
        auto &deps = n.deps;
        deps.erase(std::remove_if(deps.begin(), deps.end(),
                                  [&](uint64_t d){ return disabledIds.count(d) > 0; }),
                   deps.end());
        activeNodes.push_back(std::move(n));
      }
    }
  }

  // Timed dispatch — updates GraphExecProfile for this executable.
  auto t0 = std::chrono::steady_clock::now();
  VGREResult r = engine.dispatchGraphNodes(activeNodes, stream);
  auto t1 = std::chrono::steady_clock::now();

  double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto &p = exec->profile;
    p.last_exec_ms   = elapsedMs;
    p.total_exec_ms += elapsedMs;
    p.launch_count++;
  }

  VGRE_LOG_DEBUG("GraphManager",
      "Exec " + std::to_string(execId) + " completed in " +
      std::to_string(elapsedMs) + " ms (launch #" +
      std::to_string(exec->profile.launch_count) + ")");
  return r;
}


} // namespace core
} // namespace vgre
