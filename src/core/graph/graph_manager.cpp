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

// ── Dynamic Graph Node Fusion (Track K) ───────────────────────────────────
// Reads VGRE_GRAPH_FUSION_THRESHOLD (default 3) and, after that many replays,
// analyses the active node sequence for two fusion patterns:
//
//  1. Consecutive MEMCPY_H2D + MEMCPY_H2D feeding the same kernel → merge
//     into a single HOST node that does both copies in sequence.
//  2. Consecutive KERNEL + KERNEL with matching grid/block dims and the same
//     stream → register a fused kernel that calls both bodies sequentially
//     and replace the pair with one KERNEL node.
//
// The result is stored in GraphExec::optimizedNodes.  All future replays of
// this exec execute optimizedNodes instead of sourceGraph->nodes.

static int graph_fusion_threshold() {
    static int kThresh = []() {
        const char* env = ::getenv("VGRE_GRAPH_FUSION_THRESHOLD");
        if (env) {
            int v = std::atoi(env);
            if (v >= 1) return v;
        }
        return 3;
    }();
    return kThresh;
}

// Build an adjacency: nodeId → set of nodeIds that depend on it.
static std::unordered_map<uint64_t, std::unordered_set<uint64_t>>
build_dependents(const std::vector<GraphNode>& nodes) {
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> dep;
    for (const auto& n : nodes)
        for (uint64_t d : n.deps)
            dep[d].insert(n.nodeId);
    return dep;
}

static std::vector<GraphNode>
fuse_graph_nodes(const std::vector<GraphNode>& nodes) {
    if (nodes.size() < 2) return nodes;

    auto& engine = RuntimeEngine::instance();
    auto dependents = build_dependents(nodes);

    // Build a quick lookup: nodeId → index
    std::unordered_map<uint64_t, size_t> idx;
    for (size_t i = 0; i < nodes.size(); ++i) idx[nodes[i].nodeId] = i;

    std::vector<bool> fused(nodes.size(), false);
    std::vector<GraphNode> out;
    out.reserve(nodes.size());

    // We only allocate shared fusion context structs when needed; use a vector
    // to keep them alive for the duration of this graph exec.
    static std::vector<std::pair<std::function<void()>,
                                 std::unique_ptr<uint8_t[]>>> s_fused_ctxs;

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (fused[i]) continue;
        const GraphNode& a = nodes[i];

        // ── Pattern 1: MEMCPY H2D + MEMCPY H2D ─────────────────────────────
        if (a.type == GraphNodeType::MEMCPY &&
            a.kind == VGRE_MEMCPY_HOST_TO_DEVICE &&
            i + 1 < nodes.size()) {
            const GraphNode& b = nodes[i + 1];
            if (!fused[i + 1] &&
                b.type == GraphNodeType::MEMCPY &&
                b.kind == VGRE_MEMCPY_HOST_TO_DEVICE &&
                // b must be independent of a (no direct dep)
                std::find(b.deps.begin(), b.deps.end(), a.nodeId) == b.deps.end()) {
                // a and b both H2D, consecutive — check that the only
                // dependents of a are b (or the next kernel) and no cross-dep.
                bool safe = true;
                auto ait = dependents.find(a.nodeId);
                if (ait != dependents.end()) {
                    for (uint64_t dep : ait->second) {
                        if (dep != b.nodeId) {
                            // a has other dependents — not safe to fuse
                            // (could create ordering hazard)
                            safe = false; break;
                        }
                    }
                }
                if (safe) {
                    // Create a fused HOST node that does both copies.
                    // Capture copy params by value so they outlive this stack frame.
                    struct CopyPair { void*dst1; const void*src1; size_t n1;
                                      void*dst2; const void*src2; size_t n2; };
                    auto* cp = new CopyPair{a.dst, a.src, a.count,
                                            b.dst, b.src, b.count};
                    GraphNode fused_node;
                    fused_node.type = GraphNodeType::HOST;
                    fused_node.nodeId = a.nodeId; // reuse first id
                    fused_node.deps = a.deps;
                    // Inherit any deps of b that are not a itself
                    for (uint64_t d : b.deps)
                        if (d != a.nodeId) fused_node.deps.push_back(d);
                    fused_node.streamId = a.streamId;
                    fused_node.hostFn = [](void* ctx) {
                        auto* p = static_cast<CopyPair*>(ctx);
                        memcpy(p->dst1, p->src1, p->n1);
                        memcpy(p->dst2, p->src2, p->n2);
                    };
                    fused_node.hostUserData = static_cast<void*>(cp);
                    // Ownership: stored in static list so it isn't freed too early.
                    s_fused_ctxs.push_back({[cp]{ delete cp; },
                        std::unique_ptr<uint8_t[]>()});

                    out.push_back(std::move(fused_node));
                    fused[i + 1] = true;
                    VGRE_LOG_DEBUG("GraphManager",
                        "Graph fusion: merged H2D MEMCPY pair → single HOST node");
                    continue;
                }
            }
        }

        // ── Pattern 2: KERNEL + KERNEL (same grid/block, same stream) ───────
        if (a.type == GraphNodeType::KERNEL && i + 1 < nodes.size()) {
            const GraphNode& b = nodes[i + 1];
            if (!fused[i + 1] &&
                b.type == GraphNodeType::KERNEL &&
                b.streamId == a.streamId &&
                b.gridDim.x == a.gridDim.x && b.gridDim.y == a.gridDim.y &&
                b.gridDim.z == a.gridDim.z &&
                b.blockDim.x == a.blockDim.x && b.blockDim.y == a.blockDim.y &&
                b.blockDim.z == a.blockDim.z) {
                // Check no shared output dependency: capturedWritePtrs of a
                // must not overlap capturedReadPtrs of b (and vice-versa).
                bool noHazard = true;
                for (void* wp : a.capturedWritePtrs) {
                    for (void* rp : b.capturedReadPtrs) {
                        if (wp == rp) { noHazard = false; break; }
                    }
                    if (!noHazard) break;
                }
                // b must depend only on a (linear chain) or be independent
                bool linearDep = (b.deps.size() == 1 && b.deps[0] == a.nodeId) ||
                                  b.deps.empty();

                if (noHazard && linearDep) {
                    const KernelIR* irA = engine.getKernelIR(a.kernelId);
                    const KernelIR* irB = engine.getKernelIR(b.kernelId);
                    if (irA && irB) {
                        // Build a fused kernel source: sequential call of both bodies.
                        std::string fusedName = "__vgre_fused_" + a.kernelName +
                                                "_" + b.kernelName;
                        std::string fusedSrc = irA->source + "\n" + irB->source +
                            "\nextern \"C\" __global__ void " + fusedName +
                            "(void** argsA, void** argsB) {\n"
                            "    " + a.kernelName + "<<<gridDim,blockDim>>>(argsA);\n"
                            "    " + b.kernelName + "<<<gridDim,blockDim>>>(argsB);\n"
                            "}\n";

                        KernelId fusedId = 0;
                        if (engine.registerKernel(fusedName, fusedSrc, fusedId) ==
                            VGREResult::SUCCESS) {
                            GraphNode fused_node = a; // copy base node
                            fused_node.nodeId   = a.nodeId;
                            fused_node.kernelId = fusedId;
                            fused_node.kernelName = fusedName;
                            // Merge read/write ptr sets from both nodes
                            for (void* p : b.capturedWritePtrs)
                                fused_node.capturedWritePtrs.push_back(p);
                            for (void* p : b.capturedReadPtrs)
                                fused_node.capturedReadPtrs.push_back(p);
                            // Merge args (argsA = a's args, argsB = b's args)
                            for (const auto& arg : b.capturedArgs)
                                fused_node.capturedArgs.push_back(arg);

                            out.push_back(std::move(fused_node));
                            fused[i + 1] = true;
                            VGRE_LOG_DEBUG("GraphManager",
                                "Graph fusion: merged KERNEL pair '" +
                                a.kernelName + "' + '" + b.kernelName + "'");
                            continue;
                        }
                    }
                }
            }
        }

        out.push_back(a);
    }

    return out;
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

  // ── Dynamic graph node fusion ─────────────────────────────────────────────
  // After graph_fusion_threshold() replays, run the fusion analyser once and
  // cache the result in exec->optimizedNodes.  Later replays use the cached
  // optimised node list, avoiding repeated analysis cost.
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!exec->fusionApplied &&
        exec->profile.launch_count >= static_cast<uint64_t>(graph_fusion_threshold())) {
      exec->optimizedNodes = fuse_graph_nodes(activeNodes);
      exec->fusionApplied  = true;
      VGRE_LOG_INFO("GraphManager",
          "Graph exec " + std::to_string(execId) + ": fusion applied, " +
          std::to_string(activeNodes.size()) + " → " +
          std::to_string(exec->optimizedNodes.size()) + " nodes");
    }
    if (exec->fusionApplied)
      activeNodes = exec->optimizedNodes;
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
