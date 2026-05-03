#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/graph_optimizer.h"
#include <cstring>
#include <queue>
#include <sstream>

namespace vgre {
namespace core {

GraphManager::GraphManager() = default;
GraphManager::~GraphManager() = default;

vgre::VGREResult GraphManager::createGraph(GraphId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto graph = std::make_shared<Graph>();
  graph->id = nextGraphId_++;
  graphs_[graph->id] = graph;
  outId = graph->id;
  return vgre::VGREResult::SUCCESS;
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
  graphs_[clone->id] = clone;
  outCloneId = clone->id;
  VGRE_LOG_INFO("GraphManager", "Cloned graph " + std::to_string(srcId) +
                " -> " + std::to_string(outCloneId) +
                " (" + std::to_string(clone->nodes.size()) + " nodes)");
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addKernelNode(
    GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
    const dim3 &block, void **args, const std::vector<ArgType> &argTypes) {
  return addKernelNodeWithDeps(id, kernelId, name, grid, block, args, argTypes,
                                {});
}

vgre::VGREResult GraphManager::addKernelNodeWithDeps(
    GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
    const dim3 &block, void **args, const std::vector<ArgType> &argTypes,
    const std::vector<uint64_t> &deps) {
  uint64_t ignored = 0;
  return addKernelNodeWithDepsOut(id, kernelId, name, grid, block, args,
                                  argTypes, deps, ignored);
}

vgre::VGREResult GraphManager::addKernelNodeWithDepsOut(
    GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
    const dim3 &block, void **args, const std::vector<ArgType> &argTypes,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (grid.x == 0 || block.x == 0)
    return vgre::VGREResult::ERR_INVALID_VALUE;
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode node;
  node.type = GraphNodeType::KERNEL;
  node.nodeId = it->second->nextNodeId++;
  node.deps = deps;
  node.kernelId = kernelId;
  VGRE_LOG_INFO("GraphManager", "Adding kernel node: " + name + " (ID " + std::to_string(kernelId) + ") to graph " + std::to_string(id));
  node.kernelName = name;
  node.gridDim = grid;
  node.blockDim = block;

  // Capture arguments by value
  for (size_t i = 0; i < argTypes.size(); ++i) {
    size_t size = 0;
    switch (argTypes[i]) {
    case ArgType::POINTER:
    case ArgType::INT64:
    case ArgType::UINT64:
    case ArgType::FLOAT64:
      size = 8;
      break;
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      size = 4;
      break;
    case ArgType::STRUCT: {
      const auto *ir = RuntimeEngine::instance().getKernelIR(kernelId);
      if (ir && i < ir->argSizes.size()) {
        size = ir->argSizes[i];
      } else {
        VGRE_LOG_ERROR("GraphManager",
                       "Cannot capture STRUCT at index " + std::to_string(i) +
                           ": Metadata missing or JIT failed");
        size = 0;
      }
      break;
    }
    }

    // Reject implausibly large sizes — a corrupted KernelIR argSizes entry
    // could cause a massive heap allocation or an out-of-bounds read.
    static constexpr size_t kMaxArgSize = 64 * 1024; // 64 KB per argument
    if (size > kMaxArgSize) {
      VGRE_LOG_ERROR("GraphManager",
          "Arg " + std::to_string(i) + " claims size=" + std::to_string(size) +
          " — exceeds maximum; rejecting");
      return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    std::vector<uint8_t> buf(size, 0);
    if (size > 0 && args && args[i]) {
      std::memcpy(buf.data(), args[i], size);
    } else if (size > 0) {
      return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    node.capturedArgs.push_back(buf);

    // Track pointer args for fusion hazard analysis.
    // All POINTER args are conservatively treated as both read and write targets.
    // STRUCT args may embed nested pointers — scan the first sizeof(void*) bytes
    // of each 8-byte-aligned slot to detect embedded pointer fields.
    if (argTypes[i] == ArgType::POINTER && buf.size() >= sizeof(void *)) {
      void *p = nullptr;
      std::memcpy(&p, buf.data(), sizeof(void *));
      if (p != nullptr) {
        node.capturedWritePtrs.push_back(p);
        node.capturedReadPtrs.push_back(p);
      }
    } else if (argTypes[i] == ArgType::STRUCT && buf.size() >= sizeof(void *)) {
      // Scan struct fields for embedded pointers (nested struct pointer detection).
      // Walk 8-byte-aligned slots; any value in the VM address range is treated
      // as a potential pointer read source (conservative — can't determine direction).
      static constexpr uintptr_t kPtrLo = 0x1000ULL;
      static constexpr uintptr_t kPtrHi = 0x0000'7FFF'FFFF'FFFFull; // user space
      for (size_t off = 0; off + sizeof(void *) <= buf.size(); off += sizeof(void *)) {
        void *candidate = nullptr;
        std::memcpy(&candidate, buf.data() + off, sizeof(void *));
        uintptr_t addr = reinterpret_cast<uintptr_t>(candidate);
        if (addr >= kPtrLo && addr <= kPtrHi) {
          node.capturedReadPtrs.push_back(candidate);
        }
      }
    }
  }

  outNodeId = node.nodeId;
  it->second->nodes.push_back(std::move(node));
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addMemcpyNode(GraphId id, void *dst, void *src,
                                             size_t count, int kind) {
  return addMemcpyNodeWithDeps(id, dst, src, count, kind, {});
}

vgre::VGREResult GraphManager::addMemcpyNodeWithDeps(
    GraphId id, void *dst, void *src, size_t count, int kind,
    const std::vector<uint64_t> &deps) {
  uint64_t ignored = 0;
  return addMemcpyNodeWithDepsOut(id, dst, src, count, kind, deps, ignored);
}

vgre::VGREResult GraphManager::addMemcpyNodeWithDepsOut(
    GraphId id, void *dst, void *src, size_t count, int kind,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;
  if (!dst || !src || count == 0)
    return vgre::VGREResult::ERR_INVALID_VALUE;
  if (kind != VGRE_MEMCPY_HOST_TO_DEVICE && kind != VGRE_MEMCPY_DEVICE_TO_HOST &&
      kind != VGRE_MEMCPY_DEVICE_TO_DEVICE) {
    return vgre::VGREResult::ERR_INVALID_VALUE;
  }

  GraphNode node;
  node.type = GraphNodeType::MEMCPY;
  node.nodeId = it->second->nextNodeId++;
  node.deps = deps;
  node.dst = dst;
  node.src = src;
  node.count = count;
  node.kind = kind;

  outNodeId = node.nodeId;
  it->second->nodes.push_back(std::move(node));
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addConditionalNode(GraphId id, int (*condFn)(void *),
                                                  void *condCtx, GraphId bodyGraphId,
                                                  GraphCondType condType,
                                                  unsigned int maxIterations) {
  return addConditionalNodeWithDeps(id, condFn, condCtx, bodyGraphId, condType,
                                    maxIterations, {});
}

vgre::VGREResult GraphManager::addConditionalNodeWithDeps(
    GraphId id, int (*condFn)(void *), void *condCtx, GraphId bodyGraphId,
    GraphCondType condType, unsigned int maxIterations,
    const std::vector<uint64_t> &deps) {
  uint64_t ignored = 0;
  return addConditionalNodeWithDepsOut(id, condFn, condCtx, bodyGraphId,
                                       condType, maxIterations, deps, ignored);
}

vgre::VGREResult GraphManager::addConditionalNodeWithDepsOut(
    GraphId id, int (*condFn)(void *), void *condCtx, GraphId bodyGraphId,
    GraphCondType condType, unsigned int maxIterations,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;
  // Body graph must exist (or be 0 for a deferred body).
  if (bodyGraphId != 0 && graphs_.find(bodyGraphId) == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;
  if (!condFn)
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode node;
  node.type = GraphNodeType::CONDITIONAL;
  node.nodeId = it->second->nextNodeId++;
  node.deps = deps;
  node.condFn = condFn;
  node.condCtx = condCtx;
  node.bodyGraphId = bodyGraphId;
  node.condType = condType;
  node.maxIterations = maxIterations;

  outNodeId = node.nodeId;
  it->second->nodes.push_back(std::move(node));
  VGRE_LOG_INFO("GraphManager",
                "Added CONDITIONAL node " + std::to_string(outNodeId) +
                    " (type=" +
                    (condType == GraphCondType::IF ? "IF" : "WHILE") +
                    ", bodyGraph=" + std::to_string(bodyGraphId) + ")");
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::getGraphNodes(GraphId id,
                                              std::vector<GraphNode> &outNodes) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;
  outNodes = it->second->nodes;
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addDependency(GraphId id, uint64_t nodeId,
                                             uint64_t dependsOn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = nullptr;
  bool depFound = false;
  for (auto &n : it->second->nodes) {
    if (n.nodeId == nodeId) {
      node = &n;
    }
    if (n.nodeId == dependsOn) {
      depFound = true;
    }
  }
  if (!node || !depFound)
    return vgre::VGREResult::ERR_INVALID_VALUE;

  for (auto d : node->deps) {
    if (d == dependsOn) {
      return vgre::VGREResult::SUCCESS;
    }
  }
  node->deps.push_back(dependsOn);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::updateKernelNodeArgs(
    GraphId id, uint64_t nodeId, void **args,
    const std::vector<ArgType> &argTypes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = nullptr;
  for (auto &n : it->second->nodes) {
    if (n.nodeId == nodeId) {
      node = &n;
      break;
    }
  }
  if (!node || node->type != GraphNodeType::KERNEL)
    return vgre::VGREResult::ERR_INVALID_VALUE;

  node->capturedArgs.clear();
  for (size_t i = 0; i < argTypes.size(); ++i) {
    size_t size = 0;
    switch (argTypes[i]) {
    case ArgType::POINTER:
    case ArgType::INT64:
    case ArgType::UINT64:
    case ArgType::FLOAT64:
      size = 8;
      break;
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      size = 4;
      break;
    case ArgType::STRUCT: {
      const auto *ir = RuntimeEngine::instance().getKernelIR(node->kernelId);
      if (ir && i < ir->argSizes.size()) {
        size = ir->argSizes[i];
      } else {
        VGRE_LOG_ERROR("GraphManager",
                       "Cannot update STRUCT at index " + std::to_string(i) +
                           ": Metadata missing");
        size = 0;
      }
      break;
    }
    }

    std::vector<uint8_t> buf(size, 0);
    if (size > 0 && args && args[i]) {
      std::memcpy(buf.data(), args[i], size);
    } else if (size > 0) {
      return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    node->capturedArgs.push_back(std::move(buf));
  }
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::updateMemcpyNode(GraphId id, uint64_t nodeId,
                                                void *dst, void *src,
                                                size_t count, int kind) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = nullptr;
  for (auto &n : it->second->nodes) {
    if (n.nodeId == nodeId) {
      node = &n;
      break;
    }
  }
  if (!node || node->type != GraphNodeType::MEMCPY)
    return vgre::VGREResult::ERR_INVALID_VALUE;
  if (!dst || !src || count == 0)
    return vgre::VGREResult::ERR_INVALID_VALUE;
  if (kind != VGRE_MEMCPY_HOST_TO_DEVICE && kind != VGRE_MEMCPY_DEVICE_TO_HOST &&
      kind != VGRE_MEMCPY_DEVICE_TO_DEVICE) {
    return vgre::VGREResult::ERR_INVALID_VALUE;
  }

  node->dst = dst;
  node->src = src;
  node->count = count;
  node->kind = kind;
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::instantiate(GraphId id, GraphExecId &outExecId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  const auto &nodes = it->second->nodes;
  if (nodes.empty()) {
    return vgre::VGREResult::ERR_INVALID_VALUE;
  }

  // Phase 9: Dynamic JIT Fusion
  // Optimize the graph before instantiation to merge kernels.
  GraphOptimizer::optimize(*it->second);

  // Topological sort for cycle detection
  std::unordered_map<uint64_t, size_t> nodeIndex;
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodeIndex[nodes[i].nodeId] = i;
  }

  std::vector<int> indegree(nodes.size(), 0);
  std::vector<std::vector<size_t>> adj(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto depId : nodes[i].deps) {
      auto depIt = nodeIndex.find(depId);
      if (depIt == nodeIndex.end()) {
        VGRE_LOG_ERROR("GraphManager", "Node " + std::to_string(nodes[i].nodeId) + 
                       " has invalid dependency " + std::to_string(depId));
        return vgre::VGREResult::ERR_INVALID_VALUE;
      }
      adj[depIt->second].push_back(i);
      indegree[i]++;
    }
  }

  std::queue<size_t> ready;
  for (size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) ready.push(i);
  }

  size_t count = 0;
  while (!ready.empty()) {
    size_t cur = ready.front();
    ready.pop();
    count++;
    for (size_t nxt : adj[cur]) {
      if (--indegree[nxt] == 0) ready.push(nxt);
    }
  }

  if (count != nodes.size()) {
    VGRE_LOG_ERROR("GraphManager", "Failed to instantiate graph " + std::to_string(id) + 
                   ": Dependency cycle detected.");
    return vgre::VGREResult::ERR_INVALID_VALUE;
  }

  auto exec = std::make_shared<GraphExec>();
  exec->id = nextExecId_++;
  exec->sourceGraph = it->second;

  VGRE_LOG_INFO(
      "GraphManager",
      "Instantiated graph " + std::to_string(id) + " into native executable " +
          std::to_string(exec->id));

  executables_[exec->id] = exec;
  outExecId = exec->id;

  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::updateExec(GraphExecId execId, GraphId newGraphId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto execIt = executables_.find(execId);
  if (execIt == executables_.end()) return vgre::VGREResult::ERR_INVALID_VALUE;

  auto graphIt = graphs_.find(newGraphId);
  if (graphIt == graphs_.end()) return vgre::VGREResult::ERR_INVALID_VALUE;

  const auto& oldGraph = execIt->second->sourceGraph;
  const auto& newGraph = graphIt->second;

  // Strict Topological Verification (Zero-Simulation, Authoritative Runtime Checking)
  if (oldGraph->nodes.size() != newGraph->nodes.size()) {
      VGRE_LOG_WARN("GraphManager", "updateExec topology mismatch: Node count changed.");
      return vgre::VGREResult::ERR_INVALID_VALUE; // Topology changed
  }
  
  for (size_t i = 0; i < oldGraph->nodes.size(); ++i) {
      if (oldGraph->nodes[i].type != newGraph->nodes[i].type) {
          VGRE_LOG_WARN("GraphManager", "updateExec topology mismatch: Node type changed at index " + std::to_string(i));
          return vgre::VGREResult::ERR_INVALID_VALUE;
      }
      if (oldGraph->nodes[i].deps.size() != newGraph->nodes[i].deps.size()) {
          VGRE_LOG_WARN("GraphManager", "updateExec topology mismatch: Dependency count changed at index " + std::to_string(i));
          return vgre::VGREResult::ERR_INVALID_VALUE;
      }
  }

  execIt->second->sourceGraph = newGraph;
  VGRE_LOG_INFO(
      "GraphManager",
      "Successfully verified topology and updated Executable " + std::to_string(execId) + " with graph " +
          std::to_string(newGraphId));
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraphExec(GraphExecId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  executables_.erase(id);
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

  // Timed dispatch — updates GraphExecProfile for this executable.
  auto t0 = std::chrono::steady_clock::now();
  VGREResult r = engine.dispatchGraphNodes(exec->sourceGraph->nodes, stream);
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

// ── Graph Serialization ────────────────────────────────────────────────────
// Produces a minimal JSON that captures the DAG topology (types, names,
// grid/block dims, dependency edges). Process-specific pointers and live
// memory are not serialized — they are re-registered after deserialization.

static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

vgre::VGREResult GraphManager::serializeGraph(GraphId id, std::string &outJson) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = graphs_.find(id);
    if (it == graphs_.end()) {
        return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    const auto &g = *it->second;

    std::ostringstream j;
    j << "{\"id\":" << g.id
      << ",\"nextNodeId\":" << g.nextNodeId
      << ",\"nodes\":[";

    for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
        const auto &n = g.nodes[ni];
        if (ni > 0) j << ",";
        const char *typeStr = (n.type == GraphNodeType::KERNEL)     ? "KERNEL"
                             : (n.type == GraphNodeType::MEMCPY)     ? "MEMCPY"
                                                                      : "CONDITIONAL";
        j << "{\"nodeId\":" << n.nodeId
          << ",\"type\":\"" << typeStr << "\""
          << ",\"kernelId\":" << n.kernelId
          << ",\"kernelName\":\"" << jsonEscape(n.kernelName) << "\""
          << ",\"gridDim\":[" << n.gridDim.x << "," << n.gridDim.y << "," << n.gridDim.z << "]"
          << ",\"blockDim\":[" << n.blockDim.x << "," << n.blockDim.y << "," << n.blockDim.z << "]"
          << ",\"memcpyCount\":" << n.count
          << ",\"memcpyKind\":" << n.kind
          << ",\"condType\":" << static_cast<int>(n.condType)
          << ",\"bodyGraphId\":" << n.bodyGraphId
          << ",\"maxIterations\":" << n.maxIterations
          << ",\"deps\":[";
        for (size_t di = 0; di < n.deps.size(); ++di) {
            if (di > 0) j << ",";
            j << n.deps[di];
        }
        j << "]}";
    }
    j << "]}";

    outJson = j.str();
    VGRE_LOG_INFO("GraphManager", "Serialized graph " + std::to_string(id) +
                  " (" + std::to_string(g.nodes.size()) + " nodes, " +
                  std::to_string(outJson.size()) + " bytes)");
    return vgre::VGREResult::SUCCESS;
}

// ── Graph Deserialization ──────────────────────────────────────────────────
// Minimal JSON parser — no external dependency required.

namespace {

// Skip whitespace and return a reference to the next non-space char.
static size_t skipWS(const std::string &s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
    return pos;
}

// Expect a literal character; advance past it or return false.
static bool expectChar(const std::string &s, size_t &pos, char c) {
    pos = skipWS(s, pos);
    if (pos >= s.size() || s[pos] != c) return false;
    ++pos;
    return true;
}

// Parse a quoted JSON string; pos must point at the opening '"'.
static bool parseString(const std::string &s, size_t &pos, std::string &out) {
    pos = skipWS(s, pos);
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
            switch (s[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        ++pos;
    }
    if (pos >= s.size()) return false;
    ++pos; // consume closing '"'
    return true;
}

// Parse an integer (uint64_t); pos points at the first digit or '-'.
static bool parseInt(const std::string &s, size_t &pos, uint64_t &out) {
    pos = skipWS(s, pos);
    if (pos >= s.size()) return false;
    size_t start = pos;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    if (pos == start) return false;
    out = std::stoull(s.substr(start, pos - start));
    return true;
}

} // namespace

vgre::VGREResult GraphManager::deserializeGraph(const std::string &json, GraphId &outId) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t p = 0;
    if (!expectChar(json, p, '{')) return vgre::VGREResult::ERR_INVALID_VALUE;

    auto newGraph = std::make_shared<Graph>();
    newGraph->id = nextGraphId_++;

    // Parse top-level fields until we hit the nodes array.
    while (p < json.size() && json[p] != '}') {
        std::string key;
        if (!parseString(json, p, key)) return vgre::VGREResult::ERR_INVALID_VALUE;
        if (!expectChar(json, p, ':'))  return vgre::VGREResult::ERR_INVALID_VALUE;

        if (key == "nextNodeId") {
            uint64_t v = 0;
            if (!parseInt(json, p, v)) return vgre::VGREResult::ERR_INVALID_VALUE;
            newGraph->nextNodeId = v;
        } else if (key == "id") {
            uint64_t v = 0; parseInt(json, p, v); // original id — ignored, we assign a new one
        } else if (key == "nodes") {
            // Parse array of node objects
            if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
            p = skipWS(json, p);
            while (p < json.size() && json[p] != ']') {
                if (!expectChar(json, p, '{')) return vgre::VGREResult::ERR_INVALID_VALUE;

                GraphNode node;
                while (p < json.size() && json[p] != '}') {
                    std::string nk;
                    if (!parseString(json, p, nk)) return vgre::VGREResult::ERR_INVALID_VALUE;
                    if (!expectChar(json, p, ':'))  return vgre::VGREResult::ERR_INVALID_VALUE;

                    if (nk == "nodeId") {
                        uint64_t v = 0; parseInt(json, p, v); node.nodeId = v;
                    } else if (nk == "type") {
                        std::string tv; parseString(json, p, tv);
                        if (tv == "KERNEL") node.type = GraphNodeType::KERNEL;
                        else if (tv == "CONDITIONAL") node.type = GraphNodeType::CONDITIONAL;
                        else node.type = GraphNodeType::MEMCPY;
                    } else if (nk == "kernelId") {
                        uint64_t v = 0; parseInt(json, p, v); node.kernelId = static_cast<KernelId>(v);
                    } else if (nk == "kernelName") {
                        parseString(json, p, node.kernelName);
                    } else if (nk == "gridDim" || nk == "blockDim") {
                        // parse [x,y,z]
                        if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
                        uint64_t x = 0, y = 0, z = 0;
                        parseInt(json, p, x); expectChar(json, p, ',');
                        parseInt(json, p, y); expectChar(json, p, ',');
                        parseInt(json, p, z); expectChar(json, p, ']');
                        if (nk == "gridDim")  node.gridDim  = {static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
                        else                   node.blockDim = {static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
                    } else if (nk == "memcpyCount") {
                        uint64_t v = 0; parseInt(json, p, v); node.count = static_cast<size_t>(v);
                    } else if (nk == "memcpyKind") {
                        uint64_t v = 0; parseInt(json, p, v); node.kind = static_cast<int>(v);
                    } else if (nk == "condType") {
                        uint64_t v = 0; parseInt(json, p, v);
                        node.condType = (v == 1) ? GraphCondType::WHILE : GraphCondType::IF;
                    } else if (nk == "bodyGraphId") {
                        uint64_t v = 0; parseInt(json, p, v); node.bodyGraphId = static_cast<GraphId>(v);
                    } else if (nk == "maxIterations") {
                        uint64_t v = 0; parseInt(json, p, v); node.maxIterations = static_cast<unsigned int>(v);
                    } else if (nk == "deps") {
                        if (!expectChar(json, p, '[')) return vgre::VGREResult::ERR_INVALID_VALUE;
                        p = skipWS(json, p);
                        while (p < json.size() && json[p] != ']') {
                            uint64_t dep = 0;
                            if (!parseInt(json, p, dep)) return vgre::VGREResult::ERR_INVALID_VALUE;
                            node.deps.push_back(dep);
                            p = skipWS(json, p);
                            if (p < json.size() && json[p] == ',') ++p;
                        }
                        expectChar(json, p, ']');
                    } else {
                        // Unknown field — skip a simple value (string or number)
                        p = skipWS(json, p);
                        if (p < json.size() && json[p] == '"') {
                            std::string tmp; parseString(json, p, tmp);
                        } else {
                            while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
                        }
                    }

                    p = skipWS(json, p);
                    if (p < json.size() && json[p] == ',') ++p;
                    p = skipWS(json, p);
                }
                if (!expectChar(json, p, '}')) return vgre::VGREResult::ERR_INVALID_VALUE;
                newGraph->nodes.push_back(std::move(node));
                p = skipWS(json, p);
                if (p < json.size() && json[p] == ',') ++p;
                p = skipWS(json, p);
            }
            expectChar(json, p, ']');
        } else {
            // Unknown top-level key — skip value
            p = skipWS(json, p);
            if (p < json.size() && json[p] == '"') {
                std::string tmp; parseString(json, p, tmp);
            } else {
                while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
            }
        }

        p = skipWS(json, p);
        if (p < json.size() && json[p] == ',') ++p;
        p = skipWS(json, p);
    }

    graphs_[newGraph->id] = newGraph;
    outId = newGraph->id;

    VGRE_LOG_INFO("GraphManager", "Deserialized graph -> id=" + std::to_string(outId) +
                  " (" + std::to_string(newGraph->nodes.size()) + " nodes)");
    return vgre::VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
