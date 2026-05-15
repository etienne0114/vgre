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
  node.streamId = g_capture_stream_id;  // set by RuntimeEngine before this call
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
  // Update node index map for O(1) lookup
  it->second->nodeIndex[outNodeId] = it->second->nodes.size() - 1;
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
  // Update node index map for O(1) lookup
  it->second->nodeIndex[outNodeId] = it->second->nodes.size() - 1;
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
  // Update node index map for O(1) lookup
  it->second->nodeIndex[outNodeId] = it->second->nodes.size() - 1;
  VGRE_LOG_INFO("GraphManager",
                "Added CONDITIONAL node " + std::to_string(outNodeId) +
                    " (type=" +
                    (condType == GraphCondType::IF ? "IF" :
                     (condType == GraphCondType::WHILE ? "WHILE" : "SWITCH")) +
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

  // Use nodeIndex map for O(1) lookup instead of O(n) linear search
  auto nodeIt = it->second->nodeIndex.find(nodeId);
  auto depIt = it->second->nodeIndex.find(dependsOn);
  if (nodeIt == it->second->nodeIndex.end() || depIt == it->second->nodeIndex.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = &it->second->nodes[nodeIt->second];

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

  // Use nodeIndex map for O(1) lookup instead of O(n) linear search
  auto nodeIt = it->second->nodeIndex.find(nodeId);
  if (nodeIt == it->second->nodeIndex.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = &it->second->nodes[nodeIt->second];
  if (node->type != GraphNodeType::KERNEL)
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

  // Use nodeIndex map for O(1) lookup instead of O(n) linear search
  auto nodeIt = it->second->nodeIndex.find(nodeId);
  if (nodeIt == it->second->nodeIndex.end())
    return vgre::VGREResult::ERR_INVALID_VALUE;

  GraphNode *node = &it->second->nodes[nodeIt->second];
  if (node->type != GraphNodeType::MEMCPY)
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

  // Deep-clone the template graph into a working copy owned by GraphExec.
  // This ensures that exec-time mutations (execKernelNodeSetParams, etc.)
  // do not pollute the template graph that may be re-instantiated later.
  auto workingCopy = std::make_shared<Graph>();
  workingCopy->id = it->second->id;
  workingCopy->nextNodeId = it->second->nextNodeId;
  workingCopy->nodes = it->second->nodes;       // value-copy (deep for vectors)
  workingCopy->nodeIndex = it->second->nodeIndex;

  auto exec = std::make_shared<GraphExec>();
  exec->id = nextExecId_++;
  exec->sourceGraph = workingCopy;

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

vgre::VGREResult GraphManager::updateExecV2(GraphExecId execId, GraphId newGraphId,
                                             const std::vector<uint64_t>& nodeIds) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto execIt = executables_.find(execId);
  if (execIt == executables_.end()) return vgre::VGREResult::ERR_INVALID_VALUE;

  auto graphIt = graphs_.find(newGraphId);
  if (graphIt == graphs_.end()) return vgre::VGREResult::ERR_INVALID_VALUE;

  auto& oldGraph = execIt->second->sourceGraph;
  auto& newGraph = graphIt->second;
  if (!oldGraph || !newGraph) return vgre::VGREResult::ERR_INVALID_VALUE;

  for (uint64_t nodeId : nodeIds) {
    auto oldNit = oldGraph->nodeIndex.find(nodeId);
    auto newNit = newGraph->nodeIndex.find(nodeId);
    if (oldNit == oldGraph->nodeIndex.end() || newNit == newGraph->nodeIndex.end()) {
        VGRE_LOG_WARN("GraphManager", "updateExecV2: nodeId " + std::to_string(nodeId) +
                      " not found in old or new graph.");
        return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    auto& oldNode = oldGraph->nodes[oldNit->second];
    auto& newNode = newGraph->nodes[newNit->second];
    if (oldNode.type != newNode.type) {
        VGRE_LOG_WARN("GraphManager", "updateExecV2: node type mismatch for nodeId " +
                      std::to_string(nodeId));
        return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    if (oldNode.deps.size() != newNode.deps.size()) {
        VGRE_LOG_WARN("GraphManager", "updateExecV2: dependency count mismatch for nodeId " +
                      std::to_string(nodeId));
        return vgre::VGREResult::ERR_INVALID_VALUE;
    }
    // Copy mutable fields depending on node type
    oldNode = newNode;
  }

  VGRE_LOG_INFO("GraphManager",
      "updateExecV2: updated " + std::to_string(nodeIds.size()) + " nodes in exec " +
      std::to_string(execId));
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraphExec(GraphExecId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  executables_.erase(id);
  return vgre::VGREResult::SUCCESS;
}


} // namespace core
} // namespace vgre
