#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include <cstring>

namespace vgre {
namespace core {

GraphManager::GraphManager() = default;
GraphManager::~GraphManager() = default;

vgre::VGREResult GraphManager::createGraph(GraphId &outId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto graph = std::make_shared<Graph>();
  graph->id = nextGraphId_++;
  graphs_[graph->id] = graph;
  outId = graph->id;
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraph(GraphId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  graphs_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addKernelNode(
    GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
    const dim3 &block, void **args, const std::vector<ArgType> &argTypes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (grid.x == 0 || block.x == 0)
    return vgre::VGREResult::ERROR_INVALID_VALUE;
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERROR_INVALID_VALUE;

  GraphNode node;
  node.type = GraphNodeType::KERNEL;
  node.kernelId = kernelId;
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
    }

    std::vector<uint8_t> buf(size, 0);
    if (size > 0 && args && args[i]) {
      std::memcpy(buf.data(), args[i], size);
    } else if (size > 0) {
      return vgre::VGREResult::ERROR_INVALID_VALUE;
    }
    node.capturedArgs.push_back(buf);
  }

  it->second->nodes.push_back(node);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addMemcpyNode(GraphId id, void *dst, void *src,
                                             size_t count, int kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERROR_INVALID_VALUE;
  if (!dst || !src || count == 0)
    return vgre::VGREResult::ERROR_INVALID_VALUE;
  if (kind != VGRE_MEMCPY_HOST_TO_DEVICE && kind != VGRE_MEMCPY_DEVICE_TO_HOST &&
      kind != VGRE_MEMCPY_DEVICE_TO_DEVICE) {
    return vgre::VGREResult::ERROR_INVALID_VALUE;
  }

  GraphNode node;
  node.type = GraphNodeType::MEMCPY;
  node.dst = dst;
  node.src = src;
  node.count = count;
  node.kind = kind;

  it->second->nodes.push_back(node);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::instantiate(GraphId id, GraphExecId &outExecId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = graphs_.find(id);
  if (it == graphs_.end())
    return vgre::VGREResult::ERROR_INVALID_VALUE;

  auto exec = std::make_shared<GraphExec>();
  exec->id = nextExecId_++;
  exec->sourceGraph = it->second;

  // Node Fusing Optimization Pass
  for (const auto &node : exec->sourceGraph->nodes) {
    if (exec->fusedGroups.empty()) {
      exec->fusedGroups.push_back({node});
    } else {
      auto &lastGroup = exec->fusedGroups.back();
      const auto &lastNode = lastGroup.back();
      
      // We can fuse if they are both kernels and have the exact same Grid/Block topology.
      // This reduces dispatch overhead by grouping matching topologies into a single Executor submission.
      if (node.type == GraphNodeType::KERNEL && lastNode.type == GraphNodeType::KERNEL &&
          node.gridDim.x == lastNode.gridDim.x && node.gridDim.y == lastNode.gridDim.y && node.gridDim.z == lastNode.gridDim.z &&
          node.blockDim.x == lastNode.blockDim.x && node.blockDim.y == lastNode.blockDim.y && node.blockDim.z == lastNode.blockDim.z) {
        lastGroup.push_back(node);
      } else {
        exec->fusedGroups.push_back({node});
      }
    }
  }

  VGRE_LOG_INFO(
      "GraphManager",
      "Instantiated graph " + std::to_string(id) + " into executable " +
          std::to_string(outExecId) + " (Fused down to " +
          std::to_string(exec->fusedGroups.size()) + " dispatch chunks)");

  executables_[exec->id] = exec;
  outExecId = exec->id;

  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraphExec(GraphExecId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  executables_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::launch(GraphExecId execId, StreamId stream) {
  std::shared_ptr<GraphExec> exec;
  std::vector<std::vector<GraphNode>> groups;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executables_.find(execId);
    if (it == executables_.end())
      return vgre::VGREResult::ERROR_INVALID_VALUE;
    exec = it->second;
    groups = exec->fusedGroups;
  }

  auto &engine = RuntimeEngine::instance();
  VGRE_LOG_INFO("GraphManager", "Launching graph executable " +
                                    std::to_string(execId) + " on stream " +
                                    std::to_string(stream));

  for (const auto &group : groups) {
    if (group.empty()) continue;

    if (group[0].type == GraphNodeType::KERNEL) {
      if (group.size() == 1) {
        // Fast path for single kernel to avoid overhead of creating vectors
        const auto &node = group[0];
        struct OwnedLaunchArgs {
          std::vector<std::vector<uint8_t>> ownedData;
          std::vector<void *> argPtrs;
        };
        auto launchArgs = std::make_shared<OwnedLaunchArgs>();
        launchArgs->ownedData.reserve(node.capturedArgs.size());
        launchArgs->argPtrs.reserve(node.capturedArgs.size());
        for (const auto &buf : node.capturedArgs) {
          launchArgs->ownedData.push_back(buf); // deep copy
          launchArgs->argPtrs.push_back(launchArgs->ownedData.back().data());
        }

        auto r = engine.launchKernel(node.kernelId, node.gridDim, node.blockDim,
                                     launchArgs->argPtrs.data(), 0, stream);
        if (r != vgre::VGREResult::SUCCESS)
          return r;

        auto holdFut = engine.getScheduler().submitStreamTask(
            stream, [launchArgs]() { (void)launchArgs; });
      } else {
        // Dispatch fused multi-kernel group
        auto r = engine.launchFusedKernelGroup(group, stream);
        if (r != vgre::VGREResult::SUCCESS)
          return r;
      }
    } else if (group[0].type == GraphNodeType::MEMCPY) {
      // Memory copies cannot currently be fused easily due to different sizes/ptrs
      for (const auto &node : group) {
        auto &mm = engine.getMemoryManager();
        auto dst = node.dst;
        auto src = node.src;
        auto count = node.count;
        auto kind = node.kind;

        auto memcpyFut = engine.getScheduler().submitStreamTask(
            stream, [&mm, dst, src, count, kind]() {
              if (kind == VGRE_MEMCPY_HOST_TO_DEVICE) {
                mm.copyHostToDevice(dst, src, count);
              } else if (kind == VGRE_MEMCPY_DEVICE_TO_HOST) {
                mm.copyDeviceToHost(dst, src, count);
              } else if (kind == VGRE_MEMCPY_DEVICE_TO_DEVICE) {
                mm.copyDeviceToDevice(dst, src, count);
              }
            });
        if (memcpyFut.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
          auto memcpyRes = memcpyFut.get();
          if (memcpyRes != vgre::VGREResult::SUCCESS)
            return memcpyRes;
        }
      }
    }
  }

  return vgre::VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
