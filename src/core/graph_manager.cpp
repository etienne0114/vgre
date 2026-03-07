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
  executables_[exec->id] = exec;
  outExecId = exec->id;

  VGRE_LOG_INFO("GraphManager", "Instantiated graph " + std::to_string(id) +
                                    " into executable " +
                                    std::to_string(outExecId));
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraphExec(GraphExecId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  executables_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::launch(GraphExecId execId, StreamId stream) {
  std::shared_ptr<GraphExec> exec;
  std::vector<GraphNode> nodes;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executables_.find(execId);
    if (it == executables_.end())
      return vgre::VGREResult::ERROR_INVALID_VALUE;
    exec = it->second;
    nodes = exec->sourceGraph->nodes;
  }

  auto &engine = RuntimeEngine::instance();
  VGRE_LOG_INFO("GraphManager", "Launching graph executable " +
                                    std::to_string(execId) + " on stream " +
                                    std::to_string(stream));

  for (const auto &node : nodes) {
    if (node.type == GraphNodeType::KERNEL) {
      // Deep-copy captured arg data into a fully self-contained struct.
      // This ensures args survive even if the source Graph is destroyed
      // before async execution on this stream completes.
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

      // Keep launchArgs alive: submit a no-op follow-up task on the same stream
      // that captures the shared_ptr, ensuring the owned data lives until
      // after the kernel that uses these pointers has completed.
      auto holdFut = engine.getScheduler().submitStreamTask(
          stream, [launchArgs]() { (void)launchArgs; });
      if (holdFut.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        auto holdRes = holdFut.get();
        if (holdRes != vgre::VGREResult::SUCCESS)
          return holdRes;
      }
    } else if (node.type == GraphNodeType::MEMCPY) {
      auto &mm = engine.getMemoryManager();
      // Dispatch asynchronous memory copy on the stream.
      // Currently, we'll wrap synchronous copies inside an async task so it
      // sequences with the kernels correctly.
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

  return vgre::VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
