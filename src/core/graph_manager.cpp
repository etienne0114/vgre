#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
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

  VGRE_LOG_INFO(
      "GraphManager",
      "Instantiated graph " + std::to_string(id) + " into native executable " +
          std::to_string(outExecId));

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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = executables_.find(execId);
    if (it == executables_.end())
      return vgre::VGREResult::ERROR_INVALID_VALUE;
    exec = it->second;
  }

  auto &engine = RuntimeEngine::instance();
  VGRE_LOG_INFO("GraphManager", "Dispatching real native DAG from Executable " +
                                    std::to_string(execId) + " on stream " +
                                    std::to_string(stream));

  // Native Graph Dispatch via RuntimeEngine 
  return engine.dispatchGraphNodes(exec->sourceGraph->nodes, stream);
}

} // namespace core
} // namespace vgre
