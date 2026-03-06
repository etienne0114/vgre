#include "vgre/core/graph_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include <cstring>

namespace vgre {
namespace core {

GraphManager::GraphManager() = default;
GraphManager::~GraphManager() = default;

vgre::VGREResult GraphManager::createGraph(GraphId &outId) {
  auto graph = std::make_shared<Graph>();
  graph->id = nextGraphId_++;
  graphs_[graph->id] = graph;
  outId = graph->id;
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::destroyGraph(GraphId id) {
  graphs_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::addKernelNode(
    GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
    const dim3 &block, void **args, const std::vector<ArgType> &argTypes) {
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

    std::vector<uint8_t> buf(size);
    std::memcpy(buf.data(), args[i], size);
    node.capturedArgs.push_back(buf);
  }

  it->second->nodes.push_back(node);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::instantiate(GraphId id, GraphExecId &outExecId) {
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
  executables_.erase(id);
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult GraphManager::launch(GraphExecId execId, StreamId stream) {
  auto it = executables_.find(execId);
  if (it == executables_.end())
    return vgre::VGREResult::ERROR_INVALID_VALUE;

  auto &engine = RuntimeEngine::instance();
  VGRE_LOG_INFO("GraphManager", "Launching graph executable " +
                                    std::to_string(execId) + " on stream " +
                                    std::to_string(stream));

  for (const auto &node : it->second->sourceGraph->nodes) {
    if (node.type == GraphNodeType::KERNEL) {
      // Heap-allocate the args so they outlive the async dispatch.
      // The shared_ptr ensures cleanup after all kernels on this stream
      // complete.
      auto launchArgs = std::make_shared<std::vector<void *>>();
      for (const auto &buf : node.capturedArgs) {
        launchArgs->push_back(const_cast<uint8_t *>(buf.data()));
      }

      auto r = engine.launchKernel(node.kernelId, node.gridDim, node.blockDim,
                                   launchArgs->data(), 0, stream);
      if (r != vgre::VGREResult::SUCCESS)
        return r;

      // Keep launchArgs alive: submit a no-op follow-up task on the same stream
      // that captures the shared_ptr, ensuring the data lives until execution.
      engine.getScheduler().submitStreamTask(stream, [launchArgs]() {
        // This task exists solely to extend the lifetime of launchArgs
        // until after the kernel that uses this data has completed.
        (void)launchArgs;
      });
    }
  }

  return vgre::VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
