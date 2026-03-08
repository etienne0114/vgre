#ifndef VGRE_CORE_GRAPH_MANAGER_H
#define VGRE_CORE_GRAPH_MANAGER_H

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

enum class GraphNodeType { KERNEL, MEMCPY };

struct GraphNode {
  GraphNodeType type;

  // Kernel data
  KernelId kernelId = 0;
  std::string kernelName;
  dim3 gridDim = {1, 1, 1};
  dim3 blockDim = {1, 1, 1};
  std::vector<std::vector<uint8_t>> capturedArgs; // deep-copied arg data

  // Memcpy data
  void *dst = nullptr;
  void *src = nullptr;
  size_t count = 0;
  int kind = VGRE_MEMCPY_HOST_TO_DEVICE;
};

class Graph {
public:
  GraphId id;
  std::vector<GraphNode> nodes;
};

class GraphExec {
public:
  GraphExecId id;
  std::shared_ptr<Graph> sourceGraph;
};

class GraphManager {
public:
  GraphManager();
  ~GraphManager();

  vgre::VGREResult createGraph(GraphId &outId);
  vgre::VGREResult destroyGraph(GraphId id);

  vgre::VGREResult addKernelNode(GraphId id, KernelId kernelId,
                                 const std::string &name, const dim3 &grid,
                                 const dim3 &block, void **args,
                                 const std::vector<ArgType> &argTypes);

  vgre::VGREResult addMemcpyNode(GraphId id, void *dst, void *src, size_t count,
                                 int kind);

  vgre::VGREResult instantiate(GraphId id, GraphExecId &outExecId);
  vgre::VGREResult destroyGraphExec(GraphExecId id);
  vgre::VGREResult launch(GraphExecId execId, StreamId stream);

private:
  std::unordered_map<GraphId, std::shared_ptr<Graph>> graphs_;
  std::unordered_map<GraphExecId, std::shared_ptr<GraphExec>> executables_;
  GraphId nextGraphId_ = 1;
  GraphExecId nextExecId_ = 1;
  mutable std::mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_GRAPH_MANAGER_H
