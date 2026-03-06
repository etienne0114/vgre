#ifndef VGRE_CORE_GRAPH_MANAGER_H
#define VGRE_CORE_GRAPH_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

enum class GraphNodeType { KERNEL, MEMCPY };

struct GraphNode {
  GraphNodeType type;

  // Kernel data
  KernelId kernelId;
  std::string kernelName;
  dim3 gridDim;
  dim3 blockDim;
  std::vector<std::vector<uint8_t>> capturedArgs; // deep-copied arg data

  // Memcpy data
  void *dst;
  const void *src;
  size_t count;
  MemcpyKind kind;
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

  vgre::VGREResult instantiate(GraphId id, GraphExecId &outExecId);
  vgre::VGREResult destroyGraphExec(GraphExecId id);
  vgre::VGREResult launch(GraphExecId execId, StreamId stream);

private:
  std::unordered_map<GraphId, std::shared_ptr<Graph>> graphs_;
  std::unordered_map<GraphExecId, std::shared_ptr<GraphExec>> executables_;
  GraphId nextGraphId_ = 1;
  GraphExecId nextExecId_ = 1;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_GRAPH_MANAGER_H
