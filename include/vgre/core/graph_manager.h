#ifndef VGRE_CORE_GRAPH_MANAGER_H
#define VGRE_CORE_GRAPH_MANAGER_H

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

enum class GraphNodeType { KERNEL, MEMCPY, CONDITIONAL };

// Condition type for conditional nodes (matches cudaGraphCondType)
enum class GraphCondType : uint8_t {
    IF    = 0,  // Body subgraph executes once if condition returns non-zero
    WHILE = 1,  // Body subgraph executes repeatedly while condition returns non-zero
};

struct GraphNode {
  GraphNodeType type;
  uint64_t nodeId = 0;
  std::vector<uint64_t> deps;

  // Stream this node was captured on.  Fusion is only permitted between nodes
  // captured on the same stream (different streams imply independent ordering
  // that a fused kernel would inadvertently serialise).  0 = default stream.
  StreamId streamId = 0;

  // Kernel data
  KernelId kernelId = 0;
  std::string kernelName;
  dim3 gridDim = {1, 1, 1};
  dim3 blockDim = {1, 1, 1};
  std::vector<std::vector<uint8_t>> capturedArgs;  // deep-copied arg data
  std::vector<void *> capturedWritePtrs;            // pointer args (potential write targets)
  std::vector<void *> capturedReadPtrs;             // pointer args (potential read sources)

  // Memcpy data
  void *dst = nullptr;
  void *src = nullptr;
  size_t count = 0;
  int kind = VGRE_MEMCPY_HOST_TO_DEVICE;

  // Conditional node data
  // condFn(condCtx) returns non-zero to execute / continue body, zero to skip/stop.
  int (*condFn)(void *) = nullptr;
  void *condCtx = nullptr;
  GraphId bodyGraphId = 0;     // subgraph to execute when condition is true
  GraphCondType condType = GraphCondType::IF;
  unsigned int maxIterations = 65536;  // safety limit for WHILE loops
};

class Graph {
public:
  GraphId id;
  uint64_t nextNodeId = 1;
  std::vector<GraphNode> nodes;
};

// Per-execution profiling data stored in GraphExec and updated by launch().
struct GraphExecProfile {
  double   last_exec_ms  = 0.0;  // wall time of the most recent launch
  double   total_exec_ms = 0.0;  // cumulative wall time across all launches
  uint64_t launch_count  = 0;    // number of times this exec was launched

  double avg_exec_ms() const {
    return launch_count > 0 ? total_exec_ms / static_cast<double>(launch_count) : 0.0;
  }
};

class GraphExec {
public:
  GraphExecId id;
  std::shared_ptr<Graph> sourceGraph;
  GraphExecProfile profile;
};

// Set by RuntimeEngine before calling addKernelNodeWithDepsOut so the capturing
// stream ID propagates into GraphNode::streamId without changing every overload.
extern thread_local StreamId g_capture_stream_id;

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
  vgre::VGREResult addKernelNodeWithDeps(GraphId id, KernelId kernelId,
                                        const std::string &name,
                                        const dim3 &grid, const dim3 &block,
                                        void **args,
                                        const std::vector<ArgType> &argTypes,
                                        const std::vector<uint64_t> &deps);
  vgre::VGREResult addKernelNodeWithDepsOut(
      GraphId id, KernelId kernelId, const std::string &name, const dim3 &grid,
      const dim3 &block, void **args, const std::vector<ArgType> &argTypes,
      const std::vector<uint64_t> &deps, uint64_t &outNodeId);

  vgre::VGREResult addMemcpyNode(GraphId id, void *dst, void *src, size_t count,
                                 int kind);
  vgre::VGREResult addMemcpyNodeWithDeps(GraphId id, void *dst, void *src,
                                        size_t count, int kind,
                                        const std::vector<uint64_t> &deps);
  vgre::VGREResult addMemcpyNodeWithDepsOut(GraphId id, void *dst, void *src,
                                           size_t count, int kind,
                                           const std::vector<uint64_t> &deps,
                                           uint64_t &outNodeId);

  vgre::VGREResult addConditionalNode(GraphId id, int (*condFn)(void *),
                                      void *condCtx, GraphId bodyGraphId,
                                      GraphCondType condType,
                                      unsigned int maxIterations = 65536);
  vgre::VGREResult addConditionalNodeWithDeps(GraphId id, int (*condFn)(void *),
                                              void *condCtx, GraphId bodyGraphId,
                                              GraphCondType condType,
                                              unsigned int maxIterations,
                                              const std::vector<uint64_t> &deps);
  vgre::VGREResult addConditionalNodeWithDepsOut(GraphId id, int (*condFn)(void *),
                                                 void *condCtx, GraphId bodyGraphId,
                                                 GraphCondType condType,
                                                 unsigned int maxIterations,
                                                 const std::vector<uint64_t> &deps,
                                                 uint64_t &outNodeId);

  // Read-only access to graph nodes; used by RuntimeEngine to compile body subgraphs.
  vgre::VGREResult getGraphNodes(GraphId id, std::vector<GraphNode> &outNodes) const;

  vgre::VGREResult addDependency(GraphId id, uint64_t nodeId,
                                 uint64_t dependsOn);
  vgre::VGREResult updateKernelNodeArgs(GraphId id, uint64_t nodeId,
                                        void **args,
                                        const std::vector<ArgType> &argTypes);
  vgre::VGREResult updateMemcpyNode(GraphId id, uint64_t nodeId, void *dst,
                                    void *src, size_t count, int kind);

  vgre::VGREResult cloneGraph(GraphId srcId, GraphId &outCloneId);

  // Serialize a graph to a JSON string.  Only the topology (node types, kernel
  // names, grid/block dims, deps) is persisted — raw pointer args and live
  // memory regions are intentionally excluded because they are process-specific.
  // Returns SUCCESS and populates outJson, or ERR_INVALID_VALUE if id not found.
  vgre::VGREResult serializeGraph(GraphId id, std::string &outJson) const;

  // Deserialize a previously-serialized graph JSON into a new GraphId.
  // Kernel source code for KERNEL nodes must be re-registered separately before
  // the graph can be launched (the JSON stores kernelName but not IR/source).
  // Returns SUCCESS and populates outId, or ERR_INVALID_VALUE on parse error.
  vgre::VGREResult deserializeGraph(const std::string &json, GraphId &outId);

  // Validate graph structure: checks deps, dimensions, and pointer sanity.
  // Can be called any time after nodes are added — does NOT require instantiation.
  vgre::VGREResult validateGraph(GraphId id) const;

  // Retrieve profiling data collected by launch().
  // Populates lastExecMs, avgExecMs, launchCount from the stored GraphExecProfile.
  vgre::VGREResult getExecProfile(GraphExecId id,
                                  double &lastExecMs,
                                  double &avgExecMs,
                                  uint64_t &launchCount) const;

  vgre::VGREResult instantiate(GraphId id, GraphExecId &outExecId);
  vgre::VGREResult updateExec(GraphExecId execId, GraphId newGraphId);
  vgre::VGREResult destroyGraphExec(GraphExecId id);
  vgre::VGREResult launch(GraphExecId execId, StreamId stream);

private:
  std::unordered_map<GraphId, std::shared_ptr<Graph>> graphs_;
  std::unordered_map<GraphExecId, std::shared_ptr<GraphExec>> executables_;
  GraphId nextGraphId_ = 1;
  GraphExecId nextExecId_ = 1;
  mutable std::recursive_mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_GRAPH_MANAGER_H
