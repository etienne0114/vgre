#ifndef VGRE_CORE_GRAPH_MANAGER_H
#define VGRE_CORE_GRAPH_MANAGER_H

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

enum class GraphNodeType {
    KERNEL,
    MEMCPY,
    CONDITIONAL,
    MEMSET,       // cudaGraphAddMemsetNode
    HOST,         // cudaGraphAddHostNode
    CHILD,        // cudaGraphAddChildGraphNode
    EMPTY,        // cudaGraphAddEmptyNode
    EVENT_RECORD, // cudaGraphAddEventRecordNode
    EVENT_WAIT,   // cudaGraphAddEventWaitNode
    MEMALLOC,     // cudaGraphAddMemAllocNode
    MEMFREE,      // cudaGraphAddMemFreeNode
};

// Condition type for conditional nodes (matches cudaGraphCondType)
enum class GraphCondType : uint8_t {
    IF     = 0,  // Body executes once if condition returns non-zero
    WHILE  = 1,  // Body executes repeatedly while condition returns non-zero
    SWITCH = 2,  // Condition returns int [0..N); selects one of N child subgraphs
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

  // MEMSET node data (pitched 2D/3D fill)
  // dst is reused from the memcpy field above.
  int memsetValue = 0;     // fill byte value (cast to unsigned char)
  size_t memsetPitch = 0;  // row stride in bytes (>= memsetWidth)
  size_t memsetWidth = 0;  // bytes to write per row
  size_t memsetHeight = 0; // number of rows
  size_t memsetDepth = 1;  // number of depth slices (1 = 2D, >1 = 3D)

  // HOST node data
  void (*hostFn)(void *) = nullptr;
  void *hostUserData = nullptr;

  // CHILD node data: child graph to run inline (reuses bodyGraphId)

  // EVENT_RECORD / EVENT_WAIT node data
  void *eventHandle = nullptr;  // cudaEvent_t (Event *)

  // MEMALLOC node data
  size_t memAllocSize = 0;
  void  *memAllocPtr  = nullptr;  // pre-allocated backing pointer (set at node-add time)
  void **memAllocOutPtr = nullptr; // where CUDA caller stores the pointer (runtime output)

  // MEMFREE node data: pointer to free (stored in dst above)
};

class Graph {
public:
  GraphId id;
  uint64_t nextNodeId = 1;
  std::vector<GraphNode> nodes;
  // Index map for O(1) node lookup by nodeId instead of O(n) linear search
  std::unordered_map<uint64_t, size_t> nodeIndex;
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
  unsigned int flags = 0;
  // Per-node enabled/disabled state: nodeId -> enabled (true = runs, false = skipped).
  // Nodes not in this map are enabled by default.
  std::unordered_map<uint64_t, bool> nodeEnabled;
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
  bool graphExists(GraphId id) const;

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

  // ── New node types (P1.11-1.16) ─────────────────────────────────────────
  vgre::VGREResult addMemsetNode(GraphId id, void *dst, int value, size_t pitch,
                                 size_t width, size_t height, size_t depth,
                                 const std::vector<uint64_t> &deps,
                                 uint64_t &outNodeId);

  vgre::VGREResult addHostNode(GraphId id, void (*fn)(void *), void *userData,
                               const std::vector<uint64_t> &deps,
                               uint64_t &outNodeId);

  vgre::VGREResult addChildGraphNode(GraphId id, GraphId childGraphId,
                                     const std::vector<uint64_t> &deps,
                                     uint64_t &outNodeId);

  vgre::VGREResult addEmptyNode(GraphId id, const std::vector<uint64_t> &deps,
                                uint64_t &outNodeId);

  vgre::VGREResult addEventRecordNode(GraphId id, void *event,
                                      const std::vector<uint64_t> &deps,
                                      uint64_t &outNodeId);

  vgre::VGREResult addEventWaitNode(GraphId id, void *event,
                                    const std::vector<uint64_t> &deps,
                                    uint64_t &outNodeId);

  vgre::VGREResult addMemAllocNode(GraphId id, size_t bytesize,
                                   void **outDevPtr,
                                   const std::vector<uint64_t> &deps,
                                   uint64_t &outNodeId);

  vgre::VGREResult addMemFreeNode(GraphId id, void *devPtr,
                                  const std::vector<uint64_t> &deps,
                                  uint64_t &outNodeId);

  // ── Read-only graph introspection ────────────────────────────────────────
  vgre::VGREResult getGraphNodes(GraphId id, std::vector<GraphNode> &outNodes) const;
  vgre::VGREResult getGraphNodeCount(GraphId id, size_t &outCount) const;

  // Returns all root nodes (nodes with no incoming deps within this graph).
  vgre::VGREResult getGraphRootNodes(GraphId id,
                                     std::vector<uint64_t> &outNodeIds) const;

  // Returns all edges as (from, to) pairs.
  vgre::VGREResult getGraphEdges(GraphId id,
                                 std::vector<uint64_t> &fromNodes,
                                 std::vector<uint64_t> &toNodes,
                                 size_t &outCount) const;

  vgre::VGREResult getNodeType(GraphId id, uint64_t nodeId,
                               GraphNodeType &outType) const;

  vgre::VGREResult getNodeDependencies(GraphId id, uint64_t nodeId,
                                       std::vector<uint64_t> &outDeps,
                                       size_t &outCount) const;

  vgre::VGREResult getNodeDependentNodes(GraphId id, uint64_t nodeId,
                                         std::vector<uint64_t> &outDependents,
                                         size_t &outCount) const;

  // Find the corresponding node in a cloned graph.
  vgre::VGREResult findNodeInClone(GraphId originalGraph, uint64_t originalNodeId,
                                   GraphId clonedGraph,
                                   uint64_t &outClonedNodeId) const;

  // Serialize to DOT format for debugging (cudaGraphDebugDotPrint).
  vgre::VGREResult debugDotPrint(GraphId id, const char *path,
                                 unsigned int flags) const;

  // ── Node parameter introspection ─────────────────────────────────────────
  vgre::VGREResult getKernelNodeParams(GraphId id, uint64_t nodeId,
                                       KernelId &outKernelId,
                                       std::string &outName,
                                       dim3 &outGrid, dim3 &outBlock,
                                       std::vector<std::vector<uint8_t>> &outArgs) const;

  vgre::VGREResult getMemcpyNodeParams(GraphId id, uint64_t nodeId,
                                       void *&outDst, void *&outSrc,
                                       size_t &outCount, int &outKind) const;

  vgre::VGREResult getMemsetNodeParams(GraphId id, uint64_t nodeId,
                                       void *&outDst, int &outValue,
                                       size_t &outPitch, size_t &outWidth,
                                       size_t &outHeight, size_t &outDepth) const;

  vgre::VGREResult getHostNodeParams(GraphId id, uint64_t nodeId,
                                     void (**outFn)(void *),
                                     void *&outUserData) const;

  // ── Exec-time node mutation ───────────────────────────────────────────────
  vgre::VGREResult execKernelNodeSetParams(GraphExecId execId,
                                           uint64_t nodeId,
                                           void **args,
                                           const std::vector<ArgType> &argTypes);

  vgre::VGREResult execMemcpyNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                           void *dst, void *src,
                                           size_t count, int kind);

  vgre::VGREResult execMemsetNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                           void *dst, int value,
                                           size_t pitch, size_t width,
                                           size_t height, size_t depth);

  vgre::VGREResult execHostNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                         void (*fn)(void *), void *userData);

  vgre::VGREResult execChildGraphNodeSetParams(GraphExecId execId,
                                               uint64_t nodeId,
                                               GraphId newChildGraph);

  vgre::VGREResult execEventRecordNodeSetEvent(GraphExecId execId,
                                               uint64_t nodeId, void *event);

  vgre::VGREResult execEventWaitNodeSetEvent(GraphExecId execId,
                                             uint64_t nodeId, void *event);

  vgre::VGREResult nodeSetEnabled(GraphExecId execId, uint64_t nodeId,
                                  bool enabled);
  vgre::VGREResult nodeGetEnabled(GraphExecId execId, uint64_t nodeId,
                                  bool &outEnabled) const;

  vgre::VGREResult getExecFlags(GraphExecId execId,
                                unsigned int &outFlags) const;
  vgre::VGREResult setExecFlags(GraphExecId execId, unsigned int flags);

  // ── Dependency editing ────────────────────────────────────────────────────
  vgre::VGREResult addDependency(GraphId id, uint64_t nodeId,
                                 uint64_t dependsOn);
  vgre::VGREResult removeDependency(GraphId id, uint64_t nodeId,
                                    uint64_t dependsOn);
  vgre::VGREResult addDependencies(GraphId id,
                                   const uint64_t *fromNodes,
                                   const uint64_t *toNodes,
                                   size_t numDeps);
  vgre::VGREResult removeDependencies(GraphId id,
                                      const uint64_t *fromNodes,
                                      const uint64_t *toNodes,
                                      size_t numDeps);

  vgre::VGREResult updateKernelNodeArgs(GraphId id, uint64_t nodeId,
                                        void **args,
                                        const std::vector<ArgType> &argTypes);
  vgre::VGREResult updateMemcpyNode(GraphId id, uint64_t nodeId, void *dst,
                                    void *src, size_t count, int kind);
  vgre::VGREResult updateMemsetNodeInGraph(GraphId id, uint64_t nodeId,
                                           void *dst, int value, size_t pitch,
                                           size_t width, size_t height, size_t depth);
  vgre::VGREResult updateHostNodeInGraph(GraphId id, uint64_t nodeId,
                                         void (*fn)(void *), void *userData);

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

  // Return the exec's mutable working copy of a graph node by node ID,
  // or nullptr if the node does not exist in the exec's source graph.
  GraphNode *findExecNode(GraphExecId execId, uint64_t nodeId);

  // Query the type of a node inside an exec's working graph.
  VGREResult getExecNodeType(GraphExecId execId, uint64_t nodeId,
                             GraphNodeType &outType) const;

private:
  std::unordered_map<GraphId, std::shared_ptr<Graph>> graphs_;
  std::unordered_map<GraphExecId, std::shared_ptr<GraphExec>> executables_;
  GraphId nextGraphId_ = 1;
  GraphExecId nextExecId_ = 1;
  mutable std::recursive_mutex mutex_;

  // ── Clone-tracking for cudaGraphNodeFindInClone ─────────────────────────
  // Maps (originalGraphId, originalNodeId) -> (clonedGraphId, clonedNodeId).
  // Populated by cloneGraph().
  std::unordered_map<GraphId, std::unordered_map<uint64_t, uint64_t>> cloneNodeMap_;
  // Maps cloned graph ID -> original graph ID (for reverse lookup).
  std::unordered_map<GraphId, GraphId> cloneOriginMap_;

  // ── MemAlloc backing store ───────────────────────────────────────────────
  // Tracks pointers allocated by MEMALLOC nodes so MEMFREE can release them.
  std::unordered_map<void *, size_t> memAllocRegistry_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_GRAPH_MANAGER_H
