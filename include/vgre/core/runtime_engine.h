#ifndef VGRE_CORE_RUNTIME_ENGINE_H
#define VGRE_CORE_RUNTIME_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/robin_hood_hash_table.h"

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace vgre {

// Forward declarations
namespace compiler {
class KernelParser;
class LLVMTranslationEngine;
} // namespace compiler
namespace core {
struct GraphNode;
class Scheduler;
class MemoryManager;
class VirtualGPUDevice;
class GraphManager;
} // namespace core
namespace runtime {
class CPUParallelExecutor;
class VectorEngine;
} // namespace runtime

namespace core {

/**
 * @brief Top-level orchestrator for the VGRE Engine.
 *
 * The RuntimeEngine manages device lifecycle, kernel compilation/dispatch,
 * and unified virtual memory across iGPU and CPU backends.
 */
class RuntimeEngine {
public:
  RuntimeEngine();
  ~RuntimeEngine();

  /**
   * @brief Bootstraps the VGRE environment, sensing hardware and initializing backends.
   * @return VGREResult::SUCCESS on success.
   */
  VGREResult initialize();

  /**
   * @brief Synchronously shuts down all executors and releases hardware resources.
   */
  VGREResult shutdown();
  
  bool isInitialized() const;

  /**
   * @brief Transpiles and JIT-compiles a CUDA kernel for the current backend.
   * @param name Symbolic name of the kernel.
   * @param source CUDA C++ source code.
   * @param outId Handle to the registered kernel.
   */
  VGREResult registerKernel(const std::string &name, const std::string &source,
                            KernelId &outId);

  VGREResult loadModule(const std::string &path, ModuleHandle &outModule);

  // Register an already-JIT-compiled kernel function directly (Track N bitcode path).
  VGREResult registerPrecompiledKernel(const std::string &name,
                                        CompiledKernelFn fn, KernelId &outId);

  VGREResult getKernelFromModule(ModuleHandle module, const std::string &name,
                                 KernelId &outId);

  VGREResult getModuleGlobal(ModuleHandle module, const std::string &name,
                             void *&outAddr, size_t &outSize);

  VGREResult getKernelArgTypes(KernelId id, std::vector<ArgType> &outTypes);
  const KernelIR *getKernelIR(KernelId id) const;
  // Look up KernelIR by compiled function pointer (used for iGPU fallback).
  const KernelIR *getKernelIRByFn(CompiledKernelFn fn) const;
  // CDP: look up a kernel by name (returns 0 if not found).
  KernelId lookupKernelIdByName(const char* name) const;
  // CDP: look up by compiled function address (scans kernelFnAddrMap_).
  KernelId lookupKernelIdByFn(void* fnPtr) const;

  /**
   * @brief Dynamically fuses multiple kernels into a single JIT unit.
   * @param ids Ordered list of kernel IDs to fuse.
   * @param outFusedId New kernel ID representing the fused unit.
   */
  VGREResult fuseKernels(const std::vector<KernelId> &ids, KernelId &outFusedId, std::string* outName = nullptr);

  VGREResult unloadModule(ModuleHandle module);

  // Block until JIT compilation for kernel `id` completes and the compiled
  // function is stored in kernelCache_. Safe to call concurrently; idempotent.
  // Used by the cluster dispatch path to pre-compile before sending LAUNCH_KERNEL
  // to workers — eliminates cold-JIT latency from the cluster wait critical path.
  VGREResult ensureKernelCompiled(KernelId id);

  VGREResult launchKernel(KernelId id, const dim3 &gridDim,
                          const dim3 &blockDim, void **args,
                          size_t sharedMem = 0, StreamId stream = 0,
                          const dim3 &gridOffset = dim3(0, 0, 0));

  // Convenience: register + launch in one call
  VGREResult launchKernel(const std::string &name, const std::string &source,
                          const dim3 &gridDim, const dim3 &blockDim,
                          void **args, size_t sharedMem = 0,
                          StreamId stream = 0,
                          const dim3 &gridOffset = dim3(0, 0, 0));

  // Cooperative kernel launch: grid-wide barrier via serialized block phases
  VGREResult launchCooperativeKernel(KernelId id, const dim3 &gridDim,
                                     const dim3 &blockDim, void **args,
                                     size_t sharedMem = 0, StreamId stream = 0,
                                     const dim3 &gridOffset = dim3(0, 0, 0));
  VGREResult launchCooperativeKernel(const std::string &name,
                                     const std::string &source,
                                     const dim3 &gridDim, const dim3 &blockDim,
                                     void **args, size_t sharedMem = 0,
                                     StreamId stream = 0);

  // Multi-device cooperative launch — all devices execute concurrently with
  // independent per-device grid-wide barriers (this_grid().sync() works within
  // each device). A start-gate ensures all devices begin at the same instant,
  // mirroring hardware behaviour where all GPUs dispatch simultaneously.
  struct CoopMultiLaunchParams {
    std::string name;          ///< Kernel symbol name
    std::string source;        ///< CUDA C++ source for JIT compilation
    dim3        gridDim;       ///< Grid dimensions for this device
    dim3        blockDim;      ///< Block dimensions for this device
    void      **args    = nullptr; ///< Kernel arguments
    size_t      sharedMem = 0;    ///< Dynamic shared memory in bytes
    StreamId    stream    = 0;    ///< Target stream (0 = default)
  };
  VGREResult launchCooperativeKernelMultiDevice(
      const std::vector<CoopMultiLaunchParams> &launchList);

  // Native Graph Dispatch
  VGREResult dispatchGraphNodes(const std::vector<GraphNode>& nodes, StreamId stream);

  // Device management
  int getDeviceCount() const;
  VGREResult setDevice(DeviceId id);
  DeviceId getDeviceId() const;
  VGREResult getDeviceProperties(DeviceId id, DeviceProperties &outProps);

  // Synchronization
  VGREResult synchronize();
  VGREResult streamSynchronize(StreamId stream);

  // Unified Virtual Memory
  VGREResult malloc(size_t size, MemoryHandle &outHandle);
  VGREResult mallocManaged(size_t size, MemoryHandle &outHandle,
                           unsigned int flags = 0);

  // Peer-to-Peer (P2P)
  VGREResult deviceCanAccessPeer(DeviceId device, DeviceId peerDevice,
                                 int *canAccess);
  VGREResult deviceEnablePeerAccess(DeviceId peerDevice);
  VGREResult deviceDisablePeerAccess(DeviceId peerDevice);

  // CUDA Graphs API
  VGREResult graphCreate(GraphId &outGraph);
  VGREResult graphClone(GraphId srcGraph, GraphId &outCloneGraph);
  VGREResult streamBeginCapture(StreamId stream);
  VGREResult streamBeginCaptureToGraph(StreamId stream, uint64_t graphHandle,
                                       const std::vector<uint64_t> &dependencies = {});
  VGREResult streamEndCapture(StreamId stream, GraphId &outGraph);
  VGREResult graphInstantiate(GraphId graph, GraphExecId &outExec);
  VGREResult graphUpdateExec(GraphExecId exec, GraphId graph);
  // Phase 10: cudaGraphExecUpdate_v2 — targeted update of specific nodes only.
  VGREResult graphUpdateExecV2(GraphExecId exec, GraphId graph,
                              const std::vector<uint64_t> &nodeIds);
  VGREResult graphLaunch(GraphExecId exec, StreamId stream);
  VGREResult graphDestroy(GraphId graph);
  VGREResult graphExecDestroy(GraphExecId exec);
  VGREResult graphAddKernelNode(GraphId graph, KernelId kernelId,
                                const std::string &name, const dim3 &grid,
                                const dim3 &block, void **args,
                                const std::vector<ArgType> &argTypes,
                                const std::vector<uint64_t> &deps,
                                uint64_t &outNodeId);
  VGREResult graphAddMemcpyNode(GraphId graph, void *dst, void *src,
                                size_t count, int kind,
                                const std::vector<uint64_t> &deps,
                                uint64_t &outNodeId);
  VGREResult graphAddConditionalNode(GraphId graph, int (*condFn)(void *),
                                     void *condCtx, GraphId bodyGraph,
                                     GraphCondType condType,
                                     unsigned int maxIterations,
                                     const std::vector<uint64_t> &deps,
                                     uint64_t &outNodeId);

  // ── New node types (P1.11-1.16) ─────────────────────────────────────────
  VGREResult graphAddMemsetNode(GraphId graph, void *dst, int value,
                                size_t pitch, size_t width, size_t height,
                                size_t depth,
                                const std::vector<uint64_t> &deps,
                                uint64_t &outNodeId);
  VGREResult graphAddHostNode(GraphId graph, void (*fn)(void *),
                              void *userData,
                              const std::vector<uint64_t> &deps,
                              uint64_t &outNodeId);
  VGREResult graphAddChildGraphNode(GraphId graph, GraphId childGraph,
                                    const std::vector<uint64_t> &deps,
                                    uint64_t &outNodeId);
  VGREResult graphAddEmptyNode(GraphId graph,
                               const std::vector<uint64_t> &deps,
                               uint64_t &outNodeId);
  VGREResult graphAddEventRecordNode(GraphId graph, void *event,
                                     const std::vector<uint64_t> &deps,
                                     uint64_t &outNodeId);
  VGREResult graphAddEventWaitNode(GraphId graph, void *event,
                                   const std::vector<uint64_t> &deps,
                                   uint64_t &outNodeId);
  VGREResult graphAddMemAllocNode(GraphId graph, size_t bytesize,
                                  void **outDevPtr,
                                  const std::vector<uint64_t> &deps,
                                  uint64_t &outNodeId);
  VGREResult graphAddMemFreeNode(GraphId graph, void *devPtr,
                                 const std::vector<uint64_t> &deps,
                                 uint64_t &outNodeId);

  // ── Graph introspection (P1.17) ──────────────────────────────────────────
  VGREResult graphGetNodes(GraphId graph,
                           std::vector<GraphNode> &outNodes) const;
  VGREResult graphGetRootNodes(GraphId graph,
                               std::vector<uint64_t> &outRoots) const;
  VGREResult graphGetEdges(GraphId graph,
                           std::vector<uint64_t> &fromNodes,
                           std::vector<uint64_t> &toNodes,
                           size_t &outCount) const;
  VGREResult graphNodeGetType(GraphId graph, uint64_t nodeId,
                              GraphNodeType &outType) const;
  VGREResult graphNodeGetDependencies(GraphId graph, uint64_t nodeId,
                                      std::vector<uint64_t> &outDeps,
                                      size_t &outCount) const;
  VGREResult graphNodeGetDependentNodes(GraphId graph, uint64_t nodeId,
                                        std::vector<uint64_t> &outDependents,
                                        size_t &outCount) const;
  VGREResult graphKernelNodeGetParams(GraphId graph, uint64_t nodeId,
                                      KernelId &outKid, std::string &outName,
                                      dim3 &outGrid, dim3 &outBlock,
                                      std::vector<std::vector<uint8_t>> &outArgs) const;
  VGREResult graphMemcpyNodeGetParams(GraphId graph, uint64_t nodeId,
                                      void *&outDst, void *&outSrc,
                                      size_t &outCount, int &outKind) const;
  VGREResult graphMemsetNodeGetParams(GraphId graph, uint64_t nodeId,
                                      void *&outDst, int &outValue,
                                      size_t &outPitch, size_t &outWidth,
                                      size_t &outHeight, size_t &outDepth) const;
  VGREResult graphHostNodeGetParams(GraphId graph, uint64_t nodeId,
                                    void (**outFn)(void *),
                                    void *&outUserData) const;

  // ── Exec mutation (P1.18) ────────────────────────────────────────────────
  VGREResult graphKernelNodeSetParams(GraphId graph, uint64_t nodeId,
                                      void **args,
                                      const std::vector<ArgType> &argTypes);
  VGREResult graphMemsetNodeSetParams(GraphId graph, uint64_t nodeId,
                                      void *dst, int value,
                                      size_t pitch, size_t width,
                                      size_t height, size_t depth);
  VGREResult graphHostNodeSetParams(GraphId graph, uint64_t nodeId,
                                    void (*fn)(void *), void *userData);
  VGREResult graphExecKernelNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                          void **args,
                                          const std::vector<ArgType> &argTypes);
  VGREResult graphExecMemcpyNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                          void *dst, void *src,
                                          size_t count, int kind);
  VGREResult graphExecMemsetNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                          void *dst, int value,
                                          size_t pitch, size_t width,
                                          size_t height, size_t depth);
  VGREResult graphExecHostNodeSetParams(GraphExecId execId, uint64_t nodeId,
                                        void (*fn)(void *), void *userData);
  VGREResult graphExecChildGraphNodeSetParams(GraphExecId execId,
                                              uint64_t nodeId,
                                              GraphId newChildGraph);
  VGREResult graphExecEventRecordNodeSetEvent(GraphExecId execId,
                                              uint64_t nodeId, void *event);
  VGREResult graphExecEventWaitNodeSetEvent(GraphExecId execId,
                                            uint64_t nodeId, void *event);
  VGREResult graphNodeSetEnabled(GraphExecId execId, uint64_t nodeId,
                                 bool enabled);
  VGREResult graphNodeGetEnabled(GraphExecId execId, uint64_t nodeId,
                                 bool &outEnabled) const;
  VGREResult graphExecGetFlags(GraphExecId execId, unsigned int &outFlags) const;
  VGREResult graphExecSetFlags(GraphExecId execId, unsigned int flags);
  VGREResult graphExecGetNodeType(GraphExecId execId, uint64_t nodeId,
                                  GraphNodeType &outType) const;

  // ── Dependency editing (P1.22) ───────────────────────────────────────────
  VGREResult graphAddDependency(GraphId graph, uint64_t nodeId,
                                uint64_t dependsOn);
  VGREResult graphAddDependencies(GraphId graph,
                                  const uint64_t *fromNodes,
                                  const uint64_t *toNodes, size_t numDeps);
  VGREResult graphRemoveDependencies(GraphId graph,
                                     const uint64_t *fromNodes,
                                     const uint64_t *toNodes, size_t numDeps);
  VGREResult graphNodeFindInClone(GraphId originalGraph,
                                  uint64_t originalNodeId,
                                  GraphId clonedGraph,
                                  uint64_t &outClonedNodeId) const;
  VGREResult graphDebugDotPrint(GraphId graph, const char *path,
                                unsigned int flags) const;

  VGREResult graphUpdateKernelNode(GraphId graph, uint64_t nodeId, void **args,
                                   const std::vector<ArgType> &argTypes);
  VGREResult graphUpdateMemcpyNode(GraphId graph, uint64_t nodeId, void *dst,
                                   void *src, size_t count, int kind);

  // ── Stream capture introspection (P1.19) ─────────────────────────────────
  bool isStreamCapturing(StreamId stream) const;
  bool getStreamCaptureInfo(StreamId stream, GraphId &outGraphId) const;
  bool getStreamCaptureInfoV2(StreamId stream, GraphId &outGraphId,
                              std::vector<uint64_t> &outDeps) const;
  VGREResult streamUpdateCaptureDependencies(StreamId stream,
                                             const std::vector<uint64_t> &deps,
                                             bool replace);
  VGREResult streamCopyAttributes(StreamId dst, StreamId src);

  VGREResult recordMemcpyToGraph(StreamId stream, void *dst, const void *src, size_t count, int kind);

  // Access sub-systems
  MemoryManager &getMemoryManager();
  Scheduler &getScheduler();
  VirtualGPUDevice &getDevice();
  VirtualGPUDevice &getDevice(DeviceId id);

  // Singleton
  static RuntimeEngine &instance();

  // Per-device helpers (returns scheduler/memory manager for the calling
  // thread's current device; must be called with initialized_ == true).
  Scheduler &currentScheduler();
  MemoryManager &currentMemoryManager();

private:
  bool initialized_ = false;
  std::vector<std::unique_ptr<VirtualGPUDevice>> devices_;

  // Thread-local current device: each OS thread has its own binding so that
  // cudaSetDevice(i) in one thread does not affect other threads.
  static thread_local DeviceId tlCurrentDeviceId_;

  // Per-device MemoryManagers (index == device id).
  std::vector<std::unique_ptr<MemoryManager>> deviceMemManagers_;

  // Per-device Scheduler raw pointers (index == device id).
  // deviceSchedulers_[0] == &Scheduler::instance() (global singleton).
  // deviceSchedulers_[1..N-1] point into ownedDeviceSchedulers_.
  std::vector<Scheduler*> deviceSchedulers_;
  // Owned schedulers for devices 1..N (device 0 uses the global singleton).
  std::vector<std::unique_ptr<Scheduler>> ownedDeviceSchedulers_;

  // Kept for backwards compat; always == deviceSchedulers_[0].
  Scheduler *scheduler_ = nullptr;
  std::unique_ptr<compiler::KernelParser> parser_;
  std::unique_ptr<compiler::LLVMTranslationEngine> translator_;
  std::unique_ptr<runtime::CPUParallelExecutor> executor_;
  std::unique_ptr<runtime::VectorEngine> vectorEngine_;
  std::unique_ptr<GraphManager> graphManager_;

  // Capture state: streamId -> graphId
  std::unordered_map<StreamId, GraphId> captureState_;
  std::unordered_map<StreamId, uint64_t> lastCapturedNodeId_;
  // Initial dependency frontier used by cudaStreamBeginCaptureToGraph.
  // Consumed by the first captured node on the stream, then cleared.
  std::unordered_map<StreamId, std::vector<uint64_t>> captureSeedDeps_;

  // Kernel caches
  std::unordered_map<KernelId, CompiledKernelFn> kernelCache_;
  std::unordered_map<KernelId, KernelIR> kernelIRCache_;
  std::unordered_map<void*, KernelId> kernelFnAddrMap_; // reverse map for CDP
  std::unordered_map<KernelId, JITFuture> pendingKernels_;
  RobinHoodHashTable<std::string, KernelId, vgre::common::SipHash24> kernelNames_;
  KernelId nextKernelId_ = 1;
  std::unordered_set<KernelId> warnedSyncthreads_;
  std::thread benchmarkThread_;
  mutable std::recursive_mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_RUNTIME_ENGINE_H
