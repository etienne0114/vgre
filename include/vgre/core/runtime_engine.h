#ifndef VGRE_CORE_RUNTIME_ENGINE_H
#define VGRE_CORE_RUNTIME_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <memory>
#include <mutex>
#include <string>
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

  VGREResult getKernelFromModule(ModuleHandle module, const std::string &name,
                                 KernelId &outId);

  VGREResult getModuleGlobal(ModuleHandle module, const std::string &name,
                             void *&outAddr, size_t &outSize);

  VGREResult getKernelArgTypes(KernelId id, std::vector<ArgType> &outTypes);
  const KernelIR *getKernelIR(KernelId id) const;

  VGREResult unloadModule(ModuleHandle module);

  VGREResult launchKernel(KernelId id, const dim3 &gridDim,
                          const dim3 &blockDim, void **args,
                          size_t sharedMem = 0, StreamId stream = 0);

  // Convenience: register + launch in one call
  VGREResult launchKernel(const std::string &name, const std::string &source,
                          const dim3 &gridDim, const dim3 &blockDim,
                          void **args, size_t sharedMem = 0,
                          StreamId stream = 0);

  // Cooperative kernel launch: grid-wide barrier via serialized block phases
  VGREResult launchCooperativeKernel(KernelId id, const dim3 &gridDim,
                                     const dim3 &blockDim, void **args,
                                     size_t sharedMem = 0, StreamId stream = 0);

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
  VGREResult streamBeginCapture(StreamId stream);
  VGREResult streamEndCapture(StreamId stream, GraphId &outGraph);
  VGREResult graphInstantiate(GraphId graph, GraphExecId &outExec);
  VGREResult graphUpdateExec(GraphExecId exec, GraphId graph);
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
  VGREResult graphAddDependency(GraphId graph, uint64_t nodeId,
                                uint64_t dependsOn);
  VGREResult graphUpdateKernelNode(GraphId graph, uint64_t nodeId, void **args,
                                   const std::vector<ArgType> &argTypes);
  VGREResult graphUpdateMemcpyNode(GraphId graph, uint64_t nodeId, void *dst,
                                   void *src, size_t count, int kind);
  
  // Internal Graph Recording
  bool isStreamCapturing(StreamId stream) const;
  VGREResult recordMemcpyToGraph(StreamId stream, void *dst, const void *src, size_t count, int kind);

  // Access sub-systems
  MemoryManager &getMemoryManager();
  Scheduler &getScheduler();
  VirtualGPUDevice &getDevice();
  VirtualGPUDevice &getDevice(DeviceId id);

  // Singleton
  static RuntimeEngine &instance();

private:
  bool initialized_ = false;
  std::vector<std::unique_ptr<VirtualGPUDevice>> devices_;
  DeviceId currentDeviceId_ = 0;
  std::unique_ptr<MemoryManager> memoryManager_;
  Scheduler *scheduler_ = nullptr;
  std::unique_ptr<compiler::KernelParser> parser_;
  std::unique_ptr<compiler::LLVMTranslationEngine> translator_;
  std::unique_ptr<runtime::CPUParallelExecutor> executor_;
  std::unique_ptr<runtime::VectorEngine> vectorEngine_;
  std::unique_ptr<GraphManager> graphManager_;

  // Capture state: streamId -> graphId
  std::unordered_map<StreamId, GraphId> captureState_;

  // Kernel cache: hash(source) → compiled function
  std::unordered_map<KernelId, CompiledKernelFn> kernelCache_;
  std::unordered_map<KernelId, KernelIR> kernelIRCache_;
  KernelId nextKernelId_ = 1;
  std::unordered_set<KernelId> warnedSyncthreads_;
  mutable std::recursive_mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_RUNTIME_ENGINE_H
