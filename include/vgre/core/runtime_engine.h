#ifndef VGRE_CORE_RUNTIME_ENGINE_H
#define VGRE_CORE_RUNTIME_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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

// ── Runtime Engine (top-level orchestrator) ────────────────────────────────
class RuntimeEngine {
public:
  RuntimeEngine();
  ~RuntimeEngine();

  // Initialization
  VGREResult initialize();
  VGREResult shutdown();
  bool isInitialized() const;

  // Kernel management
  VGREResult registerKernel(const std::string &name, const std::string &source,
                            KernelId &outId);

  VGREResult loadModule(const std::string &path, ModuleHandle &outModule);

  VGREResult getKernelFromModule(ModuleHandle module, const std::string &name,
                                 KernelId &outId);

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

  // Fused Dispatch
  VGREResult launchFusedKernelGroup(const std::vector<GraphNode>& group, StreamId stream);

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
  VGREResult streamBeginCapture(StreamId stream);
  VGREResult streamEndCapture(StreamId stream, GraphId &outGraph);
  VGREResult graphInstantiate(GraphId graph, GraphExecId &outExec);
  VGREResult graphLaunch(GraphExecId exec, StreamId stream);
  VGREResult graphDestroy(GraphId graph);
  VGREResult graphExecDestroy(GraphExecId exec);
  
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
  mutable std::mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_RUNTIME_ENGINE_H
