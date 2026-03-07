#include "vgre/core/runtime_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/common/logger.h"
#include "vgre/compiler/kernel_parser.h"
#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/runtime/vector_engine.h"

#include <cstring>
#include <functional>

namespace vgre {
namespace core {

// ── Constructor / Destructor ───────────────────────────────────────────────
RuntimeEngine::RuntimeEngine() = default;
RuntimeEngine::~RuntimeEngine() {
  if (initialized_) {
    shutdown();
  }
}

// ── Initialization ─────────────────────────────────────────────────────────
VGREResult RuntimeEngine::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_)
    return VGREResult::SUCCESS;

  VGRE_LOG_INFO("RuntimeEngine", "Initializing VGRE Runtime Engine...");

  // Create sub-systems
  scheduler_ = &Scheduler::instance();
  parser_ = std::make_unique<compiler::KernelParser>();
  translator_ = std::make_unique<compiler::LLVMTranslationEngine>();
  executor_ = std::make_unique<runtime::CPUParallelExecutor>(
      scheduler_->getThreadCount());
  vectorEngine_ = std::make_unique<runtime::VectorEngine>();
  graphManager_ = std::make_unique<GraphManager>();

  // Create multiple virtual devices (e.g., 4 for complex topology testing)
  for (int i = 0; i < 4; ++i) {
    auto dev = std::make_unique<VirtualGPUDevice>(i);
    dev->detectHardware();

    // Customize topology: 2 pairs of devices, each pair on a different PCI bus
    auto props = dev->getProperties();
    props.pciBusId = (i < 2) ? 0 : 1;
    dev->setProperties(props);

    dev->createContext();
    devices_.push_back(std::move(dev));
  }

  currentDeviceId_ = 0;

  // Global memory manager (shared for now, pool sized to first device)
  memoryManager_ = std::make_unique<MemoryManager>(
      devices_[0]->getProperties().totalGlobalMem);

  initialized_ = true;

  // Register with global IPC service as a client by default.
  // The Dashboard will re-init as master.
  vgre::advanced::IPCManager::instance().initialize(false);

  VGRE_LOG_INFO("RuntimeEngine", "VGRE Runtime Engine initialized with " +
                                     std::to_string(devices_.size()) +
                                     " devices");
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::shutdown() {
  // First, drain all pending scheduler work BEFORE acquiring our mutex.
  // This prevents deadlocks where worker threads (completing tasks)
  // need RuntimeEngine::mutex_ while we're holding it during shutdown.
  if (scheduler_) {
    scheduler_->waitAll();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Shutting down VGRE Runtime Engine...");

  scheduler_ = nullptr;
  executor_.reset();
  vectorEngine_.reset();
  translator_.reset();
  parser_.reset();
  memoryManager_.reset();
  devices_.clear();

  kernelCache_.clear();
  kernelIRCache_.clear();
  initialized_ = false;

  VGRE_LOG_INFO("RuntimeEngine", "Shutdown complete");
  return VGREResult::SUCCESS;
}

// ── Device management ──────────────────────────────────────────────────────
int RuntimeEngine::getDeviceCount() const {
  return static_cast<int>(devices_.size());
}

VGREResult RuntimeEngine::setDevice(DeviceId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    return VGREResult::ERROR_INVALID_DEVICE;
  }
  currentDeviceId_ = id;
  VGRE_LOG_DEBUG("RuntimeEngine", "Switched to device " + std::to_string(id));
  return VGREResult::SUCCESS;
}

DeviceId RuntimeEngine::getDeviceId() const { return currentDeviceId_; }

VGREResult RuntimeEngine::getDeviceProperties(DeviceId id,
                                              DeviceProperties &outProps) {
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    return VGREResult::ERROR_INVALID_DEVICE;
  }
  outProps = devices_[id]->getProperties();
  return VGREResult::SUCCESS;
}

bool RuntimeEngine::isInitialized() const { return initialized_; }

// ── Kernel registration ────────────────────────────────────────────────────
VGREResult RuntimeEngine::registerKernel(const std::string &name,
                                         const std::string &source,
                                         KernelId &outId) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Registering kernel: " + name);

  // Parse kernel source
  KernelIR ir;
  auto r = parser_->parse(name, source, ir);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("RuntimeEngine", "Failed to parse kernel: " + name);
    return r;
  }

  // Translate to executable
  CompiledKernelFn fn;
  r = translator_->translate(ir, fn);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("RuntimeEngine", "Failed to translate kernel: " + name);
    return r;
  }

  KernelId id = nextKernelId_++;
  kernelCache_[id] = fn;
  kernelIRCache_[id] = ir;
  outId = id;

  VGRE_LOG_INFO("RuntimeEngine", "Kernel '" + name + "' registered with ID " +
                                     std::to_string(id));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::loadModule(const std::string &path,
                                     ModuleHandle &outModule) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Loading binary module: " + path);
  return translator_->loadBitcodeModule(path, outModule);
}

VGREResult RuntimeEngine::getKernelFromModule(ModuleHandle module,
                                              const std::string &name,
                                              KernelId &outId) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  CompiledKernelFn fn;
  auto r = translator_->getFunctionFromModule(module, name, fn);
  if (r != VGREResult::SUCCESS)
    return r;

  outId = nextKernelId_++;
  kernelCache_[outId] = fn;

  // Create metadata IR entry for binary module function.
  // We cannot parse the source from a pre-compiled binary, but we
  // preserve the function name and mark the origin for diagnostics.
  KernelIR ir;
  ir.name = name;
  ir.source = "__binary_module__"; // Sentinel: marks this as pre-compiled
  // Binary modules don't carry parsed arg types — they are resolved
  // at the JIT level via the symbol's calling convention.
  kernelIRCache_[outId] = ir;

  VGRE_LOG_INFO("RuntimeEngine", "Kernel '" + name +
                                     "' retrieved from module as ID " +
                                     std::to_string(outId));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::getKernelArgTypes(KernelId id,
                                            std::vector<ArgType> &outTypes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  auto it = kernelIRCache_.find(id);
  if (it == kernelIRCache_.end())
    return VGREResult::ERROR_INVALID_KERNEL;

  outTypes = it->second.argTypes;
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::unloadModule(ModuleHandle module) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  return translator_->unloadModule(module);
}

// ── Kernel launch (by ID) ──────────────────────────────────────────────────
VGREResult RuntimeEngine::launchKernel(KernelId id, const dim3 &gridDim,
                                       const dim3 &blockDim, void **args,
                                       size_t sharedMem, StreamId stream) {
  CompiledKernelFn fn;
  std::shared_ptr<std::vector<uint64_t>> argValues;
  std::shared_ptr<std::vector<void *>> safeArgs;

  // Critical section: lookup kernel and check capture state
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_)
      return VGREResult::ERROR_NOT_INITIALIZED;

    auto it = kernelCache_.find(id);
    if (it == kernelCache_.end()) {
      return VGREResult::ERROR_INVALID_KERNEL;
    }

    VGRE_LOG_INFO("RuntimeEngine", "Launching kernel " + std::to_string(id) +
                                       " grid(" + std::to_string(gridDim.x) +
                                       "," + std::to_string(gridDim.y) + "," +
                                       std::to_string(gridDim.z) + ")" +
                                       " block(" + std::to_string(blockDim.x) +
                                       "," + std::to_string(blockDim.y) + "," +
                                       std::to_string(blockDim.z) + ")");

    // Track shared memory allocation per launch
    if (sharedMem > 0) {
      VGRE_LOG_DEBUG("RuntimeEngine",
                     "Shared memory requested: " + std::to_string(sharedMem) +
                         " bytes for kernel " + std::to_string(id) +
                         " on stream " + std::to_string(stream));
    }

    // Check for active capture
    auto captureIt = captureState_.find(stream);
    auto irIt = kernelIRCache_.find(id);
    if (irIt == kernelIRCache_.end())
      return VGREResult::ERROR_INVALID_KERNEL;

    if (captureIt != captureState_.end()) {
      VGRE_LOG_DEBUG("RuntimeEngine", "Capturing kernel launch " +
                                          std::to_string(id) + " on stream " +
                                          std::to_string(stream));
      return graphManager_->addKernelNode(captureIt->second, id,
                                          irIt->second.name, gridDim, blockDim,
                                          args, irIt->second.argTypes);
    }

    fn = it->second;

    // Deep copy the arguments because Python/caller might drop them before
    // thread executes
    size_t numArgs = irIt->second.argTypes.size();

    // We must copy the actual argument values (8 bytes each usually) that the
    // args[i] pointers point to
    argValues = std::make_shared<std::vector<uint64_t>>(numArgs);
    safeArgs = std::make_shared<std::vector<void *>>(numArgs);

    for (size_t i = 0; i < numArgs; ++i) {
      if (args[i]) {
        // Safe type-aware copy into 8-byte slots
        switch (irIt->second.argTypes[i]) {
        case ArgType::INT32:
        case ArgType::UINT32:
        case ArgType::FLOAT32:
          // Zero-init slot and copy 4 bytes
          (*argValues)[i] = 0;
          ::memcpy(&(*argValues)[i], args[i], 4);
          break;
        default:
          // Copy full 8 bytes (pointers, 64-bit types)
          (*argValues)[i] = *reinterpret_cast<uint64_t *>(args[i]);
          break;
        }
        // safeArgs[i] points to our stable copy
        (*safeArgs)[i] = &(*argValues)[i];
      } else {
        (*argValues)[i] = 0;
        (*safeArgs)[i] = &(*argValues)[i];
      }
    }
  }
  // mutex_ released here — allows concurrent launches from different streams

  // Execute using the CPUParallelExecutor deeply integrated with the scheduler.
  // We submit a coarse-grained task: one Scheduler worker thread will invoke
  // the CPUParallelExecutor, which parallelizes blocks via OpenMP.
  auto exec = executor_.get();
  std::string kName = "unknown";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto irIt = kernelIRCache_.find(id);
    if (irIt != kernelIRCache_.end())
      kName = irIt->second.name;
  }

  scheduler_->submitStreamTask(stream, [exec, fn, gridDim, blockDim, safeArgs,
                                        argValues, kName, sharedMem]() mutable {
    auto start = std::chrono::steady_clock::now();
    exec->execute(fn, gridDim, blockDim, safeArgs->data(), sharedMem);
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Workload estimation for GFLOPS/Bandwidth tracking
    size_t flops = 0;
    size_t memBytes = 0;
    if (kName == "background_compute") {
      flops = 50ULL * 2 * gridDim.total() * blockDim.total();
      memBytes = gridDim.total() * blockDim.total() * 3 * sizeof(float);
    }

    vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
        kName, blockDim.total(), 8, ms, memBytes, flops);
  });

  return VGREResult::SUCCESS;
}

// ── Convenience: register + launch ─────────────────────────────────────────
VGREResult RuntimeEngine::launchKernel(const std::string &name,
                                       const std::string &source,
                                       const dim3 &gridDim,
                                       const dim3 &blockDim, void **args,
                                       size_t sharedMem, StreamId stream) {
  KernelId id;
  auto r = registerKernel(name, source, id);
  if (r != VGREResult::SUCCESS)
    return r;
  return launchKernel(id, gridDim, blockDim, args, sharedMem, stream);
}

// ── Synchronization ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::synchronize() {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  scheduler_->waitAll();
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamSynchronize(StreamId stream) {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  scheduler_->waitStream(stream);
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::malloc(size_t size, MemoryHandle &outHandle) {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return memoryManager_->allocate(size, outHandle, currentDeviceId_);
}

VGREResult RuntimeEngine::mallocManaged(size_t size, MemoryHandle &outHandle,
                                        unsigned int flags) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  return memoryManager_->allocateManaged(size, outHandle, currentDeviceId_,
                                         flags);
}

VGREResult RuntimeEngine::deviceCanAccessPeer(DeviceId device,
                                              DeviceId peerDevice,
                                              int *canAccess) {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  if (!canAccess)
    return VGREResult::ERROR_INVALID_VALUE;

  // Virtual devices in the same process can always "technically" access each
  // other, but we enforce specific hardware-aware peer access rules.
  *canAccess = memoryManager_->canAccessPeer(device, peerDevice) ? 1 : 0;
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::deviceEnablePeerAccess(DeviceId peerDevice) {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return memoryManager_->enablePeerAccess(currentDeviceId_, peerDevice);
}

VGREResult RuntimeEngine::deviceDisablePeerAccess(DeviceId peerDevice) {
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return memoryManager_->disablePeerAccess(currentDeviceId_, peerDevice);
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::streamBeginCapture(StreamId stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (captureState_.count(stream))
    return VGREResult::ERROR_INVALID_VALUE;

  GraphId graph;
  auto r = graphManager_->createGraph(graph);
  if (r != VGREResult::SUCCESS)
    return r;

  captureState_[stream] = graph;
  VGRE_LOG_INFO("RuntimeEngine",
                "Started capture on stream " + std::to_string(stream));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamEndCapture(StreamId stream, GraphId &outGraph) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = captureState_.find(stream);
  if (it == captureState_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  outGraph = it->second;
  captureState_.erase(it);
  VGRE_LOG_INFO("RuntimeEngine", "Ended capture on stream " +
                                     std::to_string(stream) + " -> Graph " +
                                     std::to_string(outGraph));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::graphInstantiate(GraphId graph,
                                           GraphExecId &outExec) {
  return graphManager_->instantiate(graph, outExec);
}

VGREResult RuntimeEngine::graphLaunch(GraphExecId exec, StreamId stream) {
  return graphManager_->launch(exec, stream);
}

VGREResult RuntimeEngine::graphDestroy(GraphId graph) {
  return graphManager_->destroyGraph(graph);
}

VGREResult RuntimeEngine::graphExecDestroy(GraphExecId exec) {
  return graphManager_->destroyGraphExec(exec);
}

// ── Sub-system access ──────────────────────────────────────────────────────
MemoryManager &RuntimeEngine::getMemoryManager() { return *memoryManager_; }

Scheduler &RuntimeEngine::getScheduler() { return *scheduler_; }

VirtualGPUDevice &RuntimeEngine::getDevice() {
  return *devices_[currentDeviceId_];
}

VirtualGPUDevice &RuntimeEngine::getDevice(DeviceId id) {
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    throw VGREException(VGREResult::ERROR_INVALID_DEVICE, "Invalid device ID");
  }
  return *devices_[id];
}

// ── Singleton ──────────────────────────────────────────────────────────────
RuntimeEngine &RuntimeEngine::instance() {
  static RuntimeEngine engine;
  return engine;
}

} // namespace core
} // namespace vgre
