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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <thread>
#include <thread>

namespace vgre {
namespace core {

// ── Constructor / Destructor ───────────────────────────────────────────────
RuntimeEngine::RuntimeEngine() = default;
RuntimeEngine::~RuntimeEngine() {
  if (isInitialized()) {
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

  // Create virtual devices based on hardware resources or Environment Override.
  // We aim for 1 virtual device per NUMA node to model real-world memory topology.
  int deviceCount = 1;
  const char *envCount = std::getenv("VGRE_DEVICE_COUNT");
  if (envCount) {
    deviceCount = std::atoi(envCount);
  } else {
#if defined(__linux__)
    // Count NUMA nodes via sysfs
    std::set<int> numaNodes;
    for (int i = 0; i < 1024; ++i) { // Check reasonable range of CPUs
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/node";
      std::ifstream nodeFile(path);
      int node;
      if (nodeFile >> node) {
        numaNodes.insert(node);
      } else {
        break; // Assume contiguous CPU numbering
      }
    }
    if (!numaNodes.empty()) {
      deviceCount = static_cast<int>(numaNodes.size());
    } else {
      int cores = static_cast<int>(std::thread::hardware_concurrency());
      deviceCount = std::max(1, cores / 8);
    }
#elif defined(_WIN32)
    ULONG numaNodeCount = 0;
    if (GetNumaHighestNodeNumber(&numaNodeCount)) {
      deviceCount = static_cast<int>(numaNodeCount) + 1;
    } else {
      int cores = static_cast<int>(std::thread::hardware_concurrency());
      deviceCount = std::max(1, cores / 8);
    }
#elif defined(__APPLE__)
    // macOS Unified Memory means a single 'device' is often most efficient,
    // but we can model performance/efficiency core partitions.
    uint32_t packages = 1;
    size_t len = sizeof(packages);
    if (sysctlbyname("hw.packages", &packages, &len, NULL, 0) != 0) {
        packages = 1;
    }
    deviceCount = static_cast<int>(packages);
#else
    int cores = static_cast<int>(std::thread::hardware_concurrency());
    deviceCount = std::max(1, cores / 8);
#endif
    // Safety cap for stability
    deviceCount = std::min(deviceCount, 16);
  }

  VGRE_LOG_INFO("RuntimeEngine",
                "Creating " + std::to_string(deviceCount) +
                    " hardware-aligned virtual devices...");
  
  for (int i = 0; i < deviceCount; ++i) {
    auto dev = std::make_unique<VirtualGPUDevice>(i);
    dev->detectHardware();

    // Link Adaptive Execution Engine to real hardware metrics
    if (i == 0) {
      vgre::DeviceProperties dp = dev->getProperties();
      int cores = static_cast<int>(std::thread::hardware_concurrency());
      auto& aee = advanced::AdaptiveExecutionEngine::instance();
      aee.updateHardwareMetrics(cores, dp.clockRate / 1000.0, 0.0);
      aee.runBenchmark(); // Perform real micro-benchmark to override initial guess
    }
    dev->createContext();
    devices_.push_back(std::move(dev));
  }

  currentDeviceId_ = 0;

  // Global memory manager shared by active devices.
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

  vgre::advanced::IPCManager::instance().shutdown();

  scheduler_ = nullptr;
  executor_.reset();
  vectorEngine_.reset();
  translator_.reset();
  parser_.reset();
  memoryManager_.reset();
  devices_.clear();

  kernelCache_.clear();
  kernelIRCache_.clear();
  captureState_.clear();
  nextKernelId_ = 1;
  currentDeviceId_ = 0;
  initialized_ = false;

  VGRE_LOG_INFO("RuntimeEngine", "Shutdown complete");
  return VGREResult::SUCCESS;
}

// ── Device management ──────────────────────────────────────────────────────
int RuntimeEngine::getDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
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

DeviceId RuntimeEngine::getDeviceId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return currentDeviceId_;
}

VGREResult RuntimeEngine::getDeviceProperties(DeviceId id,
                                              DeviceProperties &outProps) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    return VGREResult::ERROR_INVALID_DEVICE;
  }
  outProps = devices_[id]->getProperties();
  return VGREResult::SUCCESS;
}

bool RuntimeEngine::isInitialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_;
}

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

const KernelIR *RuntimeEngine::getKernelIR(KernelId id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernelIRCache_.find(id);
  if (it != kernelIRCache_.end()) {
    return &it->second;
  }
  return nullptr;
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
  if (gridDim.x == 0 || blockDim.x == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  if ((gridDim.y == 0 && gridDim.z != 0) ||
      (blockDim.y == 0 && blockDim.z != 0)) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

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
      if (args && args[i]) {
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
          // Copy full 8 bytes (pointers, 64-bit types) without alignment UB
          (*argValues)[i] = 0;
          ::memcpy(&(*argValues)[i], args[i], sizeof(uint64_t));
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

  auto fut = scheduler_->submitStreamTask(stream, [this, exec, fn, gridDim, blockDim,
                                                   safeArgs, argValues, id,
                                                   sharedMem]() mutable {
    auto start = std::chrono::steady_clock::now();
    exec->execute(fn, gridDim, blockDim, safeArgs->data(), sharedMem);
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Real-world performance estimation based on IR and execution time
    size_t flops = 0;
    size_t memBytes = 0;
    std::string kName = "unknown";

    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto irIt = kernelIRCache_.find(id);
      if (irIt != kernelIRCache_.end()) {
        kName = irIt->second.name;

        // More accurate metric modeling
        size_t totalThreads = gridDim.total() * blockDim.total();

        // 1. Calculate realistic memory bandwidth usage based on argument types
        memBytes = 0;
        for (auto type : irIt->second.argTypes) {
          switch (type) {
          case ArgType::POINTER:
          case ArgType::INT64:
          case ArgType::UINT64:
          case ArgType::FLOAT64:
            memBytes += 8;
            break;
          default:
            memBytes += 4;
            break;
          }
        }
        memBytes *= totalThreads;

        // 2. Realistic FLOPs based on IR content (if available) or instruction density
        // For a "real" system, we look at actual IR instruction count if possible.
        // Currently, we use a proxy based on arg count (complexity) and thread count.
        size_t instructionsPerThread = 50 + (irIt->second.argTypes.size() * 10);
        flops = totalThreads * instructionsPerThread;
      }
    }

    vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
        kName, blockDim.total(), 8, ms, memBytes, flops);

    VGRE_LOG_DEBUG("RuntimeEngine", "Kernel completed: " + kName +
                                        " | Time: " + std::to_string(ms) +
                                        "ms | Est. GFLOPS: " +
                                        std::to_string(flops / (ms * 1e6)));
  });

  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto submitResult = fut.get();
    if (submitResult != VGREResult::SUCCESS) {
      return submitResult;
    }
  }

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

// ── Fused Dispatch ─────────────────────────────────────────────────────────

struct OwnedFusedLaunchArgs {
  std::vector<std::vector<uint8_t>> ownedData;
  std::vector<void *> argPtrs;
  CompiledKernelFn fn;
  std::string name;
  size_t flops;
  size_t memBytes;
};

VGREResult RuntimeEngine::launchFusedKernelGroup(const std::vector<GraphNode>& group, StreamId stream) {
  if (group.empty()) return VGREResult::SUCCESS;

  // Pre-resolve all kernels and arguments
  auto launchData = std::make_shared<std::vector<OwnedFusedLaunchArgs>>();
  launchData->reserve(group.size());

  const dim3 gridDim = group[0].gridDim;
  const dim3 blockDim = group[0].blockDim;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return VGREResult::ERROR_NOT_INITIALIZED;

    for (const auto& node : group) {
      auto it = kernelCache_.find(node.kernelId);
      if (it == kernelCache_.end()) return VGREResult::ERROR_INVALID_KERNEL;

      auto irIt = kernelIRCache_.find(node.kernelId);
      std::string kName = (irIt != kernelIRCache_.end()) ? irIt->second.name : "unknown";

      OwnedFusedLaunchArgs args;
      args.fn = it->second;
      args.name = kName;
      args.ownedData.reserve(node.capturedArgs.size());
      args.argPtrs.reserve(node.capturedArgs.size());

      size_t memBytes = 0;
      if (irIt != kernelIRCache_.end()) {
        for (auto type : irIt->second.argTypes) {
           memBytes += (type == ArgType::POINTER || type == ArgType::INT64 || type == ArgType::UINT64 || type == ArgType::FLOAT64) ? 8 : 4;
        }
      }
      size_t totalThreads = gridDim.total() * blockDim.total();
      args.memBytes = memBytes * totalThreads;
      args.flops = totalThreads * (50 + (node.capturedArgs.size() * 10));

      for (const auto &buf : node.capturedArgs) {
        args.ownedData.push_back(buf);
        args.argPtrs.push_back(args.ownedData.back().data());
      }
      launchData->push_back(std::move(args));
    }
  }

  VGRE_LOG_INFO("RuntimeEngine", "Submitting Fused Kernel Group of size " + 
                std::to_string(group.size()) + " on stream " + std::to_string(stream));

  auto exec = executor_.get();
  
  // Submit single task to scheduler
  auto fut = scheduler_->submitStreamTask(stream, [exec, launchData, gridDim, blockDim]() {
    for (auto& args : *launchData) {
      auto start = std::chrono::steady_clock::now();
      exec->execute(args.fn, gridDim, blockDim, args.argPtrs.data(), 0);
      auto end = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();

      vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
          args.name, blockDim.total(), 8, ms, args.memBytes, args.flops);
    }
  });

  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto submitResult = fut.get();
    if (submitResult != VGREResult::SUCCESS) {
      return submitResult;
    }
  }

  return VGREResult::SUCCESS;
}

// ── Synchronization ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::synchronize() {
  Scheduler *sched = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !scheduler_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    sched = scheduler_;
  }
  sched->waitAll();
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamSynchronize(StreamId stream) {
  Scheduler *sched = nullptr;
  VirtualGPUDevice *dev = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !scheduler_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    sched = scheduler_;
    if (stream != 0) {
      if (currentDeviceId_ < 0 ||
          currentDeviceId_ >= static_cast<DeviceId>(devices_.size())) {
        return VGREResult::ERROR_INVALID_DEVICE;
      }
      dev = devices_[currentDeviceId_].get();
    }
  }
  if (stream != 0) {
    return dev->synchronizeStream(stream);
  }
  sched->waitStream(stream);
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::malloc(size_t size, MemoryHandle &outHandle) {
  MemoryManager *mm = nullptr;
  DeviceId deviceId = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !memoryManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    mm = memoryManager_.get();
    deviceId = currentDeviceId_;
  }
  return mm->allocate(size, outHandle, deviceId);
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
  MemoryManager *mm = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !memoryManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    if (device < 0 || peerDevice < 0 ||
        device >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERROR_INVALID_DEVICE;
    }
    mm = memoryManager_.get();
  }
  if (!canAccess)
    return VGREResult::ERROR_INVALID_VALUE;

  // Virtual devices in the same process can always "technically" access each
  // other, but we enforce specific hardware-aware peer access rules.
  *canAccess = mm->canAccessPeer(device, peerDevice) ? 1 : 0;
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::deviceEnablePeerAccess(DeviceId peerDevice) {
  MemoryManager *mm = nullptr;
  DeviceId current = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !memoryManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    current = currentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERROR_INVALID_DEVICE;
    }
    mm = memoryManager_.get();
  }
  return mm->enablePeerAccess(current, peerDevice);
}

VGREResult RuntimeEngine::deviceDisablePeerAccess(DeviceId peerDevice) {
  MemoryManager *mm = nullptr;
  DeviceId current = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !memoryManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    current = currentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERROR_INVALID_DEVICE;
    }
    mm = memoryManager_.get();
  }
  return mm->disablePeerAccess(current, peerDevice);
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::streamBeginCapture(StreamId stream) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
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
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
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

bool RuntimeEngine::isStreamCapturing(StreamId stream) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return captureState_.count(stream) > 0;
}

VGREResult RuntimeEngine::recordMemcpyToGraph(StreamId stream, void *dst, const void *src, size_t count, int kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  
  auto it = captureState_.find(stream);
  if (it == captureState_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  VGRE_LOG_DEBUG("RuntimeEngine", "Capturing memcpy on stream " + std::to_string(stream));
  // We must cast away constness for the graph manager (which copies data anyway)
  return graphManager_->addMemcpyNode(it->second, dst, const_cast<void*>(src), count, kind);
}

VGREResult RuntimeEngine::graphInstantiate(GraphId graph,
                                           GraphExecId &outExec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->instantiate(graph, outExec);
}

VGREResult RuntimeEngine::graphLaunch(GraphExecId exec, StreamId stream) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !graphManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
  }
  return graphManager_->launch(exec, stream);
}

VGREResult RuntimeEngine::graphDestroy(GraphId graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->destroyGraph(graph);
}

VGREResult RuntimeEngine::graphExecDestroy(GraphExecId exec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->destroyGraphExec(exec);
}

// ── Sub-system access ──────────────────────────────────────────────────────
MemoryManager &RuntimeEngine::getMemoryManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !memoryManager_) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *memoryManager_;
}

Scheduler &RuntimeEngine::getScheduler() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || !scheduler_) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *scheduler_;
}

VirtualGPUDevice &RuntimeEngine::getDevice() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || devices_.empty()) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  if (currentDeviceId_ < 0 ||
      currentDeviceId_ >= static_cast<DeviceId>(devices_.size())) {
    throw VGREException(VGREResult::ERROR_INVALID_DEVICE, "Invalid device ID");
  }
  return *devices_[currentDeviceId_];
}

VirtualGPUDevice &RuntimeEngine::getDevice(DeviceId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
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
