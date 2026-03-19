#include "vgre/core/runtime_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/runtime_profiler.h"
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
#include <mutex>
#include <fstream>
#include <functional>
#include <set>
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
      aee.updateHardwareMetrics(cores, dp.clockRate / 1000000.0, 0.0);
      if (!aee.isCalibrated()) {
        aee.runBenchmark(); // Perform real micro-benchmark to override initial guess
      }
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

  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return static_cast<int>(devices_.size());
}

VGREResult RuntimeEngine::setDevice(DeviceId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    return VGREResult::ERROR_INVALID_DEVICE;
  }
  currentDeviceId_ = id;
  VGRE_LOG_DEBUG("RuntimeEngine", "Switched to device " + std::to_string(id));
  return VGREResult::SUCCESS;
}

DeviceId RuntimeEngine::getDeviceId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return currentDeviceId_;
}

VGREResult RuntimeEngine::getDeviceProperties(DeviceId id,
                                              DeviceProperties &outProps) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    return VGREResult::ERROR_INVALID_DEVICE;
  }
  outProps = devices_[id]->getProperties();
  return VGREResult::SUCCESS;
}

bool RuntimeEngine::isInitialized() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return initialized_;
}

// ── Kernel registration ────────────────────────────────────────────────────
VGREResult RuntimeEngine::registerKernel(const std::string &name,
                                         const std::string &source,
                                         KernelId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Loading binary module: " + path);
  return translator_->loadBitcodeModule(path, outModule);
}

VGREResult RuntimeEngine::getModuleGlobal(ModuleHandle module,
                                         const std::string &name,
                                         void *&outAddr, size_t &outSize) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  return translator_->getGlobalSymbol(module, name, outAddr, outSize);
}

VGREResult RuntimeEngine::getKernelFromModule(ModuleHandle module,
                                              const std::string &name,
                                              KernelId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  auto it = kernelIRCache_.find(id);
  if (it == kernelIRCache_.end())
    return VGREResult::ERROR_INVALID_KERNEL;

  outTypes = it->second.argTypes;
  return VGREResult::SUCCESS;
}

const KernelIR *RuntimeEngine::getKernelIR(KernelId id) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = kernelIRCache_.find(id);
  if (it != kernelIRCache_.end()) {
    return &it->second;
  }
  return nullptr;
}

VGREResult RuntimeEngine::unloadModule(ModuleHandle module) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::shared_ptr<std::vector<std::vector<uint8_t>>> argValues;
  std::shared_ptr<std::vector<void *>> safeArgs;

  // Critical section: lookup kernel and check capture state
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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

    if (irIt->second.usesSyncthreads && blockDim.total() > 256) {
      if (warnedSyncthreads_.insert(id).second) {
        VGRE_LOG_WARN(
            "RuntimeEngine",
            "Kernel '" + irIt->second.name +
                "' uses __syncthreads with block size " +
                std::to_string(blockDim.total()) +
                ". VGRE only guarantees correct block barriers for blocks <= 256 "
                "threads. Results may be incorrect.");
      }
    }

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
    // thread executes. We use a vector of buffers to support arbitrary sizes (Stage 1).
    size_t numArgs = irIt->second.argTypes.size();
    argValues = std::make_shared<std::vector<std::vector<uint8_t>>>(numArgs);
    safeArgs = std::make_shared<std::vector<void *>>(numArgs);

    for (size_t i = 0; i < numArgs; ++i) {
      size_t argSize = 0;
      if (i < irIt->second.argSizes.size() && irIt->second.argSizes[i] > 0) {
        argSize = irIt->second.argSizes[i];
      } else {
        // Authoritative fallback based on type
        switch (irIt->second.argTypes[i]) {
          case ArgType::POINTER:
          case ArgType::INT64:
          case ArgType::UINT64:
          case ArgType::FLOAT64:
            argSize = 8;
            break;
          case ArgType::INT32:
          case ArgType::UINT32:
          case ArgType::FLOAT32:
            argSize = 4;
            break;
          case ArgType::STRUCT:
            VGRE_LOG_ERROR("RuntimeEngine", "Missing size for structural argument at index " + std::to_string(i));
            return VGREResult::ERROR_INVALID_VALUE;
          default:
            argSize = 8;
            break;
        }
      }
      
      (*argValues)[i].resize(argSize);
      if (args && args[i]) {
        ::memcpy((*argValues)[i].data(), args[i], argSize);
      } else {
        ::memset((*argValues)[i].data(), 0, argSize);
      }
      (*safeArgs)[i] = (*argValues)[i].data();
    }
  }
  // mutex_ released here — allows concurrent launches from different streams

  // Execute using the CPUParallelExecutor deeply integrated with the scheduler.
  // We submit a coarse-grained task: one Scheduler worker thread will invoke
  // the CPUParallelExecutor, which parallelizes blocks via OpenMP.
  auto exec = executor_.get();
  std::string kName = "unknown";
  std::vector<ArgType> argTypes;
  int streamPriority = 0;
  size_t staticSharedMem = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto irIt = kernelIRCache_.find(id);
    if (irIt != kernelIRCache_.end()) {
      kName = irIt->second.name;
      argTypes = irIt->second.argTypes;
      staticSharedMem = irIt->second.sharedMemSize;
    }
    if (stream != 0 && currentDeviceId_ >= 0 &&
        currentDeviceId_ < static_cast<DeviceId>(devices_.size())) {
      (void)devices_[currentDeviceId_]->getStreamPriority(stream,
                                                          streamPriority);
    }
  }

  auto fut = scheduler_->submitStreamTask(stream,
                                           [exec, fn, gridDim, blockDim,
                                            safeArgs, argValues, id, sharedMem,
                                            argTypes, staticSharedMem, kName]() mutable {
    auto start = std::chrono::steady_clock::now();
    size_t totalSharedMem = sharedMem + staticSharedMem;
    
    exec->execute(fn, gridDim, blockDim, safeArgs->data(), totalSharedMem);
    
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t memBytes = 0;
    size_t flops = 0;
    size_t totalThreads = gridDim.total() * blockDim.total();

    // 1. Authoritative memory accounting
    auto mm = &vgre::core::RuntimeEngine::instance().getMemoryManager();
    const auto *ir = vgre::core::RuntimeEngine::instance().getKernelIR(id);
    
    for (size_t i = 0; i < argTypes.size(); ++i) {
      if (argTypes[i] == ArgType::POINTER) {
        void *ptr = nullptr;
        if (i < argValues->size() && !(*argValues)[i].empty()) {
          ::memcpy(&ptr, (*argValues)[i].data(), sizeof(void*));
        }
        if (ptr && mm && mm->isValidHandle(ptr)) {
          memBytes += mm->getAllocationSize(ptr);
        }
      } else if (argTypes[i] == ArgType::STRUCT) {
        if (ir && i < ir->argSizes.size()) {
          memBytes += ir->argSizes[i] * totalThreads;
        }
      }
    }

    // 2. Authoritative FLOPs
    if (ir && ir->estimatedInstructionCount > 0) {
      flops = totalThreads * ir->estimatedInstructionCount;
    } else {
      flops = totalThreads; // absolute minimum baseline
    }

    vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
        kName, blockDim.total(), 8, ms, memBytes, flops);

    auto &profiler = vgre::advanced::RuntimeProfiler::instance();
    if (profiler.isEnabled()) {
      vgre::advanced::ProfileEvent ev;
      ev.kernelName = kName;
      ev.durationMs = ms;
      ev.memoryBytes = memBytes;
      ev.flops = flops;
      ev.throughputGBps = (ms > 0.0) ? (static_cast<double>(memBytes) / 1e9) / (ms / 1000.0) : 0.0;
      ev.gflops = (ms > 0.0) ? (static_cast<double>(flops) / 1e9) / (ms / 1000.0) : 0.0;
      ev.gridDim = gridDim;
      ev.blockDim = blockDim;
      ev.threadsUsed = static_cast<int>(blockDim.total());
      ev.timestamp = end;
      profiler.recordEvent(ev);
    }

    VGRE_LOG_DEBUG("RuntimeEngine", "Kernel completed: " + kName +
                                        " | Time: " + std::to_string(ms) +
                                        "ms | Est. GFLOPS: " +
                                        std::to_string((ms > 0) ? (flops / (ms * 1e6)) : 0));
  },
  streamPriority);

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

// ── Cooperative Kernel Launch ──────────────────────────────────────────────
// Grid-wide barrier semantics: On a CPU, this is implemented by executing
// all blocks in a single serialized phase (each block runs to completion
// before the next). This naturally provides grid-wide synchronization since
// all blocks share the same address space. For true cooperative semantics,
// the entire grid is dispatched as a single unit.

VGREResult RuntimeEngine::launchCooperativeKernel(KernelId id,
                                                   const dim3 &gridDim,
                                                   const dim3 &blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   StreamId stream) {
  VGRE_LOG_INFO("RuntimeEngine",
                "Cooperative kernel launch (grid=" +
                std::to_string(gridDim.total()) + " blocks, blockDim=" +
                std::to_string(blockDim.total()) + " threads)");

  // On CPU, cooperative launch semantics are satisfied by the existing
  // parallel executor since all blocks share the same process address space.
  // The key difference from a regular launch is that the caller guarantees
  // all blocks can run concurrently — which is always true on a CPU.
  return launchKernel(id, gridDim, blockDim, args, sharedMem, stream);
}

// ── Native Graph Dispatch ──────────────────────────────────────────────────

struct OwnedFusedLaunchArgs {
  std::vector<std::vector<uint8_t>> argValues;
  std::vector<void *> argPtrs;
  CompiledKernelFn fn;
  std::string name;
  size_t flops;
  size_t memBytes;
  size_t sharedMemBytes;
};

struct NativeGraphOperation {
  GraphNodeType type;
  // Kernel data
  OwnedFusedLaunchArgs kernelArgs;
  dim3 gridDim;
  dim3 blockDim;
  // Memcpy data
  void *dst;
  void *src;
  size_t count;
  int kind;
};

VGREResult RuntimeEngine::dispatchGraphNodes(const std::vector<GraphNode>& nodes, StreamId stream) {
  if (nodes.empty()) return VGREResult::SUCCESS;

  // Pre-resolve all nodes into a natively executable directed sequence
  auto compiledOps = std::make_shared<std::vector<NativeGraphOperation>>();
  compiledOps->reserve(nodes.size());

  // Build dependency graph for topological ordering
  std::unordered_map<uint64_t, size_t> nodeIndex;
  nodeIndex.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].nodeId == 0) {
      return VGREResult::ERROR_INVALID_VALUE;
    }
    nodeIndex[nodes[i].nodeId] = i;
  }

  std::vector<int> indegree(nodes.size(), 0);
  std::vector<std::vector<size_t>> adj(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto depId : nodes[i].deps) {
      auto it = nodeIndex.find(depId);
      if (it == nodeIndex.end()) {
        return VGREResult::ERROR_INVALID_VALUE;
      }
      size_t depIdx = it->second;
      adj[depIdx].push_back(i);
      indegree[i]++;
    }
  }

  std::queue<size_t> ready;
  for (size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0)
      ready.push(i);
  }
  std::vector<size_t> topoOrder;
  topoOrder.reserve(nodes.size());
  while (!ready.empty()) {
    size_t cur = ready.front();
    ready.pop();
    topoOrder.push_back(cur);
    for (size_t nxt : adj[cur]) {
      if (--indegree[nxt] == 0) {
        ready.push(nxt);
      }
    }
  }
  if (topoOrder.size() != nodes.size()) {
    VGRE_LOG_ERROR("RuntimeEngine",
                   "Graph dispatch failed: dependency cycle detected. "
                   "Verify graphAddDependency or node deps.");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_) return VGREResult::ERROR_NOT_INITIALIZED;

    for (size_t orderIdx = 0; orderIdx < topoOrder.size(); ++orderIdx) {
      const auto &node = nodes[topoOrder[orderIdx]];
      NativeGraphOperation op;
      op.type = node.type;

      if (node.type == GraphNodeType::KERNEL) {
        auto it = kernelCache_.find(node.kernelId);
        if (it == kernelCache_.end()) return VGREResult::ERROR_INVALID_KERNEL;

        auto irIt = kernelIRCache_.find(node.kernelId);
        std::string kName = (irIt != kernelIRCache_.end()) ? irIt->second.name : "unknown";

        op.gridDim = node.gridDim;
        op.blockDim = node.blockDim;
        op.kernelArgs.fn = it->second;
        op.kernelArgs.name = kName;
        op.kernelArgs.argValues.resize(node.capturedArgs.size());
        op.kernelArgs.argPtrs.resize(node.capturedArgs.size(), nullptr);

        size_t totalThreads = node.gridDim.total() * node.blockDim.total();
        size_t memBytes = 0;
        if (irIt != kernelIRCache_.end()) {
          for (size_t i = 0; i < irIt->second.argTypes.size() &&
                              i < node.capturedArgs.size();
               ++i) {
            if (irIt->second.argTypes[i] == ArgType::POINTER) {
              uint64_t raw = 0;
              std::memcpy(&raw, node.capturedArgs[i].data(), sizeof(uint64_t));
              void *ptr = reinterpret_cast<void *>(raw);
              if (memoryManager_ && memoryManager_->isValidHandle(ptr)) {
                memBytes += memoryManager_->getAllocationSize(ptr);
              }
            } else if (irIt->second.argTypes[i] == ArgType::STRUCT) {
              if (i < irIt->second.argSizes.size()) {
                memBytes += irIt->second.argSizes[i] * totalThreads;
              }
            }
          }
        }
        op.kernelArgs.memBytes = memBytes;
        
        uint64_t instCount = (irIt != kernelIRCache_.end()) ? irIt->second.estimatedInstructionCount : 1;
        op.kernelArgs.flops = totalThreads * instCount;
        op.kernelArgs.sharedMemBytes =
            (irIt != kernelIRCache_.end()) ? irIt->second.sharedMemSize : 0;

        for (size_t i = 0; i < node.capturedArgs.size(); ++i) {
          size_t copySize = node.capturedArgs[i].size();
          if (irIt != kernelIRCache_.end() && i < irIt->second.argTypes.size()) {
            if (i < irIt->second.argSizes.size() && irIt->second.argSizes[i] > 0) {
              copySize = irIt->second.argSizes[i];
            } else {
              switch (irIt->second.argTypes[i]) {
                case ArgType::INT32:
                case ArgType::UINT32:
                case ArgType::FLOAT32:
                  copySize = 4;
                  break;
                default:
                  copySize = 8;
                  break;
              }
            }
          }
          
          op.kernelArgs.argValues[i].resize(copySize);
          if (copySize > 0 && !node.capturedArgs[i].empty()) {
            ::memcpy(op.kernelArgs.argValues[i].data(), node.capturedArgs[i].data(),
                     std::min(copySize, node.capturedArgs[i].size()));
          } else {
            ::memset(op.kernelArgs.argValues[i].data(), 0, copySize);
          }
          op.kernelArgs.argPtrs[i] = op.kernelArgs.argValues[i].data();
        }
      } else if (node.type == GraphNodeType::MEMCPY) {
        op.dst = node.dst;
        op.src = node.src;
        op.count = node.count;
        op.kind = node.kind;
      }
      
      compiledOps->push_back(std::move(op));
    }
  }

  VGRE_LOG_INFO("RuntimeEngine", "Submitting Native Parallel Graph DAG of size " + 
                std::to_string(nodes.size()) + " on stream " + std::to_string(stream));

  auto exec = executor_.get();
  auto mm = memoryManager_.get();
  int streamPriority = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (stream != 0 && currentDeviceId_ >= 0 &&
        currentDeviceId_ < static_cast<DeviceId>(devices_.size())) {
      (void)devices_[currentDeviceId_]->getStreamPriority(stream, streamPriority);
    }
  }

  // Execute the compiled DAG in topological order within a single stream task.
  // The nodes have already been topo-sorted above, so we iterate sequentially.
  // This avoids the deadlock that occurs when a stream task blocks a worker
  // thread while waiting for concurrent sub-tasks that also need worker threads.
  scheduler_->submitStreamTask(stream, [exec, mm, compiledOps]() {
    for (size_t i = 0; i < compiledOps->size(); ++i) {
      auto& op = (*compiledOps)[i];
      if (op.type == GraphNodeType::KERNEL) {
        auto start = std::chrono::steady_clock::now();
        exec->execute(op.kernelArgs.fn, op.gridDim, op.blockDim,
                      op.kernelArgs.argPtrs.data(),
                      op.kernelArgs.sharedMemBytes);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
            op.kernelArgs.name, op.blockDim.total(), 8, ms, op.kernelArgs.memBytes, op.kernelArgs.flops);
      } else if (op.type == GraphNodeType::MEMCPY) {
        if (op.kind == VGRE_MEMCPY_HOST_TO_DEVICE) {
          mm->copyHostToDevice(op.dst, op.src, op.count);
        } else if (op.kind == VGRE_MEMCPY_DEVICE_TO_HOST) {
          mm->copyDeviceToHost(op.dst, op.src, op.count);
        } else if (op.kind == VGRE_MEMCPY_DEVICE_TO_DEVICE) {
          mm->copyDeviceToDevice(op.dst, op.src, op.count);
        }
      }
    }
  },
  streamPriority);

  // Async dispatch return. Wait/synchronize will propagate via main engine APIs.
  return VGREResult::SUCCESS;
}

// ── Synchronization ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::synchronize() {
  Scheduler *sched = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !memoryManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
    mm = memoryManager_.get();
    deviceId = currentDeviceId_;
  }
  return mm->allocate(size, outHandle, deviceId);
}

VGREResult RuntimeEngine::mallocManaged(size_t size, MemoryHandle &outHandle,
                                        unsigned int flags) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
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
VGREResult RuntimeEngine::graphCreate(GraphId &outGraph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->createGraph(outGraph);
}

VGREResult RuntimeEngine::streamBeginCapture(StreamId stream) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return captureState_.count(stream) > 0;
}

VGREResult RuntimeEngine::recordMemcpyToGraph(StreamId stream, void *dst, const void *src, size_t count, int kind) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->instantiate(graph, outExec);
}

VGREResult RuntimeEngine::graphUpdateExec(GraphExecId exec, GraphId graph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->updateExec(exec, graph);
}

VGREResult RuntimeEngine::graphLaunch(GraphExecId exec, StreamId stream) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_)
      return VGREResult::ERROR_NOT_INITIALIZED;
  }
  return graphManager_->launch(exec, stream);
}

VGREResult RuntimeEngine::graphDestroy(GraphId graph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->destroyGraph(graph);
}

VGREResult RuntimeEngine::graphExecDestroy(GraphExecId exec) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->destroyGraphExec(exec);
}

VGREResult RuntimeEngine::graphAddKernelNode(
    GraphId graph, KernelId kernelId, const std::string &name,
    const dim3 &grid, const dim3 &block, void **args,
    const std::vector<ArgType> &argTypes, const std::vector<uint64_t> &deps,
    uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->addKernelNodeWithDepsOut(graph, kernelId, name, grid,
                                                 block, args, argTypes, deps,
                                                 outNodeId);
}

VGREResult RuntimeEngine::graphAddMemcpyNode(
    GraphId graph, void *dst, void *src, size_t count, int kind,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->addMemcpyNodeWithDepsOut(graph, dst, src, count, kind,
                                                 deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddDependency(GraphId graph, uint64_t nodeId,
                                             uint64_t dependsOn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->addDependency(graph, nodeId, dependsOn);
}

VGREResult RuntimeEngine::graphUpdateKernelNode(
    GraphId graph, uint64_t nodeId, void **args,
    const std::vector<ArgType> &argTypes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->updateKernelNodeArgs(graph, nodeId, args, argTypes);
}

VGREResult RuntimeEngine::graphUpdateMemcpyNode(GraphId graph, uint64_t nodeId,
                                                void *dst, void *src,
                                                size_t count, int kind) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERROR_NOT_INITIALIZED;
  return graphManager_->updateMemcpyNode(graph, nodeId, dst, src, count, kind);
}

// ── Sub-system access ──────────────────────────────────────────────────────
MemoryManager &RuntimeEngine::getMemoryManager() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !memoryManager_) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *memoryManager_;
}

Scheduler &RuntimeEngine::getScheduler() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !scheduler_) {
    throw VGREException(VGREResult::ERROR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *scheduler_;
}

VirtualGPUDevice &RuntimeEngine::getDevice() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
