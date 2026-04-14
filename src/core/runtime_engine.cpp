#include "vgre/core/runtime_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/common/logger.h"
#include "vgre/compiler/kernel_parser.h"
#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/runtime/vector_engine.h"
#include "vgre/runtime/block_worker_pool.h"


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <fstream>
#include <set>
#include <thread>

namespace vgre {
namespace core {

// ── Constructor / Destructor ───────────────────────────────────────────────
RuntimeEngine::RuntimeEngine() = default;
RuntimeEngine::~RuntimeEngine() {
  if (isInitialized()) {
    // Phase 10: Automatic Chrome Trace Export on shutdown
    advanced::RuntimeProfiler::instance().exportToFile("vgre_trace.json");
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
  parser_ = std::make_unique<compiler::ClangKernelParser>();
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
      
      const char* bench_env = std::getenv("VGRE_BENCHMARK");
      if (!bench_env || std::string(bench_env) != "OFF") {
          // Perform real micro-benchmark in a background thread
          benchmarkThread_ = std::thread([&aee]() {
            aee.runBenchmark();
          });
      } else {
          VGRE_LOG_INFO("RuntimeEngine", "Background benchmark disabled via VGRE_BENCHMARK=OFF");
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

  // Initialize persistent BlockWorkerPool for authoritative block-level concurrency
  runtime::BlockWorkerPool::instance().initialize();


  // Phase 10: Controllable Background Tasks (Zero-Simulation Hardening)
  // Register with global IPC service as a client by default, unless disabled.
  const char* ipc_mode = std::getenv("VGRE_IPC_MODE");
  if (!ipc_mode || std::string(ipc_mode) != "OFF") {
      vgre::advanced::IPCManager::instance().initialize(false);
  } else {
      VGRE_LOG_INFO("RuntimeEngine", "IPC Service disabled via VGRE_IPC_MODE=OFF");
  }

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
    return VGREResult::ERR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Shutting down VGRE Runtime Engine...");

  if (benchmarkThread_.joinable()) {
      benchmarkThread_.join();
  }

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
    VGRE_LOG_ERROR("RuntimeEngine", "Invalid device ID: " + std::to_string(id) + " (Device Count: " + std::to_string(devices_.size()) + ")");
    return VGREResult::ERR_INVALID_DEVICE;
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
    VGRE_LOG_ERROR("RuntimeEngine", "Invalid device ID: " + std::to_string(id) + " (Device Count: " + std::to_string(devices_.size()) + ")");
    return VGREResult::ERR_INVALID_DEVICE;
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
    return VGREResult::ERR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Registering kernel: " + name);

  // Parse kernel source
  KernelIR ir;
  auto r = parser_->parse(name, source, ir);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("RuntimeEngine", "Failed to parse kernel: " + name);
    return r;
  }

  // Translate to executable
  KernelId id = nextKernelId_++;
  kernelIRCache_[id] = ir;
  
  // v0.1.2 Extraordinary Sophistication: Asynchronous JIT Pipelining
  // We trigger translation in the background immediately during registration.
  pendingKernels_[id] = translator_->prepare(kernelIRCache_[id]);

  outId = id;
  kernelIRCache_[id] = ir;
  kernelNames_[name] = id;
  
  // Track kernel source for runtime profiling/inspection
  vgre::advanced::RuntimeProfiler::instance().setKernelSource(name, ir.source, ir.irCode);

  VGRE_LOG_INFO("RuntimeEngine", "Kernel '" + name + "' registered with ID " +
                                     std::to_string(id));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::loadModule(const std::string &path,
                                     ModuleHandle &outModule) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

  VGRE_LOG_INFO("RuntimeEngine", "Loading binary module: " + path);
  return translator_->loadBitcodeModule(path, outModule);
}

VGREResult RuntimeEngine::getModuleGlobal(ModuleHandle module,
                                         const std::string &name,
                                         void *&outAddr, size_t &outSize) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

  return translator_->getGlobalSymbol(module, name, outAddr, outSize);
}

VGREResult RuntimeEngine::getKernelFromModule(ModuleHandle module,
                                              const std::string &name,
                                              KernelId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

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

VGREResult RuntimeEngine::fuseKernels(const std::vector<KernelId> &ids,
                                      KernelId &outFusedId, std::string* outName) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (ids.size() < 2) return VGREResult::ERR_INVALID_VALUE;

  VGRE_LOG_INFO("RuntimeEngine", "Fusing " + std::to_string(ids.size()) + " kernels at the IR level (Link-Time Fusion).");

  std::vector<KernelIR> flattenedComponents;
  std::vector<KernelId> flattenedIds;
  std::function<void(KernelId)> flatten = [&](KernelId kid) {
      auto it = kernelIRCache_.find(kid);
      if (it == kernelIRCache_.end()) return;
      
      const auto& k = it->second;
      if (!k.fusedFrom.empty()) {
          for (auto fid : k.fusedFrom) {
              flatten(fid);
          }
      } else {
          flattenedComponents.push_back(k);
          flattenedIds.push_back(kid);
      }
  };

  for (auto kid : ids) {
      flatten(kid);
  }

  std::string fusedName = "vgre_fused";
  for (const auto& c : flattenedComponents) {
      fusedName += "_" + c.name;
  }

  KernelIR fusedIR;
  VGREResult res = translator_->fuseKernels(flattenedComponents, fusedName, fusedIR);
  if (res != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("RuntimeEngine", "IR Fusion failed for " + fusedName);
      return res;
  }

  // Generate combined argument metadata for the fused kernel
  fusedIR.argTypeNames.clear();
  fusedIR.argSizes.clear();
  fusedIR.fusedFrom = flattenedIds;
  for (const auto &c : flattenedComponents) {
      fusedIR.argTypeNames.insert(fusedIR.argTypeNames.end(), c.argTypeNames.begin(), c.argTypeNames.end());
      fusedIR.argSizes.insert(fusedIR.argSizes.end(), c.argSizes.begin(), c.argSizes.end());
  }

  // Register the fused IR directly and trigger JIT preparation
  KernelId newId = nextKernelId_++;
  kernelIRCache_[newId] = fusedIR;
  pendingKernels_[newId] = translator_->prepare(kernelIRCache_[newId]);
  outFusedId = newId;
  kernelNames_[fusedName] = newId;
  if (outName) *outName = fusedName;

  VGRE_LOG_INFO("RuntimeEngine", "Fused kernel '" + fusedName + "' (IR-Linked) registered as ID " + std::to_string(outFusedId));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::getKernelArgTypes(KernelId id,
                                            std::vector<ArgType> &outTypes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

  auto it = kernelIRCache_.find(id);
  if (it == kernelIRCache_.end())
    return VGREResult::ERR_INVALID_KERNEL;

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
    return VGREResult::ERR_NOT_INITIALIZED;

  return translator_->unloadModule(module);
}

// ── Kernel launch (by ID) ──────────────────────────────────────────────────
VGREResult RuntimeEngine::launchKernel(KernelId id, const dim3 &gridDim,
                                       const dim3 &blockDim, void **args,
                                       size_t sharedMem, StreamId stream,
                                       const dim3 &gridOffset) {
  if (gridDim.x == 0 || blockDim.x == 0)
    return VGREResult::ERR_INVALID_VALUE;
  if ((gridDim.y == 0 && gridDim.z != 0) ||
      (blockDim.y == 0 && blockDim.z != 0)) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  CompiledKernelFn fn;
  std::shared_ptr<std::vector<std::vector<uint8_t>>> argValues;
  std::shared_ptr<std::vector<void *>> safeArgs;

  // Critical section: lookup kernel and check capture state
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_)
      return VGREResult::ERR_NOT_INITIALIZED;

    auto irIt = kernelIRCache_.find(id);
    if (irIt == kernelIRCache_.end()) {
      return VGREResult::ERR_INVALID_KERNEL;
    }

    VGRE_LOG_INFO("RuntimeEngine", "Launching kernel " + irIt->second.name +
                                       " grid(" + std::to_string(gridDim.x) +
                                       "," + std::to_string(gridDim.y) + "," +
                                       std::to_string(gridDim.z) + ")" +
                                       " block(" + std::to_string(blockDim.x) +
                                       "," + std::to_string(blockDim.y) + "," +
                                       std::to_string(blockDim.z) + ")");

    // Check for active capture
    auto captureIt = captureState_.find(stream);
    if (captureIt != captureState_.end()) {
    VGRE_LOG_DEBUG("RuntimeEngine", "Capturing kernel launch " +
                                          std::to_string(id) + " on stream " +
                                          std::to_string(stream));
      
      // Zero-Simulation Enhancement: Track last node ID for implicit stream dependencies
      std::vector<uint64_t> deps;
      auto lastNodeIt = lastCapturedNodeId_.find(stream);
      if (lastNodeIt != lastCapturedNodeId_.end() && lastNodeIt->second != 0) {
          deps.push_back(lastNodeIt->second);
      }

      uint64_t newNodeId = 0;
      auto res = graphManager_->addKernelNodeWithDepsOut(captureIt->second, id,
                                          irIt->second.name, gridDim, blockDim,
                                          args, irIt->second.argTypes, deps, newNodeId);
      
      if (res == VGREResult::SUCCESS) {
          lastCapturedNodeId_[stream] = newNodeId;
      }
      return res;
    }

    // Resolve compiled function (Check cache first, then pending JIT futures)
    auto cacheIt = kernelCache_.find(id);
    if (cacheIt != kernelCache_.end()) {
        fn = cacheIt->second;
    } else {
        auto pendIt = pendingKernels_.find(id);
        if (pendIt != pendingKernels_.end()) {
            VGRE_LOG_INFO("RuntimeEngine", "Resolving asynchronous JIT future for kernel: " + irIt->second.name);
            vgre::JITResult jres = pendIt->second.get(); // Synchronize with pipelined JIT task
            fn = jres.fn;
            if (!fn) {
                VGRE_LOG_ERROR("RuntimeEngine", "Asynchronous JIT failed for kernel: " + irIt->second.name);
                return VGREResult::ERR_COMPILATION;
            }
            // Propagate authoritative metadata from JIT pipeline back into cache
            irIt->second.sharedMemSize = jres.sharedMemSize;
            irIt->second.argSizes = jres.argSizes;
            irIt->second.estimatedInstructionCount = jres.estimatedInstructionCount;
            irIt->second.staticFlopCount = jres.staticFlopCount;
            
            kernelCache_[id] = fn;
            pendingKernels_.erase(id);
        } else {
            return VGREResult::ERR_INVALID_KERNEL;
        }
    }

    // Deep copy the arguments because Python/caller might drop them before
    // thread executes.
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
            return VGREResult::ERR_INVALID_VALUE;
          default:
            argSize = 8;
            break;
        }
      }
      
      (*argValues)[i].resize(argSize);
      if (args && args[i]) {
        ::memcpy((*argValues)[i].data(), args[i], argSize);
        if (irIt->second.argTypes[i] == ArgType::POINTER) {
            void* ptrValue = *(void**)(*argValues)[i].data();
            VGRE_LOG_DEBUG("RuntimeEngine", "Arg[" + std::to_string(i) + "] Type=POINTER Value=" + std::to_string((uintptr_t)ptrValue));
        }
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
  uint64_t estimatedInstructionCount = 0;
  uint64_t staticFlopCount = 0;
  bool flopCountVerified = false;
  bool usesSyncthreads = false;
  MemoryManager* rawMm = memoryManager_.get();

  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto irIt = kernelIRCache_.find(id);
    if (irIt != kernelIRCache_.end()) {
      kName = irIt->second.name;
      argTypes = irIt->second.argTypes;
      staticSharedMem = irIt->second.sharedMemSize;
      estimatedInstructionCount = irIt->second.estimatedInstructionCount;
      staticFlopCount = irIt->second.staticFlopCount;
      flopCountVerified = irIt->second.flopCountVerified;
      usesSyncthreads = irIt->second.usesSyncthreads;
    }

    if (stream != 0 && currentDeviceId_ >= 0 &&
        currentDeviceId_ < static_cast<DeviceId>(devices_.size())) {
      (void)devices_[currentDeviceId_]->getStreamPriority(stream,
                                                          streamPriority);
    }
  }

  // v0.1.2 Extraordinary Sophistication: Authoritative UVM Pre-fetching
  // Proactively touch managed pointer arguments to trigger background migration
  // before the scheduler worker thread hits the first instruction. This hides
  // UVM page-fault latency from the execution pipeline.
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (argTypes[i] == ArgType::POINTER) {
      void *ptr = nullptr;
      if (i < argValues->size() && !(*argValues)[i].empty()) {
        ::memcpy(&ptr, (*argValues)[i].data(), sizeof(void*));
      }
      if (ptr && memoryManager_->isValidHandle(ptr)) {
          size_t sz = memoryManager_->getAllocationSize(ptr);
          memoryManager_->memPrefetchAsync(ptr, sz, 0);
      }
    }
  }

  auto fut = scheduler_->submitStreamTask(stream,
                                           [exec, fn, gridDim, blockDim,
                                            safeArgs, argValues, sharedMem,
                                            argTypes, staticSharedMem, kName, gridOffset,
                                            estimatedInstructionCount, staticFlopCount,
                                            flopCountVerified,
                                            rawMm, usesSyncthreads]() mutable {
    auto start = std::chrono::steady_clock::now();
    size_t totalSharedMem = sharedMem + staticSharedMem;

    // Calculate per-block metrics for real-time instrumentation
    uint64_t flopsPerBlock = 0;
    uint64_t bytesPerBlock = 0;

    uint32_t totalBlocksCount = gridDim.total();

    // 1. Authoritative memory accounting — use pre-captured manager, not singleton
    size_t totalMemBytes = 0;
    for (size_t i = 0; i < argTypes.size(); ++i) {
      if (argTypes[i] == ArgType::POINTER) {
        void *ptr = nullptr;
        if (i < argValues->size() && !(*argValues)[i].empty()) {
          ::memcpy(&ptr, (*argValues)[i].data(), sizeof(void*));
        }
        if (ptr && rawMm && rawMm->isValidHandle(ptr)) {
          totalMemBytes += rawMm->getAllocationSize(ptr);
        }
      }
    }
    bytesPerBlock = (totalBlocksCount > 0) ? (totalMemBytes / totalBlocksCount) : 0;

    // 2. Authoritative FLOPs — use LLVM IR static analysis when available.
    // staticFlopCount is from analyzeStaticFlops() which weights actual FP
    // instructions: fadd/fsub/fmul/fdiv = 1, fma = 2, sqrt/transcendental = 4.
    // estimatedInstructionCount counts ALL instructions (loads, branches, casts)
    // which is NOT a FLOP count — never use it raw as a FLOP proxy.
    if (staticFlopCount > 0) {
      // Authoritative: LLVM IR static analysis — exact FP op count × threads.
      flopsPerBlock = blockDim.total() * staticFlopCount;
    } else if (flopCountVerified) {
      // LLVM IR ran and confirmed zero FP operations (e.g. memset/memcpy
      // kernel).  Report the absolute minimum so downstream throughput
      // calculations don't divide by zero; do NOT apply the 30% heuristic.
      VGRE_LOG_DEBUG("RuntimeEngine",
                     "Kernel '" + kName +
                     "' contains no FP operations (verified by LLVM IR); "
                     "reporting minimum FLOP baseline.");
      flopsPerBlock = blockDim.total(); // 1 FLOP per thread minimum
    } else if (estimatedInstructionCount > 0) {
      // LLVM IR analysis unavailable (kernel loaded from precompiled module or
      // JIT compilation failed).  Conservatively assume ~20% of instructions
      // are FP ops — a better-calibrated midpoint for mixed CPU kernels than
      // the old 30% that over-counted pure memory-bound workloads.
      VGRE_LOG_DEBUG("RuntimeEngine",
                     "FLOP count unverified for kernel '" + kName +
                     "'; using 20% instruction heuristic (" +
                     std::to_string(estimatedInstructionCount) + " instructions)");
      flopsPerBlock = blockDim.total() *
                      std::max(uint64_t(1), estimatedInstructionCount / 5);
    } else {
      flopsPerBlock = blockDim.total(); // absolute minimum baseline
    }

    exec->execute(fn, gridDim, blockDim, safeArgs->data(), totalSharedMem, flopsPerBlock, bytesPerBlock, gridOffset, usesSyncthreads);

    
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t memBytes = totalMemBytes;
    size_t flops = flopsPerBlock * totalBlocksCount;

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

VGREResult RuntimeEngine::launchKernel(const std::string &name,
                                       const std::string &source,
                                       const dim3 &gridDim,
                                       const dim3 &blockDim, void **args,
                                       size_t sharedMem, StreamId stream,
                                       const dim3 &gridOffset) {
  KernelId id;
  auto r = registerKernel(name, source, id);
  if (r != VGREResult::SUCCESS)
    return r;
  return launchKernel(id, gridDim, blockDim, args, sharedMem, stream, gridOffset);
}

VGREResult RuntimeEngine::launchCooperativeKernel(const std::string &name,
                                                   const std::string &source,
                                                   const dim3 &gridDim,
                                                   const dim3 &blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   StreamId stream) {
  KernelId id;
  auto r = registerKernel(name, source, id);
  if (r != VGREResult::SUCCESS)
    return r;
  return launchCooperativeKernel(id, gridDim, blockDim, args, sharedMem, stream);
}

// ── Multi-Device Cooperative Kernel Launch ────────────────────────────────
// Each virtual device runs executeCooperative() in its own std::thread so all
// devices proceed concurrently.  A shared atomic ready-counter acts as a
// start-gate: every device thread increments it and spin-waits until every
// other device is also at the gate, then all burst through simultaneously.
//
// This mirrors real hardware: NV's cudaLaunchCooperativeKernelMultiDevice
// issues all GPU launches before any SM receives a wave of threadblocks.
//
// Per-device grid-wide barriers (this_grid().sync() / vgre_jit_syncgrid)
// work correctly because each device thread has its own stack-allocated
// GridBarrierState and its own set of block-execution threads.
VGREResult RuntimeEngine::launchCooperativeKernelMultiDevice(
    const std::vector<CoopMultiLaunchParams> &launchList) {

  if (launchList.empty()) return VGREResult::SUCCESS;

  const size_t N = launchList.size();
  VGRE_LOG_INFO("RuntimeEngine",
                "Multi-device cooperative launch: " + std::to_string(N) +
                    " devices");

  // ── Phase 1: Compile every kernel and prepare arguments ──────────────────
  // All JIT work is serialised under the engine mutex before any execution
  // starts, so Phase 2 never blocks waiting for compilation.

  struct ReadyLaunch {
    CompiledKernelFn fn;
    dim3  gridDim;
    dim3  blockDim;
    std::shared_ptr<std::vector<std::vector<uint8_t>>> argValues;
    std::shared_ptr<std::vector<void *>>               safeArgs;
    size_t   sharedMem;
    StreamId stream;
    uint64_t flopsPerBlock;
    uint64_t bytesPerBlock;
  };
  std::vector<ReadyLaunch> ready(N);

  for (size_t i = 0; i < N; ++i) {
    const auto &p = launchList[i];
    if (p.gridDim.x == 0 || p.blockDim.x == 0)
      return VGREResult::ERR_INVALID_VALUE;

    // Register kernel — JIT-compiles if not already cached.
    KernelId id;
    {
      auto r = registerKernel(p.name, p.source, id);
      if (r != VGREResult::SUCCESS) return r;
    }

    // Resolve compiled function and deep-copy arguments under the mutex.
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_) return VGREResult::ERR_NOT_INITIALIZED;

    auto irIt = kernelIRCache_.find(id);
    if (irIt == kernelIRCache_.end()) return VGREResult::ERR_INVALID_KERNEL;

    // Drain async JIT future if present.
    auto cacheIt = kernelCache_.find(id);
    if (cacheIt == kernelCache_.end()) {
      auto pendIt = pendingKernels_.find(id);
      if (pendIt == pendingKernels_.end()) return VGREResult::ERR_INVALID_KERNEL;
      JITResult jres = pendIt->second.get();
      if (!jres.fn) {
        VGRE_LOG_ERROR("RuntimeEngine",
                       "JIT failed for multi-device launch of kernel: " + p.name);
        return VGREResult::ERR_COMPILATION;
      }
      irIt->second.sharedMemSize             = jres.sharedMemSize;
      irIt->second.argSizes                  = jres.argSizes;
      irIt->second.estimatedInstructionCount = jres.estimatedInstructionCount;
      irIt->second.staticFlopCount           = jres.staticFlopCount;
      kernelCache_[id] = jres.fn;
      pendingKernels_.erase(id);
      cacheIt = kernelCache_.find(id);
    }

    ready[i].fn        = cacheIt->second;
    ready[i].gridDim   = p.gridDim;
    ready[i].blockDim  = p.blockDim;
    ready[i].sharedMem = p.sharedMem;
    ready[i].stream    = p.stream;

    // Flop estimate for telemetry — mirrors the logic in launchKernel.
    {
      const auto &ir = irIt->second;
      if (ir.staticFlopCount > 0) {
        ready[i].flopsPerBlock = ready[i].blockDim.total() * ir.staticFlopCount;
      } else if (!ir.flopCountVerified && ir.estimatedInstructionCount > 0) {
        ready[i].flopsPerBlock = ready[i].blockDim.total() *
            std::max(uint64_t(1), ir.estimatedInstructionCount / 5);
      } else {
        ready[i].flopsPerBlock = ready[i].blockDim.total(); // minimum baseline
      }
    }
    ready[i].bytesPerBlock = 0;

    // Deep-copy kernel arguments so each device thread owns its own copies.
    const size_t numArgs = irIt->second.argTypes.size();
    ready[i].argValues =
        std::make_shared<std::vector<std::vector<uint8_t>>>(numArgs);
    ready[i].safeArgs = std::make_shared<std::vector<void *>>(numArgs);

    for (size_t j = 0; j < numArgs; ++j) {
      size_t argSize = 0;
      if (j < irIt->second.argSizes.size() && irIt->second.argSizes[j] > 0) {
        argSize = irIt->second.argSizes[j];
      } else {
        switch (irIt->second.argTypes[j]) {
          case ArgType::POINTER:
          case ArgType::INT64:
          case ArgType::UINT64:
          case ArgType::FLOAT64: argSize = 8; break;
          case ArgType::INT32:
          case ArgType::UINT32:
          case ArgType::FLOAT32: argSize = 4; break;
          case ArgType::STRUCT:
            VGRE_LOG_ERROR("RuntimeEngine",
                           "STRUCT arg " + std::to_string(j) +
                               " has unknown size in multi-device launch of " +
                               p.name);
            return VGREResult::ERR_INVALID_VALUE;
          default: argSize = 8; break;
        }
      }
      (*ready[i].argValues)[j].resize(argSize);
      if (p.args && p.args[j])
        ::memcpy((*ready[i].argValues)[j].data(), p.args[j], argSize);
      else
        ::memset((*ready[i].argValues)[j].data(), 0, argSize);
      (*ready[i].safeArgs)[j] = (*ready[i].argValues)[j].data();
    }
  }

  // ── Phase 2: Concurrent execution ────────────────────────────────────────
  // One std::thread per device; all threads spin on readyCount until every
  // device has passed Phase 1, then burst into executeCooperative together.
  std::atomic<int>         readyCount{0};
  std::vector<VGREResult>  results(N, VGREResult::SUCCESS);
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back(
        [this, &ready, &readyCount, &results, N, i]() {
          // Start-gate: wait until every device thread has reached this point.
          readyCount.fetch_add(1, std::memory_order_release);
          while (readyCount.load(std::memory_order_acquire) <
                 static_cast<int>(N))
            std::this_thread::yield();

          auto &rl = ready[i];
          results[i] = executor_->executeCooperative(
              rl.fn, rl.gridDim, rl.blockDim, rl.safeArgs->data(),
              rl.sharedMem, rl.flopsPerBlock, rl.bytesPerBlock);
        });
  }

  for (auto &t : threads) t.join();

  for (const auto &res : results)
    if (res != VGREResult::SUCCESS) return res;
  return VGREResult::SUCCESS;
}

// ── Cooperative Kernel Launch ──────────────────────────────────────────────
// Grid-wide barrier semantics: Cooperative kernels require grid-wide synchronization
// where all blocks can synchronize with each other. This is implemented by:
// 1. Launching all blocks concurrently (not sequentially)
// 2. Providing a grid-wide barrier mechanism accessible to all blocks
// 3. Ensuring all blocks complete before returning
//
// Implementation: We use a shared atomic counter and condition variable to implement
// a grid-wide barrier that all blocks can synchronize on.

VGREResult RuntimeEngine::launchCooperativeKernel(KernelId id,
                                                   const dim3 &gridDim,
                                                   const dim3 &blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   StreamId stream,
                                                   const dim3 &gridOffset) {
  if (gridDim.x == 0 || blockDim.x == 0)
    return VGREResult::ERR_INVALID_VALUE;
  if ((gridDim.y == 0 && gridDim.z != 0) ||
      (blockDim.y == 0 && blockDim.z != 0)) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  VGRE_LOG_INFO("RuntimeEngine",
                "Cooperative kernel launch (grid=" +
                std::to_string(gridDim.total()) + " blocks, blockDim=" +
                std::to_string(blockDim.total()) + " threads)");

  CompiledKernelFn fn;
  std::shared_ptr<std::vector<std::vector<uint8_t>>> argValues;
  std::shared_ptr<std::vector<void *>> safeArgs;

  // Critical section: lookup kernel and prepare arguments (same as regular launch)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_)
      return VGREResult::ERR_NOT_INITIALIZED;

    auto irIt = kernelIRCache_.find(id);
    if (irIt == kernelIRCache_.end()) {
      return VGREResult::ERR_INVALID_KERNEL;
    }

    // Check for active capture (cooperative kernels can be captured)
    auto captureIt = captureState_.find(stream);
    if (captureIt != captureState_.end()) {
      VGRE_LOG_DEBUG("RuntimeEngine", "Capturing cooperative kernel launch " +
                                            std::to_string(id) + " on stream " +
                                            std::to_string(stream));
      
      std::vector<uint64_t> deps;
      auto lastNodeIt = lastCapturedNodeId_.find(stream);
      if (lastNodeIt != lastCapturedNodeId_.end() && lastNodeIt->second != 0) {
          deps.push_back(lastNodeIt->second);
      }

      uint64_t newNodeId = 0;
      auto res = graphManager_->addKernelNodeWithDepsOut(captureIt->second, id,
                                          irIt->second.name, gridDim, blockDim,
                                          args, irIt->second.argTypes, deps, newNodeId);
      
      if (res == VGREResult::SUCCESS) {
          lastCapturedNodeId_[stream] = newNodeId;
      }
      return res;
    }

    // Resolve compiled function
    auto cacheIt = kernelCache_.find(id);
    if (cacheIt != kernelCache_.end()) {
        fn = cacheIt->second;
    } else {
        auto pendIt = pendingKernels_.find(id);
        if (pendIt != pendingKernels_.end()) {
            VGRE_LOG_INFO("RuntimeEngine", "Resolving asynchronous JIT future for cooperative kernel: " + irIt->second.name);
            vgre::JITResult jres = pendIt->second.get();
            fn = jres.fn;
            if (!fn) {
                VGRE_LOG_ERROR("RuntimeEngine", "Asynchronous JIT failed for cooperative kernel: " + irIt->second.name);
                return VGREResult::ERR_COMPILATION;
            }
            irIt->second.sharedMemSize = jres.sharedMemSize;
            irIt->second.argSizes = jres.argSizes;
            irIt->second.estimatedInstructionCount = jres.estimatedInstructionCount;
            irIt->second.staticFlopCount = jres.staticFlopCount;
            
            kernelCache_[id] = fn;
            pendingKernels_.erase(id);
        } else {
            return VGREResult::ERR_INVALID_KERNEL;
        }
    }

    // Deep copy arguments
    size_t numArgs = irIt->second.argTypes.size();
    argValues = std::make_shared<std::vector<std::vector<uint8_t>>>(numArgs);
    safeArgs = std::make_shared<std::vector<void *>>(numArgs);

    for (size_t i = 0; i < numArgs; ++i) {
      size_t argSize = 0;
      if (i < irIt->second.argSizes.size() && irIt->second.argSizes[i] > 0) {
        argSize = irIt->second.argSizes[i];
      } else {
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
            return VGREResult::ERR_INVALID_VALUE;
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

  // Cooperative execution: all blocks run in separate threads with a shared
  // grid-wide barrier so this_grid().sync() works correctly.
  VGRE_LOG_INFO("RuntimeEngine",
                "Cooperative kernel launch: dispatching via executeCooperative "
                "(grid-wide barriers enabled via vgre_jit_syncgrid)");

  uint64_t flopsPerBlock = 0;
  uint64_t bytesPerBlock = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto irIt = kernelIRCache_.find(id);
    if (irIt != kernelIRCache_.end()) {
        const auto &ir = irIt->second;
        if (ir.staticFlopCount > 0) {
            flopsPerBlock = blockDim.total() * ir.staticFlopCount;
        } else if (!ir.flopCountVerified && ir.estimatedInstructionCount > 0) {
            // Heuristic only when LLVM IR analysis was unavailable.
            flopsPerBlock = blockDim.total() *
                std::max(uint64_t(1), ir.estimatedInstructionCount / 5);
        } else {
            flopsPerBlock = blockDim.total(); // minimum baseline
        }
    }
  }

  return executor_->executeCooperative(fn, gridDim, blockDim,
                                       safeArgs->data(), sharedMem,
                                       flopsPerBlock, bytesPerBlock);
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
  GraphNodeType type = GraphNodeType::KERNEL;
  // Kernel data
  OwnedFusedLaunchArgs kernelArgs;
  dim3 gridDim;
  dim3 blockDim;
  bool usesSyncthreads = false;
  // Memcpy data
  void *dst = nullptr;
  void *src = nullptr;
  size_t count = 0;
  int kind = 0;
  // Conditional node data (CONDITIONAL type only)
  int (*condFn)(void *) = nullptr;
  void *condCtx = nullptr;
  GraphCondType condType = GraphCondType::IF;
  unsigned int maxIterations = 65536;
  // Pre-compiled body ops; called inline to avoid deadlock with the stream scheduler.
  std::function<void(runtime::CPUParallelExecutor *, MemoryManager *)> bodyExec;
};

// ── Graph dispatch helpers ─────────────────────────────────────────────────

// Topological sort for a flat node vector.  Returns false if a cycle is found
// or a dependency references a non-existent node.
static bool topoSortNodes(const std::vector<GraphNode> &nodes,
                          std::vector<size_t> &outOrder) {
  std::unordered_map<uint64_t, size_t> idx;
  idx.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].nodeId == 0) return false;
    idx[nodes[i].nodeId] = i;
  }
  std::vector<int> indeg(nodes.size(), 0);
  std::vector<std::vector<size_t>> adj(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto dep : nodes[i].deps) {
      auto it = idx.find(dep);
      if (it == idx.end()) return false;
      adj[it->second].push_back(i);
      indeg[i]++;
    }
  }
  std::queue<size_t> q;
  for (size_t i = 0; i < indeg.size(); ++i)
    if (indeg[i] == 0) q.push(i);
  outOrder.reserve(nodes.size());
  while (!q.empty()) {
    size_t cur = q.front(); q.pop();
    outOrder.push_back(cur);
    for (size_t nxt : adj[cur])
      if (--indeg[nxt] == 0) q.push(nxt);
  }
  return outOrder.size() == nodes.size();
}

// Execute pre-compiled ops inline (no scheduler submission) so CONDITIONAL
// body graphs run synchronously inside the parent stream task.
static void executeOpsInline(const std::vector<NativeGraphOperation> &ops,
                              runtime::CPUParallelExecutor *exec,
                              MemoryManager *mm) {
  for (size_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops[i];
    VGRE_LOG_INFO("RuntimeEngine",
                  "Worker thread executing op " + std::to_string(i) +
                      " type=" + std::to_string(static_cast<int>(op.type)));
    if (op.type == GraphNodeType::KERNEL) {
      auto start = std::chrono::steady_clock::now();
      VGRE_LOG_INFO("RuntimeEngine",
                    "Dispatching kernel '" + op.kernelArgs.name + "'");
      uint32_t blocksTotal = op.gridDim.total();
      uint64_t flopsPerBlock = blocksTotal > 0 ? op.kernelArgs.flops / blocksTotal : 0;
      uint64_t bytesPerBlock = blocksTotal > 0 ? op.kernelArgs.memBytes / blocksTotal : 0;
      exec->execute(op.kernelArgs.fn, op.gridDim, op.blockDim,
                    const_cast<void **>(op.kernelArgs.argPtrs.data()),
                    op.kernelArgs.sharedMemBytes,
                    flopsPerBlock, bytesPerBlock, dim3(0, 0, 0),
                    op.usesSyncthreads);
      auto end = std::chrono::steady_clock::now();
      double ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
          op.kernelArgs.name, op.blockDim.total(), 8, ms,
          op.kernelArgs.memBytes, op.kernelArgs.flops);
    } else if (op.type == GraphNodeType::MEMCPY) {
      if (op.kind == VGRE_MEMCPY_HOST_TO_DEVICE)
        mm->copyHostToDevice(op.dst, op.src, op.count);
      else if (op.kind == VGRE_MEMCPY_DEVICE_TO_HOST)
        mm->copyDeviceToHost(op.dst, op.src, op.count);
      else if (op.kind == VGRE_MEMCPY_DEVICE_TO_DEVICE)
        mm->copyDeviceToDevice(op.dst, op.src, op.count);
    } else if (op.type == GraphNodeType::CONDITIONAL) {
      if (!op.condFn) continue;
      if (op.condType == GraphCondType::IF) {
        if (op.condFn(op.condCtx) != 0 && op.bodyExec)
          op.bodyExec(exec, mm);
      } else { // WHILE
        unsigned int iter = 0;
        while (iter < op.maxIterations && op.condFn(op.condCtx) != 0) {
          if (op.bodyExec) op.bodyExec(exec, mm);
          ++iter;
        }
        if (iter >= op.maxIterations) {
          VGRE_LOG_WARN("RuntimeEngine",
                        "WHILE conditional node reached maxIterations limit (" +
                            std::to_string(op.maxIterations) + ")");
        }
      }
    }
  }
}

// Recursively prefetch body graph nodes for all CONDITIONAL nodes in a graph.
// Acquires only GraphManager's mutex — NOT RuntimeEngine's mutex — so this is
// safe to call before the main compilation lock is acquired, avoiding ABBA deadlock.
static void prefetchBodyGraphs(
    const std::vector<GraphNode> &nodes, GraphManager &gm,
    std::unordered_map<GraphId, std::vector<GraphNode>> &cache) {
  for (const auto &node : nodes) {
    if (node.type == GraphNodeType::CONDITIONAL && node.bodyGraphId != 0 &&
        cache.find(node.bodyGraphId) == cache.end()) {
      std::vector<GraphNode> body;
      if (gm.getGraphNodes(node.bodyGraphId, body) == VGREResult::SUCCESS) {
        cache[node.bodyGraphId] = body; // store before recursing (cycle guard)
        prefetchBodyGraphs(body, gm, cache);
      }
    }
  }
}

VGREResult RuntimeEngine::dispatchGraphNodes(const std::vector<GraphNode> &nodes,
                                             StreamId stream) {
  if (nodes.empty()) return VGREResult::SUCCESS;

  // ── Step 1: topo-sort outer nodes (no locks needed) ─────────────────────
  std::vector<size_t> topoOrder;
  if (!topoSortNodes(nodes, topoOrder)) {
    VGRE_LOG_ERROR("RuntimeEngine",
                   "Graph dispatch failed: dependency cycle or invalid dep.");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Build a topo-sorted node vector for convenience.
  std::vector<GraphNode> sortedNodes;
  sortedNodes.reserve(topoOrder.size());
  for (size_t idx : topoOrder) sortedNodes.push_back(nodes[idx]);

  // ── Step 2: prefetch body graphs (acquires only GM mutex, NOT RE mutex) ──
  std::unordered_map<GraphId, std::vector<GraphNode>> bodyCache;
  if (graphManager_) {
    prefetchBodyGraphs(sortedNodes, *graphManager_, bodyCache);
  }

  // ── Step 3: compile under RE lock ────────────────────────────────────────
  // Uses a recursive std::function so nested CONDITIONAL body subgraphs are
  // compiled eagerly and stored in op.bodyExec for inline execution later.
  using CompileFn = std::function<VGREResult(const std::vector<GraphNode> &,
                                             std::vector<NativeGraphOperation> &)>;

  auto compiledOps = std::make_shared<std::vector<NativeGraphOperation>>();
  compiledOps->reserve(sortedNodes.size());

  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_) return VGREResult::ERR_NOT_INITIALIZED;

    CompileFn compileNodes;
    compileNodes = [&](const std::vector<GraphNode> &input,
                       std::vector<NativeGraphOperation> &outOps) -> VGREResult {
      for (size_t ni = 0; ni < input.size(); ++ni) {
        const auto &node = input[ni];
        NativeGraphOperation op;
        op.type = node.type;

        if (node.type == GraphNodeType::KERNEL) {
          KernelId actualId = node.kernelId;
          if (actualId == 0) {
            auto nameIt = kernelNames_.find(node.kernelName);
            if (nameIt != kernelNames_.end()) actualId = nameIt->second;
          }

          auto it = kernelCache_.find(actualId);
          if (it == kernelCache_.end()) {
            auto pendingIt = pendingKernels_.find(actualId);
            if (pendingIt != pendingKernels_.end()) {
              VGRE_LOG_INFO("RuntimeEngine",
                            "Graph dispatch waiting for JIT of kernel: " +
                                node.kernelName + " (ID " +
                                std::to_string(actualId) + ")");
              vgre::JITResult jres = pendingIt->second.get();
              CompiledKernelFn fn = jres.fn;
              if (!fn) {
                VGRE_LOG_ERROR("RuntimeEngine",
                               "JIT returned NULL for kernel: " + node.kernelName);
              } else {
                auto &irEntry = kernelIRCache_[actualId];
                irEntry.sharedMemSize = jres.sharedMemSize;
                irEntry.argSizes = jres.argSizes;
                irEntry.estimatedInstructionCount = jres.estimatedInstructionCount;
                irEntry.staticFlopCount = jres.staticFlopCount;
              }
              kernelCache_[actualId] = fn;
              pendingKernels_.erase(pendingIt);
              it = kernelCache_.find(actualId);
            } else {
              VGRE_LOG_ERROR("RuntimeEngine",
                             "Graph dispatch failed: Kernel " + node.kernelName +
                                 " (ID " + std::to_string(actualId) +
                                 ") not found in cache or pending list.");
              return VGREResult::ERR_INVALID_KERNEL;
            }
          }

          auto irIt = kernelIRCache_.find(actualId);
          std::string kName =
              (irIt != kernelIRCache_.end()) ? irIt->second.name : "unknown";

          op.gridDim = node.gridDim;
          op.blockDim = node.blockDim;
          op.kernelArgs.fn = it->second;
          if (!op.kernelArgs.fn)
            VGRE_LOG_ERROR("RuntimeEngine",
                           "Assigned EMPTY function to op for kernel: " +
                               node.kernelName);
          op.kernelArgs.name = kName;
          op.kernelArgs.argValues.resize(node.capturedArgs.size());
          op.kernelArgs.argPtrs.resize(node.capturedArgs.size(), nullptr);

          size_t totalThreads = node.gridDim.total() * node.blockDim.total();
          size_t memBytes = 0;
          if (irIt != kernelIRCache_.end()) {
            for (size_t ai = 0; ai < irIt->second.argTypes.size() &&
                                 ai < node.capturedArgs.size(); ++ai) {
              if (irIt->second.argTypes[ai] == ArgType::POINTER) {
                uint64_t raw = 0;
                std::memcpy(&raw, node.capturedArgs[ai].data(), sizeof(uint64_t));
                void *ptr = reinterpret_cast<void *>(raw);
                if (memoryManager_ && memoryManager_->isValidHandle(ptr))
                  memBytes += memoryManager_->getAllocationSize(ptr);
              } else if (irIt->second.argTypes[ai] == ArgType::STRUCT) {
                if (ai < irIt->second.argSizes.size())
                  memBytes += irIt->second.argSizes[ai] * totalThreads;
              }
            }
          }
          op.kernelArgs.memBytes = memBytes;
          op.kernelArgs.flops =
              totalThreads *
              ((irIt != kernelIRCache_.end())
                   ? irIt->second.estimatedInstructionCount
                   : 1);
          op.kernelArgs.sharedMemBytes =
              (irIt != kernelIRCache_.end()) ? irIt->second.sharedMemSize : 0;
          op.usesSyncthreads =
              (irIt != kernelIRCache_.end()) ? irIt->second.usesSyncthreads : false;

          for (size_t ai = 0; ai < node.capturedArgs.size(); ++ai) {
            size_t copySize = node.capturedArgs[ai].size();
            if (irIt != kernelIRCache_.end() && ai < irIt->second.argTypes.size()) {
              if (ai < irIt->second.argSizes.size() &&
                  irIt->second.argSizes[ai] > 0) {
                copySize = irIt->second.argSizes[ai];
              } else {
                switch (irIt->second.argTypes[ai]) {
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
            op.kernelArgs.argValues[ai].resize(copySize);
            if (copySize > 0 && !node.capturedArgs[ai].empty()) {
              ::memcpy(op.kernelArgs.argValues[ai].data(),
                       node.capturedArgs[ai].data(),
                       std::min(copySize, node.capturedArgs[ai].size()));
            } else {
              ::memset(op.kernelArgs.argValues[ai].data(), 0, copySize);
            }
            op.kernelArgs.argPtrs[ai] = op.kernelArgs.argValues[ai].data();
          }

        } else if (node.type == GraphNodeType::MEMCPY) {
          op.dst = node.dst;
          op.src = node.src;
          op.count = node.count;
          op.kind = node.kind;

        } else if (node.type == GraphNodeType::CONDITIONAL) {
          op.condFn = node.condFn;
          op.condCtx = node.condCtx;
          op.condType = node.condType;
          op.maxIterations = node.maxIterations;

          // Pre-compile body subgraph (nodes already prefetched before the lock).
          if (node.bodyGraphId != 0) {
            auto cacheIt = bodyCache.find(node.bodyGraphId);
            if (cacheIt != bodyCache.end() && !cacheIt->second.empty()) {
              std::vector<size_t> bodyOrder;
              if (!topoSortNodes(cacheIt->second, bodyOrder)) {
                VGRE_LOG_ERROR("RuntimeEngine",
                               "Conditional body graph " +
                                   std::to_string(node.bodyGraphId) +
                                   " has a dependency cycle.");
                return VGREResult::ERR_INVALID_VALUE;
              }
              std::vector<GraphNode> sortedBody;
              sortedBody.reserve(bodyOrder.size());
              for (size_t bi : bodyOrder)
                sortedBody.push_back(cacheIt->second[bi]);

              auto bodyOpsPtr =
                  std::make_shared<std::vector<NativeGraphOperation>>();
              bodyOpsPtr->reserve(sortedBody.size());
              auto bres = compileNodes(sortedBody, *bodyOpsPtr);
              if (bres != VGREResult::SUCCESS) return bres;

              // Capture by value so the stream task lambda owns the body.
              op.bodyExec = [bodyOpsPtr](runtime::CPUParallelExecutor *e,
                                         MemoryManager *m) {
                executeOpsInline(*bodyOpsPtr, e, m);
              };
            }
          }
        }

        outOps.push_back(std::move(op));

        // Re-fix argPtrs after push_back (vector may relocate).
        auto &finalOp = outOps.back();
        if (finalOp.type == GraphNodeType::KERNEL) {
          if (!finalOp.kernelArgs.fn)
            VGRE_LOG_ERROR("RuntimeEngine",
                           "Function became EMPTY after push_back for kernel: " +
                               node.kernelName);
          for (size_t ai = 0; ai < finalOp.kernelArgs.argValues.size(); ++ai)
            finalOp.kernelArgs.argPtrs[ai] =
                finalOp.kernelArgs.argValues[ai].data();
        }
      }
      return VGREResult::SUCCESS;
    };

    auto res = compileNodes(sortedNodes, *compiledOps);
    if (res != VGREResult::SUCCESS) return res;
  }

  // ── Step 4: submit stream task ────────────────────────────────────────────
  VGRE_LOG_INFO("RuntimeEngine",
                "Submitting Native Parallel Graph DAG of size " +
                    std::to_string(nodes.size()) + " on stream " +
                    std::to_string(stream));

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
  // Sequential iteration avoids scheduler deadlock; body subgraphs run inline.
  scheduler_->submitStreamTask(
      stream,
      [exec, mm, compiledOps]() { executeOpsInline(*compiledOps, exec, mm); },
      streamPriority);

  return VGREResult::SUCCESS;
}

// ── Synchronization ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::synchronize() {
  Scheduler *sched = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !scheduler_)
      return VGREResult::ERR_NOT_INITIALIZED;
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
      return VGREResult::ERR_NOT_INITIALIZED;
    sched = scheduler_;
    if (stream != 0) {
      if (currentDeviceId_ < 0 ||
          currentDeviceId_ >= static_cast<DeviceId>(devices_.size())) {
        return VGREResult::ERR_INVALID_DEVICE;
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
      return VGREResult::ERR_NOT_INITIALIZED;
    mm = memoryManager_.get();
    deviceId = currentDeviceId_;
  }
  return mm->allocate(size, outHandle, deviceId);
}

VGREResult RuntimeEngine::mallocManaged(size_t size, MemoryHandle &outHandle,
                                        unsigned int flags) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

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
      return VGREResult::ERR_NOT_INITIALIZED;
    if (device < 0 || peerDevice < 0 ||
        device >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
    }
    mm = memoryManager_.get();
  }
  if (!canAccess)
    return VGREResult::ERR_INVALID_VALUE;

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
      return VGREResult::ERR_NOT_INITIALIZED;
    current = currentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
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
      return VGREResult::ERR_NOT_INITIALIZED;
    current = currentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
    }
    mm = memoryManager_.get();
  }
  return mm->disablePeerAccess(current, peerDevice);
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────
VGREResult RuntimeEngine::graphCreate(GraphId &outGraph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->createGraph(outGraph);
}

VGREResult RuntimeEngine::graphClone(GraphId srcGraph, GraphId &outCloneGraph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->cloneGraph(srcGraph, outCloneGraph);
}

VGREResult RuntimeEngine::streamBeginCapture(StreamId stream) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  if (captureState_.count(stream))
    return VGREResult::ERR_INVALID_VALUE;

  GraphId graph;
  auto r = graphManager_->createGraph(graph);
  if (r != VGREResult::SUCCESS)
    return r;

  captureState_[stream] = graph;
  lastCapturedNodeId_[stream] = 0; // Initialize for implicit deps
  VGRE_LOG_INFO("RuntimeEngine",
                "Started capture on stream " + std::to_string(stream));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamEndCapture(StreamId stream, GraphId &outGraph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  auto it = captureState_.find(stream);
  if (it == captureState_.end())
    return VGREResult::ERR_INVALID_VALUE;

  outGraph = it->second;
  captureState_.erase(it);
  lastCapturedNodeId_.erase(stream);
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
    return VGREResult::ERR_NOT_INITIALIZED;
  
  auto it = captureState_.find(stream);
  if (it == captureState_.end())
    return VGREResult::ERR_INVALID_VALUE;

  VGRE_LOG_DEBUG("RuntimeEngine", "Capturing memcpy on stream " + std::to_string(stream));
  // We must cast away constness for the graph manager (which copies data anyway)
  return graphManager_->addMemcpyNode(it->second, dst, const_cast<void*>(src), count, kind);
}

VGREResult RuntimeEngine::graphInstantiate(GraphId graph,
                                           GraphExecId &outExec) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->instantiate(graph, outExec);
}

VGREResult RuntimeEngine::graphUpdateExec(GraphExecId exec, GraphId graph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->updateExec(exec, graph);
}

VGREResult RuntimeEngine::graphLaunch(GraphExecId exec, StreamId stream) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || !graphManager_)
      return VGREResult::ERR_NOT_INITIALIZED;
  }
  return graphManager_->launch(exec, stream);
}

VGREResult RuntimeEngine::graphDestroy(GraphId graph) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->destroyGraph(graph);
}

VGREResult RuntimeEngine::graphExecDestroy(GraphExecId exec) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->destroyGraphExec(exec);
}

VGREResult RuntimeEngine::graphAddKernelNode(
    GraphId graph, KernelId kernelId, const std::string &name,
    const dim3 &grid, const dim3 &block, void **args,
    const std::vector<ArgType> &argTypes, const std::vector<uint64_t> &deps,
    uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;

  KernelId actualId = kernelId;
  if (actualId == 0) {
    auto nameIt = kernelNames_.find(name);
    if (nameIt != kernelNames_.end()) {
      actualId = nameIt->second;
    }
  }

  auto res = graphManager_->addKernelNodeWithDepsOut(
      graph, actualId, name, grid, block, args, argTypes, deps, outNodeId);
  if (res == VGREResult::SUCCESS && isStreamCapturing(0)) { // 0 is default stream
      lastCapturedNodeId_[0] = outNodeId;
  }
  return res;
}

VGREResult RuntimeEngine::graphAddMemcpyNode(
    GraphId graph, void *dst, void *src, size_t count, int kind,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->addMemcpyNodeWithDepsOut(graph, dst, src, count, kind,
                                                 deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddConditionalNode(
    GraphId graph, int (*condFn)(void *), void *condCtx, GraphId bodyGraph,
    GraphCondType condType, unsigned int maxIterations,
    const std::vector<uint64_t> &deps, uint64_t &outNodeId) {
  // Release RuntimeEngine lock before calling graphManager to avoid ABBA
  // deadlock (GraphManager::addKernelNodeWithDepsOut holds GM lock → RE lock).
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->addConditionalNodeWithDepsOut(
      graph, condFn, condCtx, bodyGraph, condType, maxIterations, deps, outNodeId);
}

VGREResult RuntimeEngine::graphAddDependency(GraphId graph, uint64_t nodeId,
                                             uint64_t dependsOn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->addDependency(graph, nodeId, dependsOn);
}

VGREResult RuntimeEngine::graphUpdateKernelNode(
    GraphId graph, uint64_t nodeId, void **args,
    const std::vector<ArgType> &argTypes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->updateKernelNodeArgs(graph, nodeId, args, argTypes);
}

VGREResult RuntimeEngine::graphUpdateMemcpyNode(GraphId graph, uint64_t nodeId,
                                                void *dst, void *src,
                                                size_t count, int kind) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->updateMemcpyNode(graph, nodeId, dst, src, count, kind);
}

// ── Sub-system access ──────────────────────────────────────────────────────
MemoryManager &RuntimeEngine::getMemoryManager() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !memoryManager_) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *memoryManager_;
}

Scheduler &RuntimeEngine::getScheduler() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !scheduler_) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return *scheduler_;
}

VirtualGPUDevice &RuntimeEngine::getDevice() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || devices_.empty()) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  if (currentDeviceId_ < 0 ||
      currentDeviceId_ >= static_cast<DeviceId>(devices_.size())) {
    throw VGREException(VGREResult::ERR_INVALID_DEVICE, "Invalid device ID");
  }
  return *devices_[currentDeviceId_];
}

VirtualGPUDevice &RuntimeEngine::getDevice(DeviceId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    VGRE_LOG_ERROR("RuntimeEngine", "Invalid device ID: " + std::to_string(id) + " (Device Count: " + std::to_string(devices_.size()) + ")");
    throw VGREException(VGREResult::ERR_INVALID_DEVICE, "Invalid device ID");
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
