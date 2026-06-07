#include "vgre/core/runtime_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/compiler/kernel_fusion_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/compiler/kernel_parser.h"
#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/stream_dep_tracker.h"
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
#include "vgre/common/os_backend.h"

namespace vgre {
namespace core {

// Thread-local device binding: default to device 0.
thread_local DeviceId RuntimeEngine::tlCurrentDeviceId_ = 0;

// ── Constructor / Destructor ───────────────────────────────────────────────
RuntimeEngine::RuntimeEngine() = default;
RuntimeEngine::~RuntimeEngine() {
  // Do NOT call shutdown() from the destructor.  During static-storage
  // teardown the destruction order of Meyers singletons is undefined;
  // calling IPCManager / TCPClusterManager / Scheduler methods here can
  // deadlock or access already-destroyed objects.  All explicit cleanup must
  // go through vgre_shutdown() which is called before static destruction.
}

// ── Initialization ─────────────────────────────────────────────────────────
VGREResult RuntimeEngine::initialize() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (initialized_)
    return VGREResult::SUCCESS;

#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
      VGRE_LOG_ERROR("RuntimeEngine", "Failed to initialize Winsock (WSAStartup failed)");
  }
#endif

  VGRE_LOG_INFO("RuntimeEngine", "Initializing VGRE Runtime Engine...");

  // Force StreamDepTracker construction BEFORE Scheduler.
  // Meyers singletons are destroyed in reverse construction order.  Workers
  // call StreamDepTracker::notifyKernelComplete during their final task loop,
  // so StreamDepTracker must outlive the Scheduler (which joins workers in its
  // dtor).  Constructing StreamDepTracker first ensures it is destroyed last.
  StreamDepTracker::instance();

  // Create sub-systems
  scheduler_ = &Scheduler::instance();
  parser_ = std::make_unique<compiler::ClangKernelParser>();
  translator_ = std::make_unique<compiler::LLVMTranslationEngine>();
  executor_ = std::make_unique<runtime::CPUParallelExecutor>(
      scheduler_->getThreadCount());
  vectorEngine_ = std::make_unique<runtime::VectorEngine>();
  graphManager_ = std::make_unique<GraphManager>();

  // Create virtual devices based on hardware resources or Environment Override.
  // VGRE_VIRTUAL_DEVICE_COUNT is the canonical env var; VGRE_DEVICE_COUNT is
  // a legacy alias.  Both accept an integer 1..8.
  int deviceCount = 1;
  const char *envCount = vgre_get_config("VGRE_VIRTUAL_DEVICE_COUNT");
  if (!envCount) envCount = vgre_get_config("VGRE_DEVICE_COUNT");
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

  // Safety cap: max 8 virtual devices per spec (Track H).
  deviceCount = std::max(1, std::min(deviceCount, 8));

  VGRE_LOG_INFO("RuntimeEngine",
                "Creating " + std::to_string(deviceCount) +
                    " hardware-aligned virtual devices...");

  // Partition host threads equally across devices (minimum 1 per device).
  const int totalCores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  const int threadsPerDevice = std::max(1, totalCores / deviceCount);

  for (int i = 0; i < deviceCount; ++i) {
    auto dev = std::make_unique<VirtualGPUDevice>(i);
    dev->detectHardware();

    if (i == 0) {
      vgre::DeviceProperties dp = dev->getProperties();
      auto& aee = advanced::AdaptiveExecutionEngine::instance();
      aee.updateHardwareMetrics(totalCores, dp.clockRate / 1000000.0, 0.0);
    }
    dev->createContext();

    // Per-device MemoryManager: each device gets its share of the total memory.
    const size_t devMem = devices_.empty()
        ? dev->getProperties().totalGlobalMem
        : (dev->getProperties().totalGlobalMem / static_cast<size_t>(deviceCount));
    deviceMemManagers_.push_back(std::make_unique<MemoryManager>(devMem));

    // Per-device Scheduler: device 0 uses the global Scheduler::instance() so
    // that existing direct Scheduler::instance() callers still route correctly.
    // Devices 1..N get freshly constructed schedulers with partitioned threads.
    if (i == 0) {
      deviceSchedulers_.push_back(&Scheduler::instance());
    } else {
      ownedDeviceSchedulers_.push_back(std::make_unique<Scheduler>(threadsPerDevice));
      deviceSchedulers_.push_back(ownedDeviceSchedulers_.back().get());
    }

    devices_.push_back(std::move(dev));
  }

  tlCurrentDeviceId_ = 0;

  // scheduler_ stays pointing at the global singleton (device 0 scheduler).
  scheduler_ = &Scheduler::instance();

  initialized_ = true;

  // Phase 10: Controllable Background Tasks (Zero-Simulation Hardening)
  // Register with global IPC service as a client by default, unless disabled.
  const char* ipc_mode = vgre_get_config("VGRE_IPC_MODE");
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
  // Export profiler trace before tearing down subsystems.
  {
    const char* tracePath = ::getenv("VGRE_TRACE_PATH");
    std::string traceFile = tracePath ? std::string(tracePath)
                                      : vgre::os::get_temp_dir() + "/vgre_trace.json";
    advanced::RuntimeProfiler::instance().exportToFile(traceFile);
  }

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
  vgre::advanced::TCPClusterManager::instance().shutdown();

  scheduler_ = nullptr;
  deviceSchedulers_.clear();
  // Destroy owned schedulers (devices 1..N) in reverse order so their worker
  // threads join cleanly before any shared state is torn down.
  for (int i = static_cast<int>(ownedDeviceSchedulers_.size()) - 1; i >= 0; --i) {
    ownedDeviceSchedulers_[i].reset();
  }
  ownedDeviceSchedulers_.clear();
  deviceMemManagers_.clear();
  executor_.reset();
  vectorEngine_.reset();
  translator_.reset();
  parser_.reset();
  devices_.clear();

  kernelCache_.clear();
  kernelIRCache_.clear();
  kernelFnAddrMap_.clear();
  pendingKernels_.clear();
  kernelNames_.clear();
  warnedSyncthreads_.clear();
  captureState_.clear();
  lastCapturedNodeId_.clear();
  captureSeedDeps_.clear();
  nextKernelId_ = 1;
  tlCurrentDeviceId_ = 0;
  initialized_ = false;

  VGRE_LOG_INFO("RuntimeEngine", "Shutdown complete");

#ifdef _WIN32
  WSACleanup();
#endif

  return VGREResult::SUCCESS;
}

// ── Device management ──────────────────────────────────────────────────────
int RuntimeEngine::getDeviceCount() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return static_cast<int>(devices_.size());
}

VGREResult RuntimeEngine::setDevice(DeviceId id) {
  // Validate under the lock, then write to the thread-local.
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
      VGRE_LOG_ERROR("RuntimeEngine",
          "Invalid device ID: " + std::to_string(id) +
          " (Device Count: " + std::to_string(devices_.size()) + ")");
      return VGREResult::ERR_INVALID_DEVICE;
    }
  }
  tlCurrentDeviceId_ = id;
  VGRE_LOG_DEBUG("RuntimeEngine", "Thread bound to device " + std::to_string(id));
  return VGREResult::SUCCESS;
}

DeviceId RuntimeEngine::getDeviceId() const {
  return tlCurrentDeviceId_;
}

// ── Per-device sub-system access ───────────────────────────────────────────
Scheduler &RuntimeEngine::currentScheduler() {
  DeviceId id = tlCurrentDeviceId_;
  if (id < 0 || id >= static_cast<DeviceId>(deviceSchedulers_.size()))
    id = 0;
  return *deviceSchedulers_[static_cast<size_t>(id)];
}

MemoryManager &RuntimeEngine::currentMemoryManager() {
  DeviceId id = tlCurrentDeviceId_;
  if (id < 0 || id >= static_cast<DeviceId>(deviceMemManagers_.size()))
    id = 0;
  return *deviceMemManagers_[static_cast<size_t>(id)];
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

  // Deduplication: if the kernel is already registered by name, return its
  // existing ID immediately.  This prevents an uninitialized outId (non-zero
  // garbage) from colliding with a previously-compiled kernel's cache entry.
  {
    auto nameIt = kernelNames_.find(name);
    if (nameIt != kernelNames_.end()) {
      outId = nameIt->second;
      VGRE_LOG_INFO("RuntimeEngine",
                    "Kernel '" + name + "' already registered (ID=" +
                        std::to_string(outId) + "), returning cached entry");
      return VGREResult::SUCCESS;
    }
  }

  // Parse kernel source
  KernelIR ir;
  auto r = parser_->parse(name, source, ir);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("RuntimeEngine", "Failed to parse kernel: " + name);
    return r;
  }

#ifdef ENABLE_VGRE_KERNEL_FUSION
  // Kernel Fusion Engine: detect and register fused variants
  {
    auto& fusion = compiler::KernelFusionEngine::instance();
    auto meta = fusion.tryFuse(ir);
    if (meta.pattern != compiler::FusionPattern::NONE) {
      VGRE_LOG_INFO("RuntimeEngine",
                    "Fusion pattern detected for '" + name + "': " +
                    std::to_string(static_cast<int>(meta.pattern)));
      auto fusedSrc = fusion.generateFusedSource(meta, ir);
      if (!fusedSrc.empty()) {
        KernelIR fusedIr;
        auto fr = parser_->parse(meta.fusedKernelName, fusedSrc, fusedIr);
        if (fr == VGREResult::SUCCESS) {
          KernelId fusedId = nextKernelId_++;
          kernelIRCache_[fusedId] = fusedIr;
          pendingKernels_[fusedId] = translator_->prepare(kernelIRCache_[fusedId]);
          kernelNames_[meta.fusedKernelName] = fusedId;
          VGRE_LOG_INFO("RuntimeEngine",
                        "Fused kernel '" + meta.fusedKernelName +
                        "' registered with ID " + std::to_string(fusedId));
        }
      }
    }
  }
#endif // ENABLE_VGRE_KERNEL_FUSION

  // Translate to executable
  // If the caller provided a non-zero kernel id, treat it as authoritative.
  // This is required for production TCPCluster: the master assigns kernel IDs
  // and workers must preserve them to keep dispatch and result tracking
  // consistent across processes.
  // Use caller-provided ID only when it looks like a deliberate pre-assignment
  // (TCPCluster master assigns IDs to workers).  Any outId >= nextKernelId_
  // is a forward reference: extend the counter.  outId == 0 always allocates
  // a fresh ID, preventing uninitialized-variable collisions.
  KernelId id;
  if (outId != 0 && outId < (nextKernelId_ + 65536)) {
    // TCP-cluster pre-assignment: honour only if the ID is free or already
    // belongs to THIS kernel (idempotent re-register).  If a different kernel
    // owns the ID (uninitialized-variable collision), allocate a fresh ID.
    auto existIt = kernelIRCache_.find(outId);
    if (existIt == kernelIRCache_.end() || existIt->second.name == name) {
      id = outId;
      if (id >= nextKernelId_) nextKernelId_ = id + 1;
    } else {
      VGRE_LOG_WARN("RuntimeEngine",
          "registerKernel: outId=" + std::to_string(outId) +
          " already owned by '" + existIt->second.name +
          "', ignoring and allocating fresh ID for '" + name + "'");
      id = nextKernelId_++;
    }
  } else {
    id = nextKernelId_++;
  }
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
  kernelFnAddrMap_[fn.get()] = outId;

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

VGREResult RuntimeEngine::registerPrecompiledKernel(const std::string &name,
                                                     CompiledKernelFn fn,
                                                     KernelId &outId) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;
  if (!fn)
    return VGREResult::ERR_INVALID_VALUE;

  auto existIt = kernelNames_.find(name);
  if (existIt != kernelNames_.end()) {
    outId = existIt->second;
    kernelCache_[outId] = fn;
    kernelFnAddrMap_[fn.get()] = outId;
    return VGREResult::SUCCESS;
  }

  outId = nextKernelId_++;
  kernelCache_[outId] = fn;
  kernelFnAddrMap_[fn.get()] = outId;
  kernelNames_[name] = outId;
  KernelIR ir;
  ir.name = name;
  ir.source = "__llvm_bitcode__";
  kernelIRCache_[outId] = ir;

  VGRE_LOG_INFO("RuntimeEngine", "Precompiled kernel '" + name +
                                     "' registered as ID " +
                                     std::to_string(outId));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::fuseKernels(const std::vector<KernelId> &ids,
                                      KernelId &outFusedId, std::string* outName) {
#ifndef ENABLE_VGRE_KERNEL_FUSION
  (void)ids; (void)outFusedId; (void)outName;
  return VGREResult::ERR_NOT_SUPPORTED;
#else
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
#endif // ENABLE_VGRE_KERNEL_FUSION
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

const KernelIR *RuntimeEngine::getKernelIRByFn(CompiledKernelFn fn) const {
  if (!fn) return nullptr;
  // CompiledKernelFn is a shared_ptr<std::function<...>>.
  // The reverse map stores the raw shared_ptr pointer as the key.
  KernelId id = lookupKernelIdByFn(static_cast<void *>(fn.get()));
  return (id != 0) ? getKernelIR(id) : nullptr;
}

KernelId RuntimeEngine::lookupKernelIdByName(const char* name) const {
  if (!name) return 0;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = kernelNames_.find(std::string(name));
  return (it != kernelNames_.end()) ? it->second : KernelId{0};
}

KernelId RuntimeEngine::lookupKernelIdByFn(void* fnPtr) const {
  if (!fnPtr) return 0;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = kernelFnAddrMap_.find(fnPtr);
  return (it != kernelFnAddrMap_.end()) ? it->second : KernelId{0};
}

VGREResult RuntimeEngine::unloadModule(ModuleHandle module) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

  return translator_->unloadModule(module);
}

// ── Kernel launch / cooperative / graph dispatch ───────────────────────────
// Implementations are in runtime_engine_launch.cpp and runtime_engine_graph.cpp

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
    if (!initialized_ || deviceSchedulers_.empty())
      return VGREResult::ERR_NOT_INITIALIZED;
    sched = &currentScheduler();
    if (stream != 0) {
      DeviceId id = tlCurrentDeviceId_;
      if (id < 0 || id >= static_cast<DeviceId>(devices_.size()))
        return VGREResult::ERR_INVALID_DEVICE;
      dev = devices_[id].get();
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
    if (!initialized_ || deviceMemManagers_.empty())
      return VGREResult::ERR_NOT_INITIALIZED;
    mm = &currentMemoryManager();
    deviceId = tlCurrentDeviceId_;
  }
  return mm->allocate(size, outHandle, deviceId);
}

VGREResult RuntimeEngine::mallocManaged(size_t size, MemoryHandle &outHandle,
                                        unsigned int flags) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || deviceMemManagers_.empty())
    return VGREResult::ERR_NOT_INITIALIZED;
  return currentMemoryManager().allocateManaged(size, outHandle,
                                                tlCurrentDeviceId_, flags);
}

VGREResult RuntimeEngine::deviceCanAccessPeer(DeviceId device,
                                              DeviceId peerDevice,
                                              int *canAccess) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || deviceMemManagers_.empty())
      return VGREResult::ERR_NOT_INITIALIZED;
    if (device < 0 || peerDevice < 0 ||
        device >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
    }
  }
  if (!canAccess)
    return VGREResult::ERR_INVALID_VALUE;

  // All virtual devices in the same process share address space — P2P is
  // always available (same as cudaDeviceCanAccessPeer returning 1).
  *canAccess = 1;
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::deviceEnablePeerAccess(DeviceId peerDevice) {
  MemoryManager *mm = nullptr;
  DeviceId current = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || deviceMemManagers_.empty())
      return VGREResult::ERR_NOT_INITIALIZED;
    current = tlCurrentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
    }
    mm = &currentMemoryManager();
  }
  return mm->enablePeerAccess(current, peerDevice);
}

VGREResult RuntimeEngine::deviceDisablePeerAccess(DeviceId peerDevice) {
  MemoryManager *mm = nullptr;
  DeviceId current = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!initialized_ || deviceMemManagers_.empty())
      return VGREResult::ERR_NOT_INITIALIZED;
    current = tlCurrentDeviceId_;
    if (current < 0 || current >= static_cast<DeviceId>(devices_.size()) ||
        peerDevice < 0 ||
        peerDevice >= static_cast<DeviceId>(devices_.size())) {
      return VGREResult::ERR_INVALID_DEVICE;
    }
    mm = &currentMemoryManager();
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
  lastCapturedNodeId_[stream] = 0;
  captureSeedDeps_.erase(stream);
  VGRE_LOG_INFO("RuntimeEngine",
                "Started capture on stream " + std::to_string(stream));
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamBeginCaptureToGraph(
    StreamId stream, uint64_t graphHandle,
    const std::vector<uint64_t> &dependencies) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  if (captureState_.count(stream))
    return VGREResult::ERR_INVALID_VALUE;

  GraphId graph = static_cast<GraphId>(graphHandle);
  if (!graphManager_->graphExists(graph)) {
    // Provided graph handle is unknown; fall back to creating a new one
    VGRE_LOG_WARN("RuntimeEngine",
                  "streamBeginCaptureToGraph: graph " + std::to_string(graph) +
                  " not found — allocating new graph");
    auto r = graphManager_->createGraph(graph);
    if (r != VGREResult::SUCCESS) return r;
  }

  captureState_[stream] = graph;
  lastCapturedNodeId_[stream] = 0;
  captureSeedDeps_[stream] = dependencies;
  VGRE_LOG_INFO("RuntimeEngine",
                "Started capture-to-graph on stream " + std::to_string(stream) +
                " into graph " + std::to_string(graph) +
                " (seed deps=" + std::to_string(dependencies.size()) + ")");
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
  captureSeedDeps_.erase(stream);
  VGRE_LOG_INFO("RuntimeEngine", "Ended capture on stream " +
                                     std::to_string(stream) + " -> Graph " +
                                     std::to_string(outGraph));
  return VGREResult::SUCCESS;
}

bool RuntimeEngine::isStreamCapturing(StreamId stream) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return captureState_.count(stream) > 0;
}

bool RuntimeEngine::getStreamCaptureInfo(StreamId stream,
                                          GraphId &outGraphId) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = captureState_.find(stream);
  if (it == captureState_.end()) {
    outGraphId = 0;
    return false;
  }
  outGraphId = it->second;
  return true;
}

bool RuntimeEngine::getStreamCaptureInfoV2(StreamId stream,
                                            GraphId &outGraphId,
                                            std::vector<uint64_t> &outDeps) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = captureState_.find(stream);
  if (it == captureState_.end()) {
    outGraphId = 0;
    outDeps.clear();
    return false;
  }
  outGraphId = it->second;

  // Current capture dependency frontier = lastCapturedNodeId_ for this stream.
  outDeps.clear();
  auto nodeIt = lastCapturedNodeId_.find(stream);
  if (nodeIt != lastCapturedNodeId_.end() && nodeIt->second != 0)
    outDeps.push_back(nodeIt->second);

  auto seedIt = captureSeedDeps_.find(stream);
  if (seedIt != captureSeedDeps_.end())
    for (auto d : seedIt->second)
      outDeps.push_back(d);

  return true;
}

VGREResult RuntimeEngine::streamUpdateCaptureDependencies(
    StreamId stream, const std::vector<uint64_t> &deps, bool replace) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (captureState_.find(stream) == captureState_.end())
    return VGREResult::ERR_INVALID_VALUE;

  if (replace) {
    captureSeedDeps_[stream] = deps;
    lastCapturedNodeId_.erase(stream);
  } else {
    auto &existing = captureSeedDeps_[stream];
    for (auto d : deps) existing.push_back(d);
  }
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::streamCopyAttributes(StreamId dst, StreamId src) {
  // VGRE stream attributes (priority, flags) are stored in the CUDART shim's
  // g_streamMeta table which is not accessible here.  The RuntimeEngine has
  // no additional per-stream state beyond capture, so this is a no-op from
  // the engine's perspective.  The CUDART shim layer handles the metadata copy.
  (void)dst; (void)src;
  if (!initialized_) return VGREResult::ERR_NOT_INITIALIZED;
  return VGREResult::SUCCESS;
}

VGREResult RuntimeEngine::recordMemcpyToGraph(StreamId stream, void *dst, const void *src, size_t count, int kind) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;

  auto it = captureState_.find(stream);
  if (it == captureState_.end())
    return VGREResult::ERR_INVALID_VALUE;

  VGRE_LOG_DEBUG("RuntimeEngine", "Capturing memcpy on stream " + std::to_string(stream));
  std::vector<uint64_t> deps;
  auto seedIt = captureSeedDeps_.find(stream);
  if (seedIt != captureSeedDeps_.end() && !seedIt->second.empty()) {
    deps = seedIt->second;
    seedIt->second.clear();
  } else {
    auto lastNodeIt = lastCapturedNodeId_.find(stream);
    if (lastNodeIt != lastCapturedNodeId_.end() && lastNodeIt->second != 0) {
      deps.push_back(lastNodeIt->second);
    }
  }

  uint64_t nodeId = 0;
  auto r = graphManager_->addMemcpyNodeWithDepsOut(
      it->second, dst, const_cast<void*>(src), count, kind, deps, nodeId);
  if (r == VGREResult::SUCCESS) {
    lastCapturedNodeId_[stream] = nodeId;
  }
  return r;
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

VGREResult RuntimeEngine::graphUpdateExecV2(GraphExecId exec, GraphId graph,
                                            const std::vector<uint64_t>& nodeIds) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || !graphManager_)
    return VGREResult::ERR_NOT_INITIALIZED;
  return graphManager_->updateExecV2(exec, graph, nodeIds);
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
  if (!initialized_ || deviceMemManagers_.empty()) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return currentMemoryManager();
}

Scheduler &RuntimeEngine::getScheduler() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || deviceSchedulers_.empty()) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  return currentScheduler();
}

VirtualGPUDevice &RuntimeEngine::getDevice() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!initialized_ || devices_.empty()) {
    throw VGREException(VGREResult::ERR_NOT_INITIALIZED,
                        "Runtime engine is not initialized");
  }
  DeviceId id = tlCurrentDeviceId_;
  if (id < 0 || id >= static_cast<DeviceId>(devices_.size())) {
    throw VGREException(VGREResult::ERR_INVALID_DEVICE, "Invalid device ID");
  }
  return *devices_[id];
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
  static RuntimeEngine inst;
  return inst;
}

} // namespace core
} // namespace vgre
