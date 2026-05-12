#include "vgre/core/runtime_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/runtime/cpu_parallel_executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

namespace vgre {
namespace core {

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
      auto seedIt = captureSeedDeps_.find(stream);
      if (seedIt != captureSeedDeps_.end() && !seedIt->second.empty()) {
        deps = seedIt->second;
        seedIt->second.clear(); // consume frontier on first captured node
      } else {
        auto lastNodeIt = lastCapturedNodeId_.find(stream);
        if (lastNodeIt != lastCapturedNodeId_.end() && lastNodeIt->second != 0) {
          deps.push_back(lastNodeIt->second);
        }
      }

      uint64_t newNodeId = 0;
      g_capture_stream_id = stream;  // propagate stream into GraphNode::streamId
      auto res = graphManager_->addKernelNodeWithDepsOut(captureIt->second, id,
                                          irIt->second.name, gridDim, blockDim,
                                          args, irIt->second.argTypes, deps, newNodeId);
      g_capture_stream_id = 0;

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
            kernelFnAddrMap_[fn.get()] = id;
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
      // LLVM IR analysis unavailable (precompiled module / JIT failed).
      // Use the AdaptiveExecutionEngine's calibrated flopPerInstruction ratio
      // which is measured from actual hardware via perf_event benchmarks.
      // Falls back to the static memory-access count fraction when calibration
      // has not yet run (startup).
      double fpi = advanced::AdaptiveExecutionEngine::instance().getFlopPerInstruction();
      uint64_t flopEst;
      if (fpi > 0.0) {
        flopEst = std::max(uint64_t(1),
            static_cast<uint64_t>(static_cast<double>(estimatedInstructionCount) * fpi));
      } else {
        // flopPerInstruction not yet calibrated: use 1 FP op per 4 instructions.
        // This is conservative but prevents telemetry inflation for non-compute kernels.
        flopEst = std::max(uint64_t(1), estimatedInstructionCount / 4);
      }
      VGRE_LOG_DEBUG("RuntimeEngine",
                     "FLOP count calibrated for kernel '" + kName +
                     "': " + std::to_string(flopEst) + " FLOPs/thread from " +
                     std::to_string(estimatedInstructionCount) + " instructions" +
                     (fpi > 0.0 ? " (ratio=" + std::to_string(fpi) + ")" : " (fallback)"));
      flopsPerBlock = blockDim.total() * flopEst;
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
      kernelFnAddrMap_[jres.fn.get()] = id;
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
        double fpi = advanced::AdaptiveExecutionEngine::instance().getFlopPerInstruction();
        uint64_t flopEst = (fpi > 0.0)
            ? std::max(uint64_t(1), static_cast<uint64_t>(ir.estimatedInstructionCount * fpi))
            : std::max(uint64_t(1), ir.estimatedInstructionCount / 4);
        ready[i].flopsPerBlock = ready[i].blockDim.total() * flopEst;
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
  // One std::thread per device. All threads wait on a condition_variable start-
  // gate so every device begins executeCooperative simultaneously. Using a cv
  // avoids the yield()-spin that wastes CPU between thread creation and launch.
  std::atomic<int>         readyCount{0};
  std::mutex               startMutex;
  std::condition_variable  startCv;
  std::vector<VGREResult>  results(N, VGREResult::SUCCESS);
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    threads.emplace_back(
        [this, &ready, &readyCount, &startMutex, &startCv, &results, N, i]() {
          // Announce this thread is ready.
          {
            std::unique_lock<std::mutex> lk(startMutex);
            readyCount.fetch_add(1, std::memory_order_release);
          }
          startCv.notify_all();
          // Block until ALL device threads have reached the gate.
          {
            std::unique_lock<std::mutex> lk(startMutex);
            startCv.wait(lk, [&readyCount, N] {
              return readyCount.load(std::memory_order_acquire) >= static_cast<int>(N);
            });
          }

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

      uint64_t newNodeId = 0;
      g_capture_stream_id = stream;  // propagate stream into GraphNode::streamId
      auto res = graphManager_->addKernelNodeWithDepsOut(captureIt->second, id,
                                          irIt->second.name, gridDim, blockDim,
                                          args, irIt->second.argTypes, deps, newNodeId);
      g_capture_stream_id = 0;

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
            kernelFnAddrMap_[fn.get()] = id;
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
            double fpi = advanced::AdaptiveExecutionEngine::instance().getFlopPerInstruction();
            uint64_t flopEst = (fpi > 0.0)
                ? std::max(uint64_t(1), static_cast<uint64_t>(ir.estimatedInstructionCount * fpi))
                : std::max(uint64_t(1), ir.estimatedInstructionCount / 4);
            flopsPerBlock = blockDim.total() * flopEst;
        } else {
            flopsPerBlock = blockDim.total(); // minimum baseline
        }
    }
  }

  return executor_->executeCooperative(fn, gridDim, blockDim,
                                       safeArgs->data(), sharedMem,
                                       flopsPerBlock, bytesPerBlock);
}

} // namespace core
} // namespace vgre
