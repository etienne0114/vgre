#include "vgre/core/runtime_engine.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"
#include "vgre/core/event.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/runtime/cpu_parallel_executor.h"

#include <chrono>
#include <cstring>
#include <functional>
#include <queue>
#include <unordered_map>

namespace vgre {
namespace core {

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
  // For SWITCH: bodyExecs[i] = subgraph for branch i; bodyExec = default (branch 0).
  std::function<void(runtime::CPUParallelExecutor *, MemoryManager *)> bodyExec;
  std::vector<std::function<void(runtime::CPUParallelExecutor *, MemoryManager *)>> bodyExecs;

  // MEMSET node
  int   memsetValue  = 0;
  size_t memsetPitch  = 0;
  size_t memsetWidth  = 0;
  size_t memsetHeight = 0;
  size_t memsetDepth  = 1;

  // HOST node
  void (*hostFn)(void *) = nullptr;
  void *hostUserData     = nullptr;

  // EVENT_RECORD / EVENT_WAIT
  void *eventHandle = nullptr;
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
      } else if (op.condType == GraphCondType::WHILE) {
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
      } else { // SWITCH — integer condition selects one of N child subgraphs
        int branch = op.condFn(op.condCtx);
        int nBranches = static_cast<int>(op.bodyExecs.size());
        // child[0] = default; child[1..N] = branches 0..N-1
        if (nBranches > 0) {
          int target = (branch >= 0 && branch < nBranches - 1) ? branch + 1 : 0;
          if (op.bodyExecs[target]) op.bodyExecs[target](exec, mm);
        } else if (op.bodyExec) {
          // Fallback: single body exec (behaves like IF with integer condition)
          if (branch != 0) op.bodyExec(exec, mm);
        }
      }

    } else if (op.type == GraphNodeType::MEMSET) {
      // 2D / 3D pitched fill: for each depth slice, iterate over rows.
      if (!op.dst || op.memsetWidth == 0 || op.memsetHeight == 0) continue;
      auto *base = static_cast<uint8_t *>(op.dst);
      size_t depth = (op.memsetDepth == 0) ? 1 : op.memsetDepth;
      auto ms0 = std::chrono::steady_clock::now();
      for (size_t d = 0; d < depth; ++d) {
        for (size_t h = 0; h < op.memsetHeight; ++h) {
          // Slice offset: d * (pitch * height); row offset: h * pitch.
          size_t sliceOff = d * op.memsetPitch * op.memsetHeight;
          memset(base + sliceOff + h * op.memsetPitch,
                      op.memsetValue, op.memsetWidth);
        }
      }
      {
        auto ms1 = std::chrono::steady_clock::now();
        auto &profiler = vgre::advanced::RuntimeProfiler::instance();
        if (profiler.isEnabled()) {
          size_t bytes = op.memsetPitch * op.memsetHeight * depth;
          double dur = std::chrono::duration<double, std::milli>(ms1 - ms0).count();
          vgre::advanced::ProfileEvent pev;
          pev.kernelName = "graph::memset";
          pev.durationMs = dur;
          pev.memoryBytes = bytes;
          pev.throughputGBps = (dur > 0.0) ? (bytes / 1e9) / (dur / 1000.0) : 0.0;
          pev.timestamp = ms1;
          profiler.recordEvent(pev);
          mm->recordMemoryBandwidth(bytes, dur);
        }
      }

    } else if (op.type == GraphNodeType::HOST) {
      auto ms0 = std::chrono::steady_clock::now();
      if (op.hostFn) op.hostFn(op.hostUserData);
      {
        auto ms1 = std::chrono::steady_clock::now();
        auto &profiler = vgre::advanced::RuntimeProfiler::instance();
        if (profiler.isEnabled()) {
          double dur = std::chrono::duration<double, std::milli>(ms1 - ms0).count();
          vgre::advanced::ProfileEvent pev;
          pev.kernelName = "graph::host";
          pev.durationMs = dur;
          pev.timestamp = ms1;
          profiler.recordEvent(pev);
        }
      }

    } else if (op.type == GraphNodeType::CHILD) {
      // Child graph body was pre-compiled into op.bodyExec.
      auto ms0 = std::chrono::steady_clock::now();
      if (op.bodyExec) op.bodyExec(exec, mm);
      {
        auto ms1 = std::chrono::steady_clock::now();
        auto &profiler = vgre::advanced::RuntimeProfiler::instance();
        if (profiler.isEnabled()) {
          double dur = std::chrono::duration<double, std::milli>(ms1 - ms0).count();
          vgre::advanced::ProfileEvent pev;
          pev.kernelName = "graph::child";
          pev.durationMs = dur;
          pev.timestamp = ms1;
          profiler.recordEvent(pev);
        }
      }

    } else if (op.type == GraphNodeType::EMPTY) {
      // No-op dependency placeholder — nothing to do.

    } else if (op.type == GraphNodeType::EVENT_RECORD) {
      // Mark the event as recorded (timestamp = now).
      if (op.eventHandle) {
        auto *ev = static_cast<vgre::core::Event *>(op.eventHandle);
        ev->record(0 /*streamId*/);
        auto &profiler = vgre::advanced::RuntimeProfiler::instance();
        if (profiler.isEnabled()) {
          vgre::advanced::ProfileEvent pev;
          pev.kernelName = "graph::event_record";
          pev.durationMs = 0.0;
          pev.timestamp = std::chrono::steady_clock::now();
          profiler.recordEvent(pev);
        }
      }

    } else if (op.type == GraphNodeType::EVENT_WAIT) {
      // Synchronously wait for the event to be recorded.
      if (op.eventHandle) {
        auto *ev = static_cast<vgre::core::Event *>(op.eventHandle);
        ev->synchronize();
      }

    } else if (op.type == GraphNodeType::MEMALLOC) {
      // Memory was pre-allocated at node-add time; no work to do here.
      // (The output pointer was written into *memAllocOutPtr at add time.)

    } else if (op.type == GraphNodeType::MEMFREE) {
      // Release the backing allocation that was pre-allocated.
      if (op.dst && mm)
        mm->free(op.dst);
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
    // CONDITIONAL and CHILD both reference a sub-graph via bodyGraphId.
    bool needsFetch = (node.type == GraphNodeType::CONDITIONAL ||
                       node.type == GraphNodeType::CHILD);
    if (needsFetch && node.bodyGraphId != 0 &&
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
              kernelFnAddrMap_[fn.get()] = actualId;
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
                memcpy(&raw, node.capturedArgs[ai].data(), sizeof(uint64_t));
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

        } else if (node.type == GraphNodeType::MEMSET) {
          op.dst          = node.dst;
          op.memsetValue  = node.memsetValue;
          op.memsetPitch  = node.memsetPitch;
          op.memsetWidth  = node.memsetWidth;
          op.memsetHeight = node.memsetHeight;
          op.memsetDepth  = node.memsetDepth;

        } else if (node.type == GraphNodeType::HOST) {
          op.hostFn       = node.hostFn;
          op.hostUserData = node.hostUserData;

        } else if (node.type == GraphNodeType::CHILD) {
          // Pre-compile child graph nodes.
          if (node.bodyGraphId != 0) {
            auto cacheIt = bodyCache.find(node.bodyGraphId);
            if (cacheIt != bodyCache.end() && !cacheIt->second.empty()) {
              std::vector<size_t> childOrder;
              if (topoSortNodes(cacheIt->second, childOrder)) {
                std::vector<GraphNode> sortedChild;
                sortedChild.reserve(childOrder.size());
                for (size_t ci : childOrder)
                  sortedChild.push_back(cacheIt->second[ci]);

                auto childOpsPtr =
                    std::make_shared<std::vector<NativeGraphOperation>>();
                childOpsPtr->reserve(sortedChild.size());
                auto cres = compileNodes(sortedChild, *childOpsPtr);
                if (cres == VGREResult::SUCCESS) {
                  op.bodyExec = [childOpsPtr](runtime::CPUParallelExecutor *e,
                                              MemoryManager *m) {
                    executeOpsInline(*childOpsPtr, e, m);
                  };
                }
              }
            }
          }

        } else if (node.type == GraphNodeType::EMPTY) {
          // No data to compile.

        } else if (node.type == GraphNodeType::EVENT_RECORD ||
                   node.type == GraphNodeType::EVENT_WAIT) {
          op.eventHandle = node.eventHandle;

        } else if (node.type == GraphNodeType::MEMALLOC) {
          // Pre-allocated; no compile-time work beyond marking the type.
          op.dst = node.memAllocPtr;

        } else if (node.type == GraphNodeType::MEMFREE) {
          op.dst = node.dst;  // pointer to free
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

} // namespace core
} // namespace vgre
