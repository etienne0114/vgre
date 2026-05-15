/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/compiler/kernel_parser.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// To avoid name conflicts, we define exactly the symbols frameworks need.

using namespace vgre::api;

// Local dim3 definition (matches CUDA runtime struct)
struct dim3 {
  unsigned int x, y, z;
};

// Cross-file helpers defined in cudart_shim.cpp
std::string vgre_lookup_kernel_name(const void *hostFn);
std::string vgre_lookup_kernel_source(const void *hostFn);

extern "C" {

// ── Cooperative Launch ────────────────────────────────────────────────────

cudaError_t cudaLaunchCooperativeKernel(const void *hostFun, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
  if (!hostFun || gridDim.x == 0 || blockDim.x == 0)
    return cudaErrorInvalidValue;

  std::string kernelName =
      vgre_lookup_kernel_name(hostFun);
  std::string kernelSource =
      vgre_lookup_kernel_source(hostFun);
  if (kernelName.empty() || kernelSource.empty())
    return cudaErrorInvalidDeviceFunction;

  vgre::dim3 vgreGrid(gridDim.x, gridDim.y, gridDim.z);
  vgre::dim3 vgreBlock(blockDim.x, blockDim.y, blockDim.z);

  // Route through the cooperative execution path so that
  // this_grid().sync() / vgre_jit_syncgrid() works correctly.
  return vgre::api::CUDAInterceptor::instance().launchCooperativeKernel(
      kernelName, kernelSource, vgreGrid, vgreBlock, args, sharedMem, stream);
}

struct cudaLaunchParams {
  const void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};

cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    cudaLaunchParams *launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  // flags interpretation (matches CUDA docs):
  //   cudaCooperativeLaunchMultiDeviceNoPreSync  (0x01) — we always start fresh,
  //   cudaCooperativeLaunchMultiDeviceNoPostSync (0x02) — caller syncs manually.
  // Both are no-ops in VGRE's CPU model; the ready-gate in Phase 2 of
  // launchCooperativeKernelMultiDevice already ensures simultaneous start.
  (void)flags;

  if (!launchParamsList || numDevices == 0) return cudaErrorInvalidValue;

  // Phase 1 — resolve kernel names from the module registry before handing
  // off to RuntimeEngine (registry lookup is not thread-safe, so done here
  // on the calling thread before device threads are spawned).
  std::vector<vgre::core::RuntimeEngine::CoopMultiLaunchParams> params;
  params.reserve(numDevices);

  for (unsigned i = 0; i < numDevices; ++i) {
    const auto &p = launchParamsList[i];
    if (!p.func || p.gridDim.x == 0 || p.blockDim.x == 0)
      return cudaErrorInvalidValue;

    std::string name = vgre_lookup_kernel_name(p.func);
    std::string src  = vgre_lookup_kernel_source(p.func);
    if (name.empty() || src.empty()) return cudaErrorInvalidDeviceFunction;

    params.push_back({
        name, src,
        vgre::dim3(p.gridDim.x, p.gridDim.y, p.gridDim.z),
        vgre::dim3(p.blockDim.x, p.blockDim.y, p.blockDim.z),
        p.args,
        p.sharedMem,
        // cudaStream_t is opaque; cast to uint64_t for VGRE's StreamId.
        static_cast<vgre::StreamId>(static_cast<uintptr_t>(p.stream))
    });
  }

  auto r = vgre::core::RuntimeEngine::instance()
               .launchCooperativeKernelMultiDevice(params);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize) {
  if (!numBlocks || blockSize <= 0) return cudaErrorInvalidValue;

  // ── Query current device SM limits (architecture-aware) ─────────────────────
  vgre::DeviceProperties dp{};
  vgre::core::RuntimeEngine::instance().getDeviceProperties(
      vgre::core::RuntimeEngine::instance().getDeviceId(), dp);

  const int kMaxWarpsPerSM     = dp.maxWarpsPerSM;
  const int kMaxBlocksPerSM    = dp.maxBlocksPerSM;
  const int kMaxThreadsPerSM   = dp.maxThreadsPerSM;
  const int kMaxRegsPerSM      = dp.maxRegsPerSM;
  const int kMaxSharedMemPerSM = dp.maxSharedMemPerSM;

  int warpsPerBlock = (blockSize + dp.warpSize - 1) / dp.warpSize;

  // Limit 1: warp capacity
  int limitWarps = kMaxWarpsPerSM / std::max(1, warpsPerBlock);

  // Limit 2: thread capacity
  int limitThreads = kMaxThreadsPerSM / blockSize;

  // ── Look up kernel metadata via RuntimeEngine ──────────────────────────────
  int registersPerThread = 32; // conservative default
  size_t staticSMem      = 0;

  if (func) {
    // Path: host stub pointer → device name → KernelId → KernelIR
    std::string kName = vgre_lookup_kernel_name(func);
    if (!kName.empty()) {
      vgre::KernelId kid =
          vgre::core::RuntimeEngine::instance().lookupKernelIdByName(kName.c_str());
      if (kid != 0) {
        const vgre::KernelIR* ir =
            vgre::core::RuntimeEngine::instance().getKernelIR(kid);
        if (ir) {
          staticSMem = ir->sharedMemSize;
          if (ir->registersPerThread > 0 && ir->registersPerThread != 32) {
            registersPerThread = ir->registersPerThread;
          } else {
            // Try to parse registers from PTX on demand
            std::string ptx = vgre_lookup_kernel_source(func);
            int parsedRegs = vgre::compiler::parsePTXRegisterCount(ptx, kName);
            if (parsedRegs > 0) {
              registersPerThread = parsedRegs;
              // Cache the parsed value back into KernelIR for future calls
              const_cast<vgre::KernelIR*>(ir)->registersPerThread = parsedRegs;
            }
          }
        }
      }
    }
  }

  // Limit 3: register pressure
  int regsPerBlock = warpsPerBlock * dp.warpSize * registersPerThread;
  int limitRegs = (regsPerBlock > 0) ? (kMaxRegsPerSM / regsPerBlock) : kMaxBlocksPerSM;

  // Limit 4: shared memory
  size_t totalSMem = staticSMem + dynamicSMemSize;
  int limitSMem = (totalSMem > 0)
      ? static_cast<int>(kMaxSharedMemPerSM / totalSMem)
      : kMaxBlocksPerSM;

  // Limit 5: hard SM block cap
  int active = std::min({limitWarps, limitThreads, limitRegs, limitSMem, kMaxBlocksPerSM});
  *numBlocks = std::max(1, active);

  VGRE_LOG_DEBUG("Occupancy",
      "blockSize=" + std::to_string(blockSize) +
      " warps=" + std::to_string(warpsPerBlock) +
      " regs/thread=" + std::to_string(registersPerThread) +
      " smem=" + std::to_string(totalSMem) +
      " → activeBlocks=" + std::to_string(*numBlocks) +
      " (limitW=" + std::to_string(limitWarps) +
      " limitT=" + std::to_string(limitThreads) +
      " limitR=" + std::to_string(limitRegs) +
      " limitS=" + std::to_string(limitSMem) +
      " arch=" + std::to_string(dp.major) + "." + std::to_string(dp.minor) + ")");
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize, unsigned int /*flags*/) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, func, blockSize, dynamicSMemSize);
}

} // extern "C"
