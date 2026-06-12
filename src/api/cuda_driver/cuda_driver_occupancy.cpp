// CUDA Driver API — occupancy queries
//
// Implements the exact four-constraint occupancy model from the CUDA Programming Guide §K.3:
//   maxActiveBlocksPerSM = min(
//       floor(maxThreadsPerSM  / threadsPerBlock),   // thread limit
//       floor(maxWarpsPerSM    / warpsPerBlock),      // warp limit
//       floor(sharedMemPerSM   / smemPerBlock),       // shared memory limit
//       maxBlocksPerSM                                // hardware block limit
//   )
//
// sharedMemPerSM = 167936 bytes (Ampere SM80: 164 KB L1/smem pool).
// smemPerBlock   = dynamicSMemSize + kSmemOverhead (1 KB alignment).

#include "cuda_driver_internal.h"
#include "vgre/core/occupancy.h"   // vgre::perf::compute_occupancy — the shared occupancy model
#include <algorithm>

// Ampere-class virtual device: 164 KB shared memory per SM (GA100/GA102/GA104).
static constexpr size_t kSharedMemPerSM = 167936;
// 1 KB per-block overhead for bookkeeping/alignment (matches cuOccupancy docs).
static constexpr size_t kSmemOverhead   = 1024;

// maxActiveBlocksPerSM via the shared vgre::perf occupancy model (the single
// source of the block/warp/register/shared-memory math, P3-21), plus the CUDA
// thread-count constraint and the per-block shared-memory overflow check that the
// driver API adds on top.
static int computeMaxActiveBlocks(const vgre::DeviceProperties& props,
                                  int threadsPerBlock, size_t dynamicSMem) {
    if (threadsPerBlock <= 0) return 0;

    const int warpSize     = (props.warpSize > 0)        ? props.warpSize        : 32;
    const int maxThreadsSM = (props.maxThreadsPerSM > 0) ? props.maxThreadsPerSM : 2048;

    // Per-block shared-memory request (+ the documented 1 KB overhead); a request
    // exceeding the per-block maximum yields zero occupancy.
    size_t smemPerBlock = 0;
    if (dynamicSMem > 0) {
        smemPerBlock = dynamicSMem + kSmemOverhead;
        if (smemPerBlock > props.sharedMemPerBlock) return 0;
    }

    vgre::perf::SMLimits sm;
    sm.warpSize       = warpSize;
    sm.maxWarpsPerSM  = (props.maxWarpsPerSM > 0)  ? props.maxWarpsPerSM  : 64;
    sm.maxBlocksPerSM = (props.maxBlocksPerSM > 0) ? props.maxBlocksPerSM : 32;
    sm.smemPerSM      = static_cast<int>(kSharedMemPerSM);
    vgre::perf::KernelUsage k;
    k.threadsPerBlock = threadsPerBlock;
    k.regsPerThread   = 0;                          // device-function reg usage not modeled here
    k.smemPerBlock    = static_cast<int>(smemPerBlock);

    int blocks = vgre::perf::compute_occupancy(sm, k).activeBlocksPerSM;
    blocks = std::min(blocks, maxThreadsSM / threadsPerBlock);   // CUDA thread-count limit
    return (blocks > 0) ? blocks : 0;
}

extern "C" {

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func,
    int blockSize, size_t dynamicSMemSize) {
  (void)func;
  if (!numBlocks || blockSize <= 0) return CUDA_ERROR_INVALID_VALUE;

  vgre::DeviceProperties props{};
  auto dev = vgre::core::RuntimeEngine::instance().getDeviceId();
  if (vgre::core::RuntimeEngine::instance().getDeviceProperties(dev, props)
      != vgre::VGREResult::SUCCESS) {
    return CUDA_ERROR_UNKNOWN;
  }

  *numBlocks = computeMaxActiveBlocks(props, blockSize, dynamicSMemSize);
  return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    void *blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
  (void)func;
  (void)blockSizeToDynamicSMemSize;
  if (!blockSize) return CUDA_ERROR_INVALID_VALUE;

  vgre::DeviceProperties props{};
  auto dev = vgre::core::RuntimeEngine::instance().getDeviceId();
  if (vgre::core::RuntimeEngine::instance().getDeviceProperties(dev, props)
      != vgre::VGREResult::SUCCESS) {
    return CUDA_ERROR_UNKNOWN;
  }

  const int warpSize = (props.warpSize > 0) ? props.warpSize : 32;
  int maxBlock = props.maxThreadsPerBlock;
  if (blockSizeLimit > 0 && blockSizeLimit < maxBlock) maxBlock = blockSizeLimit;

  // Enumerate all warp-aligned block sizes from warpSize to maxBlock.
  // Objective: maximise total active warps = activeBlocksPerSM * warpsPerBlock.
  // This matches the documented goal of cuOccupancyMaxPotentialBlockSize.
  int bestBlockSize   = warpSize;
  int bestActiveWarps = 0;

  for (int bs = warpSize; bs <= maxBlock; bs += warpSize) {
    int activeBlocks  = computeMaxActiveBlocks(props, bs, dynamicSMemSize);
    int warpsPerBlock = (bs + warpSize - 1) / warpSize;
    int activeWarps   = activeBlocks * warpsPerBlock;
    if (activeWarps > bestActiveWarps ||
        (activeWarps == bestActiveWarps && bs < bestBlockSize)) {
      bestActiveWarps = activeWarps;
      bestBlockSize   = bs;
    }
  }

  *blockSize = bestBlockSize;

  if (minGridSize) {
    int sms = props.multiProcessorCount;
    if (sms <= 0) sms = 1;
    int activePerSM = computeMaxActiveBlocks(props, bestBlockSize, dynamicSMemSize);
    if (activePerSM < 1) activePerSM = 1;
    *minGridSize = activePerSM * sms;
  }

  return CUDA_SUCCESS;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    void *blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  (void)flags;
  return cuOccupancyMaxPotentialBlockSize(
      minGridSize, blockSize, func,
      blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

} // extern "C"
