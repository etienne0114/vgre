// CUDA Driver API — occupancy queries

#include "cuda_driver_internal.h"

extern "C" {

// VGRE emulates a single virtual GPU.  Occupancy is a heuristic based on
// the emulated device properties and the requested block configuration.

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

  int sms = props.multiProcessorCount;
  if (sms <= 0) sms = 1;

  // Heuristic: each "SM" can hold ~2048 threads (Ampere-like).
  constexpr int kThreadsPerSM = 2048;
  int blocksPerSM = kThreadsPerSM / blockSize;
  if (blocksPerSM < 1) blocksPerSM = 1;

  // Shared memory limitation heuristic: 48 KB per block default.
  size_t totalSMem = props.sharedMemPerBlock;
  if (dynamicSMemSize > 0 && dynamicSMemSize > totalSMem) {
    *numBlocks = 0;
    return CUDA_SUCCESS;
  }
  if (dynamicSMemSize > 0) {
    int smemLimited = static_cast<int>(totalSMem / dynamicSMemSize);
    if (smemLimited < blocksPerSM) blocksPerSM = smemLimited;
  }

  *numBlocks = blocksPerSM;
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

  // Prefer a block size of 256 threads (good balance for CPU emulation).
  int preferred = 256;
  if (blockSizeLimit > 0 && preferred > blockSizeLimit)
    preferred = blockSizeLimit;
  if (preferred > props.maxThreadsPerBlock)
    preferred = props.maxThreadsPerBlock;

  *blockSize = preferred;

  if (minGridSize) {
    int sms = props.multiProcessorCount;
    if (sms <= 0) sms = 1;
    // minGridSize = number of blocks to saturate all SMs.
    int blocksPerSM = 2048 / preferred;
    if (blocksPerSM < 1) blocksPerSM = 1;
    *minGridSize = blocksPerSM * sms;
  }

  (void)dynamicSMemSize;
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
