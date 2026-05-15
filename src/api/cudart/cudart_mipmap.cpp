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

extern "C" {

// ── Mipmapped Array API ────────────────────────────────────────────────────
// cudaMipmappedArray_t is an opaque handle; we use TextureId cast to void*.

#include "vgre/core/texture_manager.h"

cudaError_t cudaMallocMipmappedArray(void **mipmappedArrayPtr,
                                     const cudaChannelFormatDesc *desc,
                                     size_t width, size_t height,
                                     unsigned int numLevels,
                                     unsigned int flags) {
  (void)flags;
  if (!mipmappedArrayPtr || !desc || width == 0)
    return cudaErrorInvalidValue;

  // Determine element size from channel descriptor (x+y+z+w bits → bytes)
  size_t bits = static_cast<size_t>(desc->x + desc->y + desc->z + desc->w);
  size_t elementSize = (bits == 0) ? 4 : (bits + 7) / 8;

  vgre::core::TextureDescriptor td;
  td.filterMode = vgre::core::TextureFilterMode::LINEAR;
  td.addressMode = vgre::core::TextureAddressMode::CLAMP;

  vgre::core::TextureId id = 0;
  auto r = vgre::core::TextureManager::instance().createMipmappedArray(
      id, width, height, elementSize, numLevels, td);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorMemoryAllocation;

  *mipmappedArrayPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(id));
  return cudaSuccess;
}

cudaError_t cudaFreeMipmappedArray(void *mipmappedArray) {
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  vgre::core::TextureManager::instance().destroyCudaArray(id);
  return cudaSuccess;
}

// Returns a pointer to a specific mip level (as a cudaArray_t = void*).
cudaError_t cudaGetMipmappedArrayLevel(void **levelArrayPtr,
                                       void *mipmappedArray,
                                       unsigned int level) {
  if (!levelArrayPtr || !mipmappedArray) return cudaErrorInvalidValue;
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  void *ptr = vgre::core::TextureManager::instance().getMipmapLevelData(id, level);
  if (!ptr) return cudaErrorInvalidValue;
  *levelArrayPtr = ptr;
  return cudaSuccess;
}

// Generate mip levels from the base (level 0) data using box filtering.
cudaError_t cudaGenerateMipmaps(void *mipmappedArray) {
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  auto r = vgre::core::TextureManager::instance().generateMipmaps(id);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

} // extern "C"
