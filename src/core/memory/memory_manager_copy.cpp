#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>

#include <cstring>
#include <chrono>
#include <immintrin.h>

namespace vgre {
namespace core {

/**
 * @brief Non-temporal streaming memcpy for large buffers.
 * Bypasses the cache for writes to prevent cache pollution on large data transfers.
 */
#if defined(__x86_64__)
__attribute__((target("avx")))
#endif
static void streamingMemcpy(void* dst, const void* src, size_t bytes) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);

    // Threshold for streaming stores (1MB). Must be 32-byte aligned for AVX.
    if (bytes >= 1024 * 1024 && 
        (reinterpret_cast<uintptr_t>(d) & 31) == 0 && 
        (reinterpret_cast<uintptr_t>(s) & 31) == 0) {
        
        size_t i = 0;
        for (; i + 32 <= bytes; i += 32) {
            __m256i val = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + i));
            _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), val);
        }
        if (i < bytes) {
            memcpy(d + i, s + i, bytes - i);
        }
        _mm_sfence(); // Memory fence to ensure non-temporal stores are visible
    } else {
        memcpy(dst, src, bytes);
    }
}

std::unordered_map<MemoryHandle, Allocation>::iterator
MemoryManager::findAllocationForPtr(void* ptr, size_t& outOffset) {
  if (!ptr) return allocations_.end();
  uint8_t* target = static_cast<uint8_t*>(ptr);
  // upper_bound gives first entry with base > target; step back to get candidate
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    uint8_t* base = rit->first;
    size_t   sz   = rit->second;
    if (target >= base && target < base + sz) {
      outOffset = static_cast<size_t>(target - base);
      return allocations_.find(static_cast<MemoryHandle>(static_cast<void*>(base)));
    }
  }
  return allocations_.end();
}

VGREResult MemoryManager::copyHostToDevice(MemoryHandle dst, const void *src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERR_INVALID_VALUE;

  // ── Phase 1: look up allocation metadata under lock (fast, O(log n)) ──────
  void*  dstPtr   = nullptr;
  bool   isManaged = false;
  void*  regionPtr = nullptr;
  size_t regionSz  = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offset = 0;
    auto it = findAllocationForPtr(dst, offset);
    if (it == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offset + bytes > it->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "H2D copy overflow: requested " + std::to_string(bytes) +
                         " bytes at offset " + std::to_string(offset) +
                         " but allocation is " + std::to_string(it->second.size) + " bytes");
      return VGREResult::ERR_INVALID_VALUE;
    }
    dstPtr    = static_cast<uint8_t*>(it->second.ptr) + offset;
    isManaged = it->second.isManaged;
    regionPtr = it->second.ptr;
    regionSz  = it->second.size;
    it->second.isResidentOnHost = true;
  }

  // ── Phase 2: mprotect + memcpy WITHOUT holding the mutex ─────────────────
  // Allocations do not move (their base address is stable), so it is safe to
  // dereference dstPtr without the lock. This allows other threads to call
  // malloc/free/H2D/D2H concurrently instead of serialising behind a memcpy.
  if (isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(regionPtr, regionSz, PAGE_READWRITE, &oldProtect);
#else
    mprotect(regionPtr, regionSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(),
                                      dstPtr, bytes, actualSize);
      if (dr != VGREResult::SUCCESS || actualSize != bytes)
        streamingMemcpy(dstPtr, src, bytes);
    } else {
      streamingMemcpy(dstPtr, src, bytes);
    }
  } else {
    streamingMemcpy(dstPtr, src, bytes);
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "h2d_copy", 1, 1, ms, bytes, 0);

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToHost(void *dst, MemoryHandle src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERR_INVALID_VALUE;

  void*  srcPtr    = nullptr;
  bool   isManaged = false;
  void*  regionPtr = nullptr;
  size_t regionSz  = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offset = 0;
    auto it = findAllocationForPtr(src, offset);
    if (it == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offset + bytes > it->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "D2H copy overflow: requested " + std::to_string(bytes) +
                         " bytes at offset " + std::to_string(offset) +
                         " but allocation is " + std::to_string(it->second.size) + " bytes");
      return VGREResult::ERR_INVALID_VALUE;
    }
    srcPtr    = static_cast<uint8_t*>(it->second.ptr) + offset;
    isManaged = it->second.isManaged;
    regionPtr = it->second.ptr;
    regionSz  = it->second.size;
    it->second.isResidentOnHost = true;
  }

  if (isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(regionPtr, regionSz, PAGE_READWRITE, &oldProtect);
#else
    mprotect(regionPtr, regionSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(srcPtr, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(),
                                      dst, bytes, actualSize);
      if (dr != VGREResult::SUCCESS || actualSize != bytes)
        streamingMemcpy(dst, srcPtr, bytes);
    } else {
      streamingMemcpy(dst, srcPtr, bytes);
    }
  } else {
    streamingMemcpy(dst, srcPtr, bytes);
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "d2h_copy", 1, 1, ms, bytes, 0);
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToDevice(MemoryHandle dst, MemoryHandle src,
                                             size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERR_INVALID_VALUE;

  void*    dstPtr     = nullptr;
  void*    srcPtr     = nullptr;
  bool     dstManaged = false, srcManaged = false;
  void*    dstRegion  = nullptr, *srcRegion = nullptr;
  size_t   dstRegSz   = 0, srcRegSz = 0;
  DeviceId dstDeviceId_ = 0, srcDeviceId_ = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offsetDst = 0, offsetSrc = 0;
    auto itDst = findAllocationForPtr(dst, offsetDst);
    auto itSrc = findAllocationForPtr(src, offsetSrc);
    if (itDst == allocations_.end() || itSrc == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offsetDst + bytes > itDst->second.size || offsetSrc + bytes > itSrc->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "D2D copy overflow: requested " + std::to_string(bytes) +
                         " bytes but dst=" + std::to_string(itDst->second.size) +
                         " src=" + std::to_string(itSrc->second.size) + " bytes");
      return VGREResult::ERR_INVALID_VALUE;
    }
    if (itDst->second.deviceId != itSrc->second.deviceId) {
      if (!peerAccessMap_[itDst->second.deviceId][itSrc->second.deviceId] &&
          !peerAccessMap_[itSrc->second.deviceId][itDst->second.deviceId]) {
        VGRE_LOG_ERROR("MemoryManager",
                       "P2P access denied between device " +
                           std::to_string(itSrc->second.deviceId) + " and " +
                           std::to_string(itDst->second.deviceId));
        return VGREResult::ERR_INVALID_VALUE;
      }
    }
    dstPtr    = static_cast<uint8_t*>(itDst->second.ptr) + offsetDst;
    srcPtr    = static_cast<uint8_t*>(itSrc->second.ptr) + offsetSrc;
    dstManaged = itDst->second.isManaged;
    srcManaged = itSrc->second.isManaged;
    dstRegion  = itDst->second.ptr; dstRegSz = itDst->second.size;
    srcRegion  = itSrc->second.ptr; srcRegSz = itSrc->second.size;
    dstDeviceId_ = itDst->second.deviceId;
    srcDeviceId_ = itSrc->second.deviceId;
  }

  if (dstManaged) {
#if defined(_WIN32)
    DWORD op; VirtualProtect(dstRegion, dstRegSz, PAGE_READWRITE, &op);
#else
    mprotect(dstRegion, dstRegSz, PROT_READ | PROT_WRITE);
#endif
  }
  if (srcManaged) {
#if defined(_WIN32)
    DWORD op; VirtualProtect(srcRegion, srcRegSz, PAGE_READWRITE, &op);
#else
    mprotect(srcRegion, srcRegSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  streamingMemcpy(dstPtr, srcPtr, bytes);
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  // "Real" Bandwidth Modeling based on topology
  double bandwidthGBps = h2dBandwidth_;
  double baseLatencyMs = 0.0;

  if (dstDeviceId_ != srcDeviceId_) {
    auto srcProps = RuntimeEngine::instance()
                        .getDevice(srcDeviceId_)
                        .getProperties();
    auto dstProps = RuntimeEngine::instance()
                        .getDevice(dstDeviceId_)
                        .getProperties();

    if (srcProps.pciDomainId != dstProps.pciDomainId) {
      bandwidthGBps = h2dBandwidth_ * 0.5; // Cross-node QPI/NUMA
      baseLatencyMs = 0.030;               // 30us
    } else if (srcProps.pciBusId != dstProps.pciBusId) {
      bandwidthGBps = h2dBandwidth_;       // PCIe Switch
      baseLatencyMs = 0.015;               // 15us
    } else if (srcProps.pciDeviceId != dstProps.pciDeviceId) {
      bandwidthGBps = d2dBandwidth_;       // Same Bus NVLink/PCIe
      baseLatencyMs = 0.005;               // 5us
    } else {
      bandwidthGBps = d2dBandwidth_ * 2.0; // Internal
      baseLatencyMs = 0.001;               // 1us
    }
  } else {
    bandwidthGBps = d2dBandwidth_ * 2.5;   // Intra-device L1/L2 speed
    baseLatencyMs = 0.0;                   // 0us
  }

  // Authoritative Phase 3 (Zero-Simulation): 
  // We no longer artificially throttle transfers with sleep_for.
  // The system reports authoritative host physical performance.
  double expectedTransferMs = ((double)bytes / 1e9) / bandwidthGBps * 1000.0;
  double expectedTotalMs = expectedTransferMs + baseLatencyMs;

  // Update elapsed time to reflect what the 'modeled' time would be for telemetry,
  // but execute at the raw host speed.
  if (expectedTotalMs > ms) {
    ms = expectedTotalMs; 
  }

  // Telemetry: Record P2P execution metrics
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "p2p_transfer", 1, 1, ms, bytes, 0);

  VGRE_LOG_DEBUG("MemoryManager", "P2P Copy: " + std::to_string(bytes) +
                                      " bytes | Modeled BW: " +
                                      std::to_string(bandwidthGBps) + " GB/s");

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::enablePeerAccess(DeviceId currentDevice,
                                           DeviceId peerDevice) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = true;
  VGRE_LOG_INFO("MemoryManager", "Enabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::disablePeerAccess(DeviceId currentDevice,
                                            DeviceId peerDevice) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = false;
  VGRE_LOG_INFO("MemoryManager", "Disabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

bool MemoryManager::canAccessPeer(DeviceId currentDevice,
                                  DeviceId peerDevice) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = peerAccessMap_.find(currentDevice);
  if (it == peerAccessMap_.end())
    return false;
  auto it2 = it->second.find(peerDevice);
  if (it2 == it->second.end())
    return false;
  return it2->second;
}

DeviceId MemoryManager::getOwnerDevice(MemoryHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return -1;
  return it->second.deviceId;
}


} // namespace core
} // namespace vgre
