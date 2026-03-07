#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <thread>

namespace vgre {
namespace core {

#if defined(_WIN32)
static PVOID g_vehHandler = nullptr;
#else
static struct sigaction old_sa;
#endif
static MemoryManager *g_memoryManagerId = nullptr;

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  g_memoryManagerId = this;
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  setupSignalHandler();

  std::thread([this]() { this->calibrateBandwidth(); }).detach();
}

MemoryManager::~MemoryManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[handle, alloc] : allocations_) {
    if (alloc.ptr) {
      if (alloc.isManaged) {
        munmap(alloc.ptr, alloc.size);
      } else {
        alignedFree(alloc.ptr);
      }
      alloc.ptr = nullptr;
    }
  }
  allocations_.clear();
  usedMemory_ = 0;
  g_memoryManagerId = nullptr;
  VGRE_LOG_DEBUG("MemoryManager", "Destroyed — all allocations freed");
}

void MemoryManager::setupSignalHandler() {
#if defined(_WIN32)
  g_vehHandler = AddVectoredExceptionHandler(1, MemoryManager::vectoredHandler);
  if (!g_vehHandler) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup VEH handler for UVM");
  }
#else
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = MemoryManager::segfaultHandler;
  if (sigaction(SIGSEGV, &sa, &old_sa) == -1) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup SIGSEGV handler for UVM");
  }
#endif
}

#if defined(_WIN32)
LONG
    CALLBACK MemoryManager::vectoredHandler(PEXCEPTION_POINTERS exceptionInfo) {
  if (exceptionInfo->ExceptionRecord->ExceptionCode ==
      EXCEPTION_ACCESS_VIOLATION) {
    void *addr = reinterpret_cast<void *>(
        exceptionInfo->ExceptionRecord->ExceptionInformation[1]);

    if (!g_memoryManagerId)
      return EXCEPTION_CONTINUE_SEARCH;

    {
      std::lock_guard<std::mutex> lock(g_memoryManagerId->mutex_);
      for (auto &[handle, alloc] : g_memoryManagerId->allocations_) {
        uintptr_t base = reinterpret_cast<uintptr_t>(alloc.ptr);
        uintptr_t target = reinterpret_cast<uintptr_t>(addr);
        if (alloc.isManaged && target >= base && target < base + alloc.size) {
          DWORD oldProtect;
          if (VirtualProtect(alloc.ptr, alloc.size, PAGE_READWRITE,
                             &oldProtect)) {
            const_cast<Allocation &>(alloc).isResidentOnHost = true;
            g_memoryManagerId->faultCount_++;
            return EXCEPTION_CONTINUE_EXECUTION;
          }
          break;
        }
      }
    }
  }
  return EXCEPTION_CONTINUE_SEARCH;
}
#else
void MemoryManager::segfaultHandler(int sig, siginfo_t *si, void *unused) {
  void *addr = si->si_addr;
  if (!g_memoryManagerId)
    goto fallback;

  {
    std::lock_guard<std::mutex> lock(g_memoryManagerId->mutex_);

    for (auto &[handle, alloc] : g_memoryManagerId->allocations_) {
      uintptr_t base = reinterpret_cast<uintptr_t>(alloc.ptr);
      uintptr_t target = reinterpret_cast<uintptr_t>(addr);
      if (alloc.isManaged && target >= base && target < base + alloc.size) {
        // Restore permissions
        if (mprotect(alloc.ptr, alloc.size, PROT_READ | PROT_WRITE) == -1) {
          break;
        }
        const_cast<Allocation &>(alloc).isResidentOnHost = true;
        g_memoryManagerId->faultCount_++;
        return;
      }
    }
  }

fallback:
  if (old_sa.sa_flags & SA_SIGINFO) {
    if (old_sa.sa_sigaction &&
        reinterpret_cast<void *>(old_sa.sa_sigaction) !=
            reinterpret_cast<void *>(SIG_DFL) &&
        reinterpret_cast<void *>(old_sa.sa_sigaction) !=
            reinterpret_cast<void *>(SIG_IGN)) {
      old_sa.sa_sigaction(sig, si, unused);
      return;
    }
  } else {
    if (old_sa.sa_handler && old_sa.sa_handler != SIG_DFL &&
        old_sa.sa_handler != SIG_IGN) {
      old_sa.sa_handler(sig);
      return;
    }
  }

  // Default action: restore default handler and re-raise
  struct sigaction dfl;
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  dfl.sa_flags = 0;
  sigaction(sig, &dfl, nullptr);
  raise(sig);
}
#endif

void *MemoryManager::alignedAlloc(size_t size, size_t alignment) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void MemoryManager::alignedFree(void *ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

VGREResult MemoryManager::allocate(size_t size, MemoryHandle &outHandle,
                                   DeviceId deviceId) {
  if (size == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  constexpr size_t ALIGN = 64;
  size_t alignedSize = (size + ALIGN - 1) & ~(ALIGN - 1);

  if (usedMemory_ + alignedSize > poolSize_)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  void *ptr = alignedAlloc(alignedSize, ALIGN);
  if (!ptr)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  std::memset(ptr, 0, alignedSize);

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = ALIGN;
  alloc.inUse = true;
  alloc.isManaged = false;
  alloc.isResidentOnHost = true;
  alloc.deviceId = deviceId;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  usedMemory_ += alignedSize;
  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManaged(size_t size, MemoryHandle &outHandle,
                                          DeviceId deviceId,
                                          unsigned int flags) {
  if (size == 0)
    return VGREResult::ERROR_INVALID_VALUE;

  long pageSize = sysconf(_SC_PAGESIZE);
  size_t alignedSize = (size + pageSize - 1) & ~(pageSize - 1);

  if (usedMemory_ + alignedSize > poolSize_)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  // flags == 2 corresponds to cudaMemAttachHost. If the memory is explicitly
  // attached to the host, we can map it read/write immediately, bypassing the
  // first-touch page fault overhead since we know the host needs access first.
#if defined(_WIN32)
  DWORD protect = (flags == 2) ? PAGE_READWRITE : PAGE_NOACCESS;
  void *ptr =
      VirtualAlloc(NULL, alignedSize, MEM_COMMIT | MEM_RESERVE, protect);
  if (!ptr)
    return VGREResult::ERROR_OUT_OF_MEMORY;
#else
  int prot = (flags == 2) ? (PROT_READ | PROT_WRITE) : PROT_NONE;

#if defined(__APPLE__)
  void *ptr = mmap(NULL, alignedSize, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
  void *ptr = mmap(NULL, alignedSize, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
  if (ptr == MAP_FAILED)
    return VGREResult::ERROR_OUT_OF_MEMORY;
#endif

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = pageSize;
  alloc.inUse = true;
  alloc.isManaged = true;
  alloc.isResidentOnHost =
      (flags == 2); // If mapped R/W initially, it's resident
  alloc.deviceId = deviceId;
  alloc.attachmentFlags = flags;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  usedMemory_ += alignedSize;
  outHandle = ptr;

  VGRE_LOG_INFO("MemoryManager",
                "Allocated UVM Managed memory: " + std::to_string(alignedSize) +
                    " bytes at " +
                    std::to_string(reinterpret_cast<uintptr_t>(ptr)));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::free(MemoryHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  size_t freed = it->second.size;
  if (it->second.isManaged) {
#if defined(_WIN32)
    VirtualFree(it->second.ptr, 0, MEM_RELEASE);
#else
    munmap(it->second.ptr, it->second.size);
#endif
  } else {
    alignedFree(it->second.ptr);
  }
  allocations_.erase(it);
  usedMemory_ -= freed;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyHostToDevice(MemoryHandle dst, const void *src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(dst);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  if (it->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->second.ptr, it->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  // Feature 8: Memory Compression Integration
  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && compBuffer.size() < bytes) {
      // Simulate transfer of compressed payload
      std::memcpy(dst, compBuffer.data(), compBuffer.size());
      // Decompress in-place (simulated) since dst receives compressed data over
      // bus
      size_t actualSize = 0;
      compEngine.decompress(compBuffer.data(), compBuffer.size(), dst, bytes,
                            actualSize);
      VGRE_LOG_DEBUG("MemoryManager",
                     "H2D Transfer compressed " + std::to_string(bytes) +
                         " -> " + std::to_string(compBuffer.size()) + " bytes");
    } else {
      std::memcpy(dst, src, bytes);
    }
  } else {
    std::memcpy(dst, src, bytes);
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
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(src);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  if (it->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->second.ptr, it->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  // Feature 8: Memory Compression Integration
  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && compBuffer.size() < bytes) {
      // Decompress into host dst
      size_t actualSize = 0;
      compEngine.decompress(compBuffer.data(), compBuffer.size(), dst, bytes,
                            actualSize);
      VGRE_LOG_DEBUG("MemoryManager",
                     "D2H Transfer compressed " + std::to_string(bytes) +
                         " -> " + std::to_string(compBuffer.size()) + " bytes");
    } else {
      std::memcpy(dst, src, bytes);
    }
  } else {
    std::memcpy(dst, src, bytes);
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
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto itDst = allocations_.find(dst);
  auto itSrc = allocations_.find(src);
  if (itDst == allocations_.end() || itSrc == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  // Cross-device check: if different devices, check P2P access
  if (itDst->second.deviceId != itSrc->second.deviceId) {
    if (!peerAccessMap_[itDst->second.deviceId][itSrc->second.deviceId] &&
        !peerAccessMap_[itSrc->second.deviceId][itDst->second.deviceId]) {
      VGRE_LOG_ERROR("MemoryManager",
                     "P2P access denied between device " +
                         std::to_string(itSrc->second.deviceId) + " and " +
                         std::to_string(itDst->second.deviceId));
      return VGREResult::ERROR_INVALID_VALUE;
    }
  }

  if (itDst->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(itDst->second.ptr, itDst->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(itDst->second.ptr, itDst->second.size, PROT_READ | PROT_WRITE);
#endif
  }
  if (itSrc->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(itSrc->second.ptr, itSrc->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(itSrc->second.ptr, itSrc->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  // Feature 7: Multi-GPU P2P Transfer Latency Simulation
  if (itDst->second.deviceId != itSrc->second.deviceId) {
    // Model PCIe 3.0 x16 latency (~1us) and bandwidth (~12 GB/s)
    double pcieGBps = 12.0;
    double delayMs = 0.001 + (bytes / (pcieGBps * 1024 * 1024 * 1024)) * 1000.0;
    std::this_thread::sleep_for(
        std::chrono::duration<double, std::milli>(delayMs));
  }

  std::memcpy(dst, src, bytes);
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  // "Real" Bandwidth Modeling based on topology
  double bandwidthGBps = h2dBandwidth_;

  if (itDst->second.deviceId != itSrc->second.deviceId) {
    auto &srcProps = RuntimeEngine::instance()
                         .getDevice(itSrc->second.deviceId)
                         .getProperties();
    auto &dstProps = RuntimeEngine::instance()
                         .getDevice(itDst->second.deviceId)
                         .getProperties();

    // If on the same PCI bus, use our measured D2D baseline
    if (srcProps.pciBusId == dstProps.pciBusId) {
      bandwidthGBps = d2dBandwidth_;
    } else {
      bandwidthGBps = h2dBandwidth_ * 0.5; // Simulate inter-bus PCIe penalty
    }
  } else {
    bandwidthGBps = d2dBandwidth_ * 2.0; // Device-local copy (L2/HBM)
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
  std::lock_guard<std::mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = true;
  VGRE_LOG_INFO("MemoryManager", "Enabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::disablePeerAccess(DeviceId currentDevice,
                                            DeviceId peerDevice) {
  std::lock_guard<std::mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = false;
  VGRE_LOG_INFO("MemoryManager", "Disabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

bool MemoryManager::canAccessPeer(DeviceId currentDevice,
                                  DeviceId peerDevice) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = peerAccessMap_.find(currentDevice);
  if (it == peerAccessMap_.end())
    return false;
  auto it2 = it->second.find(peerDevice);
  if (it2 == it->second.end())
    return false;
  return it2->second;
}

DeviceId MemoryManager::getOwnerDevice(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return -1;
  return it->second.deviceId;
}

size_t MemoryManager::getTotalMemory() const { return poolSize_; }
size_t MemoryManager::getUsedMemory() const { return usedMemory_.load(); }
size_t MemoryManager::getFreeMemory() const {
  return poolSize_ - usedMemory_.load();
}

bool MemoryManager::isValidHandle(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.count(handle) > 0;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return nullptr;

  if (it->second.isManaged) {
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
    const_cast<Allocation &>(it->second).isResidentOnHost = true;
  }

  return it->second.ptr;
}

void MemoryManager::getPageResidency(uint8_t outMap[1024]) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::memset(outMap, 0, 1024);

  if (poolSize_ == 0)
    return;
  size_t bytesPerPage = poolSize_ / 1024;
  if (bytesPerPage == 0)
    bytesPerPage = 1;

  for (auto const &[handle, alloc] : allocations_) {
    if (!alloc.isResidentOnHost)
      continue;

    // This is a simplification since handles are absolute pointers.
    // We treat the "address space" as the range of pointers we've allocated.
    // For a more realistic map, we'd use an actual virtual address space.
    // For now, we'll map allocations to the grid based on their relative
    // position in the pool.

    // Actually, let's just use a simple hash-based or deterministic mapping for
    // the UI to ensure the grid looks "real" and changes as memory is used.
    size_t startCell =
        (reinterpret_cast<size_t>(alloc.ptr) % poolSize_) / bytesPerPage;
    size_t numCells = (alloc.size + bytesPerPage - 1) / bytesPerPage;

    for (size_t i = 0; i < numCells; ++i) {
      size_t cell = (startCell + i) % 1024;
      outMap[cell] = 1;
    }
  }

  // Update fault rate (exponential moving average)
  static auto lastCheck = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - lastCheck).count();
  if (dt > 0.1) {
    float currentRate = static_cast<float>(faultCount_.exchange(0) / dt);
    pageFaultRate_ = pageFaultRate_ * 0.7f + currentRate * 0.3f;
    lastCheck = now;
  }
}

int MemoryManager::getResidentPageCount() const {
  uint8_t map[1024];
  getPageResidency(map);
  int count = 0;
  for (int i = 0; i < 1024; ++i)
    if (map[i])
      count++;
  return count;
}

void MemoryManager::calibrateBandwidth() {
  const size_t testSize = 4 * 1024 * 1024; // 4MB calibration
  void *hostPtr = alignedAlloc(testSize, 64);
  void *devicePtr = alignedAlloc(testSize, 64);

  if (!hostPtr || !devicePtr) {
    if (hostPtr)
      alignedFree(hostPtr);
    if (devicePtr)
      alignedFree(devicePtr);
    return;
  }

  // H2D Calibration
  auto start = std::chrono::steady_clock::now();
  std::memcpy(devicePtr, hostPtr, testSize);
  auto end = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(end - start).count();
  h2dBandwidth_.store(
      (static_cast<double>(testSize) / (1024.0 * 1024.0 * 1024.0)) / sec);

  // D2D Calibration
  start = std::chrono::steady_clock::now();
  std::memcpy(devicePtr, hostPtr,
              testSize); // In virtual mode these are same speed
  end = std::chrono::steady_clock::now();
  sec = std::chrono::duration<double>(end - start).count();
  d2dBandwidth_.store(
      (static_cast<double>(testSize) / (1024.0 * 1024.0 * 1024.0)) / sec);
  d2hBandwidth_.store(h2dBandwidth_.load());

  alignedFree(hostPtr);
  alignedFree(devicePtr);

  VGRE_LOG_INFO("MemoryManager", "Bandwidth Calibrated — Host-to-Device: " +
                                     std::to_string(h2dBandwidth_.load()) +
                                     " GB/s");
}

} // namespace core
} // namespace vgre
