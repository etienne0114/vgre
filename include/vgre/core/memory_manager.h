#ifndef VGRE_CORE_MEMORY_MANAGER_H
#define VGRE_CORE_MEMORY_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/interval_tree.h"
#include "vgre/core/page_table.h"

#include <atomic>
#include <cstddef>
#include <list>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <vector>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace core {

// ── Allocation record ──────────────────────────────────────────────────────
struct Allocation {
  void *ptr = nullptr;
  size_t size = 0;
  size_t alignment = 64;
  bool inUse = false;
  bool isManaged = false;
  bool isResidentOnHost = true;
  DeviceId deviceId = 0;
  unsigned int attachmentFlags =
      0; // Support for cudaMemAttachGlobal / cudaMemAttachHost

  // NUMA Node Bitmap — bit i set when NUMA node i has accessed this allocation.
  // Plain uint64_t; updated only while holding the shard mutex (non-atomic is
  // safe here because all writers are serialised by the mutex_).
  uint64_t nodeAccessBitmap{0};
};

// ── Memory pool handle ─────────────────────────────────────────────────────
using PoolHandle = uint64_t;

// Slab chunk for backing memory
struct PoolSlab {
  void *ptr = nullptr;
  size_t size = 0;
  PoolSlab *next = nullptr;
};

// Intrusive free-block node for the Michael-Scott two-lock queue.
// The first word of every free slab block is reinterpreted as FreeBlock*.
// Invariant: next == nullptr iff this is the tail node (M&S 1996, I4).
struct FreeBlock {
  FreeBlock* next{nullptr};
};

struct MemoryPool {
  PoolHandle id = 0;
  size_t blockSize = 0;            // Minimum allocation granularity
  int    numaNode  = -1;           // Preferred NUMA node for slab allocation (-1 = any)

  // ── Michael-Scott Two-Lock Queue (M&S 1996) ──────────────────────────────
  // freeList is the HEAD pointer (dummy sentinel); freeListTail is the TAIL.
  // Invariants (M&S 1996):
  //   I1: head_mutex guards freeList; at most one thread holds it at a time.
  //   I2: tail_mutex guards freeListTail; at most one thread holds it at a time.
  //   I3: freeListDummy (sentinel) is never dequeued; head always points at it
  //       or past it — the real first node is freeList->next.
  //   I4: freeList and freeListTail form a valid singly-linked chain.
  //   I5: Enqueue: {lock tail; tail->next = new; tail = new; unlock tail}
  //   I6: Dequeue: {lock head; next = head->next; if !next → empty;
  //                 head = next; unlock head} — returns *next
  // Using unique_ptr<mutex> because std::mutex is not movable/copyable and
  // MemoryPool is stored by value in an unordered_map (which may rehash/move).
  FreeBlock*  freeList{nullptr};            // head pointer (dummy sentinel)
  FreeBlock*  freeListTail{nullptr};        // tail pointer for enqueue
  FreeBlock*  freeListDummy{nullptr};       // dummy sentinel node (never freed)
  std::unique_ptr<std::mutex> head_mutex;   // guards head (dequeue path, M&S I1)
  std::unique_ptr<std::mutex> tail_mutex;   // guards tail (enqueue path, M&S I2)

  PoolSlab *slabList = nullptr;    // List of backing slabs
  size_t totalAllocated = 0;
  size_t peakAllocated = 0;
  size_t allocCount = 0;
  size_t freeCount = 0;
  // Tracks oversized allocations (size > blockSize) that bypass the slab.
  // Maps raw pointer → allocation size so freeToPool can release them correctly.
  std::unordered_map<void*, size_t> oversizedAllocs;
  // Tracks active slab-backed allocations for pointer provenance validation.
  std::unordered_set<void*> liveSlabAllocs;
  // Tracks slab address ranges for quick membership checks.
  std::vector<std::pair<uint8_t*, uint8_t*>> slabRanges;
  // cudaMemPool attributes
  uint64_t releaseThreshold = ~uint64_t{0};
  bool reuseFollowEventDependencies = true;
  bool reuseAllowOpportunistic = true;
  bool reuseAllowInternalDependencies = true;
  // location device -> access flags (cudaMemAccessFlags*)
  std::unordered_map<int, unsigned int> accessFlagsByDevice;

  // Constructor: initialise two-lock mutexes and dummy sentinel (M&S I3).
  MemoryPool()
    : head_mutex(std::make_unique<std::mutex>()),
      tail_mutex(std::make_unique<std::mutex>()) {}

  // Move-only (mutex unique_ptrs are movable; raw data is trivially moved).
  MemoryPool(MemoryPool&&) = default;
  MemoryPool& operator=(MemoryPool&&) = default;
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;
};

// ── Dynamic UVM region tracking for signal-safe lookup ─────────────────────
struct ManagedRegion {
  void *ptr = nullptr;
  size_t size = 0;
  mutable std::atomic<bool> isResidentOnHost{false};

  // Delta-Sync: Dirty page tracking
  size_t pageCount = 0;
  uint8_t* dirtyPages = nullptr; // Shared array (1 = dirty, 0 = clean)
  size_t pageSize = 4096; // Initialized by MemoryManager::registerManagedRegion to the system page size

  // Authoritative UVM Usage Tracking
  mutable std::atomic<long long> lastAccessTime{0};
  mutable std::atomic<uint32_t> accessCount{0};
  mutable std::atomic<int>  preferredLocation{-1}; // -1 = None, 0 = Host, >0 = DeviceID
  mutable std::atomic<bool> isReadMostly{false};   // set by cudaMemAdvise SetReadMostly
  mutable std::atomic<int>  lastPrefetchDev{-1};   // device of last cudaMemPrefetchAsync
  mutable std::atomic<uint32_t> accessedByMask{0}; // bit i => cudaMemAdviseSetAccessedBy(i)
  mutable std::atomic<uint32_t> conflictCount{0};  // Host-side faults against device preference
  mutable std::atomic<bool> hostAccessible{false}; // true if mapped R/W (flags==2), skip re-protection in clearDirtyPages

  // Per-device NUMA access counters (updated from SIGSEGV handler via signal-safe atomics).
  static constexpr int kMaxDevices = 4;
  mutable std::atomic<uint32_t> deviceAccessCounts[kMaxDevices];

  // NUMA Node Bitmap — bit i set when NUMA node i has caused a page fault on
  // this region. Allows migration policy to make per-region placement decisions.
  mutable std::atomic<uint64_t> nodeAccessBitmap{0};

  // Per-device EMA access rates for adaptive NUMA migration policy.
  // rate[d] = exponential moving average of access rate on device d (accesses/s).
  // Updated by the migration background thread.
  mutable float emaAccessRate[kMaxDevices] = {0.f, 0.f, 0.f, 0.f};

  // Predictive prefetch stride state.
  // We track the last 4 faulted pages to detect linear strides.
  // prefetchStride = 0 means no stride detected.
  mutable uintptr_t lastFaultPages[4] = {0, 0, 0, 0};
  mutable uintptr_t prefetchStride    = 0;
  mutable int       prefetchStrideConf = 0;  // confidence counter (0-4)

  ManagedRegion() = default;
  // Helper: copy all scalar fields from other (both copy-ctor and copy-assign use this)
  void copyScalarFields(const ManagedRegion& other) {
    isResidentOnHost.store(other.isResidentOnHost.load(std::memory_order_relaxed));
    lastAccessTime.store(other.lastAccessTime.load(std::memory_order_relaxed));
    accessCount.store(other.accessCount.load(std::memory_order_relaxed));
    preferredLocation.store(other.preferredLocation.load(std::memory_order_relaxed));
    isReadMostly.store(other.isReadMostly.load(std::memory_order_relaxed));
    lastPrefetchDev.store(other.lastPrefetchDev.load(std::memory_order_relaxed));
    accessedByMask.store(other.accessedByMask.load(std::memory_order_relaxed));
    conflictCount.store(other.conflictCount.load(std::memory_order_relaxed));
    hostAccessible.store(other.hostAccessible.load(std::memory_order_relaxed));
    for (int i = 0; i < kMaxDevices; ++i)
      deviceAccessCounts[i].store(other.deviceAccessCounts[i].load(std::memory_order_relaxed));
    nodeAccessBitmap.store(other.nodeAccessBitmap.load(std::memory_order_relaxed));
    for (int i = 0; i < kMaxDevices; ++i) emaAccessRate[i] = other.emaAccessRate[i];
    for (int i = 0; i < 4; ++i) lastFaultPages[i] = other.lastFaultPages[i];
    prefetchStride    = other.prefetchStride;
    prefetchStrideConf = other.prefetchStrideConf;
  }

  ManagedRegion(const ManagedRegion &other)
    : ptr(other.ptr), size(other.size), pageCount(other.pageCount),
      dirtyPages(other.dirtyPages), pageSize(other.pageSize) {
    copyScalarFields(other);
  }
  ManagedRegion &operator=(const ManagedRegion &other) {
    if (this != &other) {
      ptr = other.ptr; size = other.size;
      pageCount = other.pageCount; dirtyPages = other.dirtyPages; pageSize = other.pageSize;
      copyScalarFields(other);
    }
    return *this;
  }
  
  // Note: Destructor does NOT delete dirtyPages because it's shared across tables.
  // MemoryManager::unregisterManagedRegion handles the actual deletion.
  ~ManagedRegion() = default;

  // Helper to mark page dirty in signal handler
  void markDirty(void* faultAddr) const {
      if (!dirtyPages || pageCount == 0 || !ptr) return;
      uintptr_t offset = reinterpret_cast<uintptr_t>(faultAddr) - reinterpret_cast<uintptr_t>(ptr);
      size_t pageIdx = offset / pageSize;
      if (pageIdx < pageCount) {
          dirtyPages[pageIdx] = 1;
      }
  }

  // For sorted array lookup
  bool operator<(const ManagedRegion &other) const { return ptr < other.ptr; }
};

/**
 * @brief Virtual GPU Memory Manager.
 *
 * Backs the virtual VRAM with host-aligned memory regions and implements
 * Unified Virtual Memory (UVM) semantics via signal-safe page faulting.
 */
class MemoryManager {
public:
  explicit MemoryManager(size_t poolSize = 4ULL * 1024 * 1024 * 1024);
  ~MemoryManager();

  /**
   * @brief Allocates standard device-local memory.
   */
  VGREResult allocate(size_t size, MemoryHandle &outHandle,
                      DeviceId deviceId = 0);

  /**
   * @brief Allocates UVM (Unified) memory that migrates between CPU/GPU on fault.
   */
  VGREResult allocateManaged(size_t size, MemoryHandle &outHandle,
                             DeviceId deviceId = 0, unsigned int flags = 0);
  
  /**
   * @brief Allocates UVM memory at a specific virtual address (for cluster identity mapping).
   */
  VGREResult allocateManagedAt(void* addr, size_t size, MemoryHandle &outHandle,
                               DeviceId deviceId = 0, unsigned int flags = 0);

  VGREResult free(MemoryHandle handle);

  // UVM Migration Hints
  VGREResult memAdvise(const void *ptr, size_t count, int advice, DeviceId deviceId);
  // UVM range attribute queries (for cudaMemRangeGetAttribute)
  int  getPreferredLocation(void *ptr) const;   // -1 = CPU, >=0 = deviceId
  bool isReadMostly(void *ptr) const;
  int  getLastPrefetchLocation(void *ptr) const; // -1 = unknown/cpu, >=0 deviceId
  uint32_t getAccessedByMask(void *ptr) const;   // bitmask from cudaMemAdviseSetAccessedBy
  VGREResult memPrefetchAsync(const void *ptr, size_t count, DeviceId dstDevice);

  // Transfers
  VGREResult copyHostToDevice(MemoryHandle dst, const void *src, size_t bytes);
  VGREResult copyDeviceToHost(void *dst, MemoryHandle src, size_t bytes);
  VGREResult copyDeviceToDevice(MemoryHandle dst, MemoryHandle src,
                                size_t bytes);

  // Queries
  size_t getTotalMemory() const;
  size_t getUsedMemory() const;
  size_t getFreeMemory() const;
  bool isValidHandle(MemoryHandle handle) const;
  size_t getAllocationSize(MemoryHandle handle) const;
  size_t getAllocationSizeFromPointer(void *ptr) const;

  // ── Pointer introspection (for cudaPointerGetAttributes) ─────────────────
  // Returns true if ptr (or an offset within a tracked allocation) is known.
  // On success fills: size, isManaged, deviceId, attachmentFlags.
  bool getPointerAttributes(void *ptr, size_t &outSize, bool &outIsManaged,
                            DeviceId &outDevice,
                            unsigned int &outAttachmentFlags) const;

  // UVM Residency for Dashboard
  void getPageResidency(uint8_t outMap[1024]) const;
  int getResidentPageCount() const;
  float getPageFaultRate() const { return pageFaultRate_.load(); }

  // Get raw pointer from handle (for kernel execution)
  void *getPointer(MemoryHandle handle) const;

  // P2P Management
  VGREResult enablePeerAccess(DeviceId currentDevice, DeviceId peerDevice);
  VGREResult disablePeerAccess(DeviceId currentDevice, DeviceId peerDevice);
  bool canAccessPeer(DeviceId currentDevice, DeviceId peerDevice) const;
  DeviceId getOwnerDevice(MemoryHandle handle) const;

  // Memory Pool APIs
  VGREResult createPool(PoolHandle &outHandle, size_t blockSize = 256);
  VGREResult destroyPool(PoolHandle handle);
  VGREResult allocateFromPool(PoolHandle poolHandle, size_t size, MemoryHandle &outHandle);
  VGREResult freeToPool(PoolHandle poolHandle, MemoryHandle handle);
  // Pool statistics for cudaMemPoolGetAttribute
  uint64_t getPoolUsedBytes(PoolHandle handle) const;
  uint64_t getPoolPeakBytes(PoolHandle handle) const;
  // Release unused slabs while keeping at least minBytes (for cudaMemPoolTrimTo)
  VGREResult trimPool(PoolHandle handle, size_t minBytes);
  VGREResult setPoolReleaseThreshold(PoolHandle handle, uint64_t threshold);
  VGREResult getPoolReleaseThreshold(PoolHandle handle, uint64_t &threshold) const;
  VGREResult setPoolReuseFlags(PoolHandle handle,
                               bool followEventDeps,
                               bool opportunistic,
                               bool internalDeps);
  VGREResult getPoolReuseFlags(PoolHandle handle,
                               bool &followEventDeps,
                               bool &opportunistic,
                               bool &internalDeps) const;
  VGREResult setPoolAccess(PoolHandle handle, int device, unsigned int flags);
  VGREResult getPoolAccess(PoolHandle handle, int device, unsigned int &flags) const;

  // Delta-Sync: Dirty Page Management
  VGREResult getDirtyPages(MemoryHandle handle, std::vector<std::pair<size_t, size_t>>& outDirtyRanges) const;
  VGREResult clearDirtyPages(MemoryHandle handle);

  const std::unordered_map<MemoryHandle, Allocation>& getAllocations() const { return allocations_; }
  const std::unordered_map<PoolHandle, MemoryPool>& getPools() const { return pools_; }

  // GPU Memory Bandwidth Model
  // Records bytes accessed and execution time for one kernel block dispatch.
  // Used to compute effective bandwidth and compare against GPU peak bandwidth.
  void recordMemoryBandwidth(size_t bytes, double execMs);

  struct MemoryBandwidthStats {
    double effective_bandwidth_gbps{0};  // Measured CPU-side bandwidth (bytes / time)
    double gpu_peak_bandwidth_gbps{0};   // Reference GPU peak (default: A100 HBM3 = 2000 GB/s)
    double gpu_speedup_factor{0};        // Estimated speedup for bandwidth-bound workloads
    double coalescing_efficiency{1.0};   // Access pattern efficiency vs ideal (1.0 = stride-1)
    bool is_bandwidth_bound{false};      // True when effective_bw < 10% of CPU peak
    uint64_t total_kernels_sampled{0};   // Number of kernel dispatches contributing to the stats

    // ── Roofline Model ────────────────────────────────────────────────────
    // Classifies each kernel as compute-bound or memory-bound using the
    // Roofline model: ridge_point = peak_compute / peak_bandwidth.
    // If arithmetic_intensity < ridge_point → memory-bound.
    // If arithmetic_intensity >= ridge_point → compute-bound.
    double peak_compute_gflops{0};          // Measured or estimated peak (GFLOPs/s)
    double ridge_point_flops_per_byte{0};   // peak_compute / peak_bandwidth
    double avg_arithmetic_intensity{0};     // EMA of FLOPs/byte across sampled kernels
    double predicted_performance_gflops{0}; // Roofline-predicted achievable GFLOPs/s
    bool   last_kernel_compute_bound{false};// true = compute-bound, false = memory-bound
  };

  // Returns accumulated bandwidth statistics since last reset (or process start).
  MemoryBandwidthStats getMemoryBandwidthStats() const;

  // Update the Workload Characterization Cache for a named kernel.
  // bytes = memory traffic, execMs = measured time, flops = arithmetic ops.
  // The cache stores per-kernel EMA profiles used by the Roofline model and
  // by the graph optimizer's cost-benefit fusion analysis.
  void updateKernelCharacterization(const std::string& name,
                                     size_t bytes, double execMs,
                                     double flops = 0.0);

  // Reset bandwidth counters (call between benchmark phases to isolate measurements).
  void resetMemoryBandwidthStats();

  // ── TLB telemetry (Fix 3) ──────────────────────────────────────────────
  struct TlbStats {
    uint64_t l1Hits;
    uint64_t l1Misses;
    uint64_t l2Hits;
    uint64_t l2Misses;
    double   l1HitRate;
    double   l2HitRate;
  };
  // Returns cumulative TLB hit/miss counts since process start.
  // Useful for production observability: log at regular intervals to confirm
  // ≥85% L1 hit rate under steady-state workloads.
  static TlbStats getTlbStats();

  // Singleton convenience
  static MemoryManager &instance();

private:
#if defined(_WIN32)
  static LONG CALLBACK vectoredHandler(PEXCEPTION_POINTERS exceptionInfo);
#else
  static void segfaultHandler(int sig, siginfo_t *si, void *unused);
#endif
  void setupSignalHandler();
  void teardownSignalHandler();
  void *alignedAlloc(size_t size, size_t alignment);
  void alignedFree(void *ptr);

  bool registerManagedRegion(void *ptr, size_t size, bool hostAccessible = false);
  void unregisterManagedRegion(void *ptr);

  void calibrateBandwidth();
  void startMigrationThread();
  void stopMigrationThread();
  void migrationLoop();

  // O(log n) allocation range lookup — finds allocation containing ptr.
  // Must be called with mutex_ held.
  std::unordered_map<MemoryHandle, Allocation>::iterator
  findAllocationForPtr(void* ptr, size_t& outOffset);

  size_t poolSize_;
  std::atomic<size_t> usedMemory_{0};
  std::unordered_map<MemoryHandle, Allocation> allocations_;
  // Sorted by base address — enables O(log n) range lookup for H2D/D2H/D2D
  // copies instead of the O(n) linear scan over allocations_.
  std::map<uint8_t*, size_t> allocRange_;
  // Replaced recursive_mutex with shared_mutex (Fix 1: global mutex bottleneck).
  // Rationale: profiling showed 10-50% CPU cycles wasted in lock contention for
  // read-heavy paths (isValidHandle, getPointer, getAllocationSize, H2D/D2H lookup).
  // No recursive acquisition exists in the current call graph — unregisterManagedRegion
  // requires the caller to hold the lock but does NOT re-acquire it.
  // Read paths use shared_lock (N concurrent readers); write paths use unique_lock.
  mutable std::shared_mutex mutex_;
  
  std::mutex memoryFreedMutex_;
  std::condition_variable memoryFreedCv_;
  std::mutex allocatorMutex_;
  std::condition_variable allocatorCv_;

  // Signal-safe lookup structure (RCU-protected Interval Tree)
  struct RegionTreeContainer {
    MemoryIntervalTree<ManagedRegion> tree;
    size_t count;
  };
  std::atomic<RegionTreeContainer*> activeTree_{nullptr};
  std::vector<RegionTreeContainer*> retiredTrees_; // Cleanup deferred until activeHandlers_ == 0

  // O(1) radix page table for signal-safe fault lookup.
  // Updated alongside the RCU interval tree; provides constant-time address
  // resolution in the SIGSEGV/VEH handler without tree traversal.
  RadixPageTable pageTable_;

  // RCU grace-period counter: incremented on signal-handler entry, decremented
  // on exit.  The destructor spin-waits until this reaches 0 before freeing
  // retired trees, preventing use-after-free when a SIGSEGV fires between the
  // tree swap and the subsequent delete of the old pointer.
  mutable std::atomic<int> activeHandlers_{0};
  std::list<ManagedRegion> masterRegions_; // Master list with STABLE ADDRESSES protected by mutex_

  // Delta-Sync: Dirty Page Management
  std::atomic<double> h2dBandwidth_{25.0};
  std::atomic<double> d2hBandwidth_{25.0};
  std::atomic<double> d2dBandwidth_{50.0};

  // GPU Memory Bandwidth Model — tracks kernel memory throughput vs GPU reference
  mutable std::mutex bwMutex_;
  double bwTotalBytes_{0};        // cumulative bytes accessed across all tracked dispatches
  double bwTotalMs_{0};           // cumulative execution time for those dispatches
  uint64_t bwKernelCount_{0};     // number of kernel dispatches sampled
  // GPU reference peak bandwidth in GB/s (A100 SXM5 HBM3 = 2000 GB/s).
  // Overridable at runtime via VGRE_GPU_PEAK_BANDWIDTH_GBPS environment variable.
  double gpuPeakBandwidthGbps_{2000.0};

  // UVM metrics
  mutable std::atomic<float> pageFaultRate_{0.0f};
  mutable std::atomic<uint64_t> faultCount_{0};

  // P2P tracking: currentDevice -> set<peerDevice>
  std::unordered_map<DeviceId, std::unordered_map<DeviceId, bool>>
      peerAccessMap_;

  // Memory pools
  std::unordered_map<PoolHandle, MemoryPool> pools_;
  std::atomic<PoolHandle> nextPoolId_{1};

  // Adaptive UVM page-migration background thread
  std::thread         migrationThread_;
  std::atomic<bool>   migrationStop_{false};
  std::mutex          migrationMutex_;
  std::condition_variable migrationCv_;

  // Pending-fault ring buffer (signal-safe enqueue in segfault/VEH handler)
  // Background drainer thread processes pending faults outside signal context.
  size_t pendingFaultCapacity_{4096};
  struct PendingFault { 
    uintptr_t addr; 
    std::atomic<bool> ready{false};
  };
  PendingFault* pendingRing_{nullptr};
  std::atomic<uint64_t> pendingHead_{0}; // next write index (monotonic)
  std::atomic<uint64_t> pendingTail_{0}; // next read index (monotonic)
  std::atomic<uint64_t> pendingDropped_{0};
  std::thread pendingDrainerThread_;
  std::atomic<bool> pendingDrainerStop_{false};

  // Signal-safe enqueue (called from segfaultHandler / vectoredHandler)
  inline void enqueuePendingFault(uintptr_t addr) {
    if (!pendingRing_) return;
    uint64_t head;
    do {
      head = pendingHead_.load(std::memory_order_relaxed);
      uint64_t tail = pendingTail_.load(std::memory_order_acquire);
      if (head - tail >= pendingFaultCapacity_) {
        pendingDropped_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    } while (!pendingHead_.compare_exchange_weak(head, head + 1, std::memory_order_relaxed));

    size_t slot = static_cast<size_t>(head % pendingFaultCapacity_);
    pendingRing_[slot].addr = addr;
    pendingRing_[slot].ready.store(true, std::memory_order_release);
  }

  // Start/stop drainer
  void startPendingDrainer();
  void stopPendingDrainer();
  void pendingDrainerLoop();
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_MEMORY_MANAGER_H
