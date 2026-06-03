#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cstring>
#include <thread>
#include <mutex>
#include <algorithm>
#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/stat.h>   // stat() for NUMA node detection in calibrateBandwidth
#include <cstdio>       // snprintf
// mbind(2) policy constants — defined here to avoid a libnuma dependency.
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef MPOL_MF_MOVE
#  define MPOL_MF_MOVE   0x2
#endif
#ifndef SYS_mbind
#  define SYS_mbind 237  // x86_64
#endif
#endif  // __linux__

// ── Thread-local TLB Cache (Track 7.1) ───────────────────────────────────────
// Optimized 8-way set-associative TLB with 256 sets for caching virtual→ManagedRegion*
// translations inside signal handlers. Uses CLOCK replacement policy.
// These helpers are defined at file scope, before the signal handlers.

namespace vgre {
namespace core {

// Forward-declare ManagedRegion so TLB helpers compile (full def is in the header)
struct ManagedRegion;

} // namespace core
} // namespace vgre

// TLB enabled by default. Use `static const` (not constexpr) because
// vgre_get_config() is a runtime function — constexpr would be a compile error.
static const bool kTlbEnabled = []() -> bool {
    const char* e = vgre_get_config("VGRE_DISABLE_TLB");
    return !(e && std::atoi(e) == 1);
}();

// ── TLB hit-rate telemetry (Fix 3) ────────────────────────────────────────
// Per-process atomics for L1 and L2 hit/miss counts. These are intentionally
// global (not per-thread) so the overall hit rate is observable across threads.
// Query via vgre_get_tlb_stats() or VGRE_LOG at shutdown.
namespace {
struct VgreTlbStats {
    std::atomic<uint64_t> l1Hits{0};
    std::atomic<uint64_t> l1Misses{0};
    std::atomic<uint64_t> l2Hits{0};
    std::atomic<uint64_t> l2Misses{0};

    double l1HitRate() const {
        uint64_t h = l1Hits.load(std::memory_order_relaxed);
        uint64_t m = l1Misses.load(std::memory_order_relaxed);
        return (h + m) == 0 ? 0.0 : static_cast<double>(h) / static_cast<double>(h + m);
    }
    double l2HitRate() const {
        uint64_t h = l2Hits.load(std::memory_order_relaxed);
        uint64_t m = l2Misses.load(std::memory_order_relaxed);
        return (h + m) == 0 ? 0.0 : static_cast<double>(h) / static_cast<double>(h + m);
    }
};
} // namespace
static VgreTlbStats g_tlbStats;

// ── TLB Dimensions ────────────────────────────────────────────────────────────
static constexpr int kTlbSets = 256;
static constexpr int kTlbWays = 8;

// ── Structure-of-Arrays TLB set for vectorized lookup ────────────────────────
// Storing tags and regions as separate contiguous arrays allows AVX2 to load
// 4 tags in a single 256-bit register and compare all 4 simultaneously.
struct alignas(64) TlbSet {
    uintptr_t              tags[kTlbWays];    // page tags (0 = invalid)
    vgre::core::ManagedRegion* regions[kTlbWays]; // corresponding region pointers
    uint64_t               access[kTlbWays];  // LRU access counter per way
    uint8_t                clock;             // CLOCK replacement hand
};

// L1 TLB — thread-local, 256 sets × 8 ways
static thread_local TlbSet  t_tlb[kTlbSets];
static thread_local uint64_t t_tlb_ctr = 0;

// ── L2 Thread-Local TLB (larger, slower) ─────────────────────────────────────
static constexpr int kL2TlbSets = 1024;
static constexpr int kL2TlbWays = 16;

struct alignas(64) L2TlbSet {
    uintptr_t              tags[kL2TlbWays];
    vgre::core::ManagedRegion* regions[kL2TlbWays];
    uint64_t               access[kL2TlbWays];
};
static thread_local L2TlbSet  t_l2[kL2TlbSets];
static thread_local uint64_t  t_l2_ctr = 0;

// ── L2 Shared TLB (process-wide) ─────────────────────────────────────────────
// Sharded to allow concurrent access from all worker threads.
// 16 shards × 256 sets × 4 ways = 16K entries total.
// Uses atomic pointer to the region so no mutex is needed on the hot read path.
static constexpr int kSharedL2Shards = 16;
static constexpr int kSharedL2Sets   = 256;
static constexpr int kSharedL2Ways   = 4;

struct alignas(64) SharedTlbEntry {
    std::atomic<uintptr_t>              tag{0};
    std::atomic<vgre::core::ManagedRegion*> region{nullptr};
    std::atomic<uint64_t>               access{0};
};

struct alignas(4096) SharedTlbShard {
    SharedTlbEntry sets[kSharedL2Sets][kSharedL2Ways];
};

static SharedTlbShard g_sharedL2[kSharedL2Shards];
static std::atomic<uint64_t> g_sharedL2Ctr{0};

// ── NUMA Latency Model ────────────────────────────────────────────────────────
// Calibrated inter-NUMA bandwidth in GB/s. Index [src][dst].
// Initialised lazily on first access; updated by calibrateBandwidth().
static constexpr int kMaxNumaNodes = 8;
static std::atomic<double> g_numaLatencyNs[kMaxNumaNodes][kMaxNumaNodes]{};
static std::atomic<double> g_numaBandwidthGBps[kMaxNumaNodes][kMaxNumaNodes]{};

// Expose inter-node bandwidth for compression break-even calculation
static double getInterNodeBandwidthGBps(int srcNode, int dstNode) {
    if (srcNode < 0 || dstNode < 0 ||
        srcNode >= kMaxNumaNodes || dstNode >= kMaxNumaNodes)
        return 20.0; // conservative default
    double bw = g_numaBandwidthGBps[srcNode][dstNode].load(std::memory_order_relaxed);
    return (bw > 0.0) ? bw : (srcNode == dstNode ? 40.0 : 20.0);
}

// ── Forward declarations ──────────────────────────────────────────────────────
inline void tlbFill(uintptr_t addr, vgre::core::ManagedRegion* r);
inline void tlbEvict(uintptr_t addr, size_t size);

// ── Vectorized TLB Lookup ────────────────────────────────────────────────────
// For kTlbWays == 8, AVX2 can compare 4 tags per 256-bit register in two rounds.
inline vgre::core::ManagedRegion* tlbLookup(uintptr_t addr) {
    if (!kTlbEnabled) return nullptr;

    const uintptr_t page = addr >> 12;

    // ── L1 TLB (vectorized tag comparison) ───────────────────────────────────
    const int l1_set = static_cast<int>(page % static_cast<uintptr_t>(kTlbSets));
    TlbSet& S1 = t_tlb[l1_set];

#if defined(__AVX2__) && defined(ENABLE_VGRE_TLB)
    // Load 4 tags at a time using 256-bit vector (4 × 64-bit)
    const __m256i vtgt = _mm256_set1_epi64x(static_cast<int64_t>(page));
    // First 4 ways
    {
        __m256i vtags = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(S1.tags));
        __m256i vcmp  = _mm256_cmpeq_epi64(vtags, vtgt);
        int     mask  = _mm256_movemask_epi8(vcmp); // 8 bits → 4 bits active
        if (mask) {
            int way = __builtin_ctz(static_cast<unsigned>(mask)) >> 3; // 8 bytes per lane
            S1.access[way] = ++t_tlb_ctr;
            g_tlbStats.l1Hits.fetch_add(1, std::memory_order_relaxed);
            return S1.regions[way];
        }
    }
    // Ways 4-7
    if constexpr (kTlbWays > 4) {
        __m256i vtags = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(S1.tags + 4));
        __m256i vcmp  = _mm256_cmpeq_epi64(vtags, vtgt);
        int     mask  = _mm256_movemask_epi8(vcmp);
        if (mask) {
            int way = 4 + (__builtin_ctz(static_cast<unsigned>(mask)) >> 3);
            S1.access[way] = ++t_tlb_ctr;
            g_tlbStats.l1Hits.fetch_add(1, std::memory_order_relaxed);
            return S1.regions[way];
        }
    }
#else
    for (int w = 0; w < kTlbWays; ++w) {
        if (S1.tags[w] == page) {
            S1.access[w] = ++t_tlb_ctr;
            g_tlbStats.l1Hits.fetch_add(1, std::memory_order_relaxed);
            return S1.regions[w];
        }
    }
#endif
    g_tlbStats.l1Misses.fetch_add(1, std::memory_order_relaxed);

    // ── L2 Thread-Local TLB ──────────────────────────────────────────────────
    const int l2_set = static_cast<int>(page % static_cast<uintptr_t>(kL2TlbSets));
    L2TlbSet& S2 = t_l2[l2_set];
    for (int w = 0; w < kL2TlbWays; ++w) {
        if (S2.tags[w] == page) {
            S2.access[w] = ++t_l2_ctr;
            tlbFill(addr, S2.regions[w]);  // promote to L1
            g_tlbStats.l2Hits.fetch_add(1, std::memory_order_relaxed);
            return S2.regions[w];
        }
    }
    g_tlbStats.l2Misses.fetch_add(1, std::memory_order_relaxed);

    // ── L2 Shared TLB (process-wide) ─────────────────────────────────────────
    const int shard  = static_cast<int>(page % static_cast<uintptr_t>(kSharedL2Shards));
    const int s2_set = static_cast<int>((page / kSharedL2Shards) % kSharedL2Sets);
    for (int w = 0; w < kSharedL2Ways; ++w) {
        SharedTlbEntry& e = g_sharedL2[shard].sets[s2_set][w];
        uintptr_t etag = e.tag.load(std::memory_order_acquire);
        if (etag == page) {
            vgre::core::ManagedRegion* rptr =
                e.region.load(std::memory_order_acquire);
            if (rptr) {
                e.access.fetch_add(1, std::memory_order_relaxed);
                tlbFill(addr, rptr);  // promote to L1 + thread-local L2
                return rptr;
            }
        }
    }

    return nullptr;
}

inline void tlbFill(uintptr_t addr, vgre::core::ManagedRegion* r) {
    if (!kTlbEnabled) return;
    const uintptr_t page = addr >> 12;

    // ── Fill L1 TLB (CLOCK replacement) ──────────────────────────────────────
    const int l1_set = static_cast<int>(page % static_cast<uintptr_t>(kTlbSets));
    TlbSet& S1 = t_tlb[l1_set];
    // Find invalid slot or evict CLOCK victim
    for (int i = 0; i < kTlbWays; ++i) {
        int w = (S1.clock + i) % kTlbWays;
        if (S1.tags[w] == 0) {
            S1.tags[w] = page; S1.regions[w] = r; S1.access[w] = ++t_tlb_ctr;
            S1.clock = static_cast<uint8_t>((w + 1) % kTlbWays);
            goto fill_l2;
        }
    }
    // All ways occupied: evict at clock hand
    { int w = S1.clock;
      S1.tags[w] = page; S1.regions[w] = r; S1.access[w] = ++t_tlb_ctr;
      S1.clock = static_cast<uint8_t>((w + 1) % kTlbWays); }

fill_l2:
    // ── Fill thread-local L2 TLB (LRU eviction) ──────────────────────────────
    const int l2_set = static_cast<int>(page % static_cast<uintptr_t>(kL2TlbSets));
    L2TlbSet& S2 = t_l2[l2_set];
    int lru = 0; uint64_t lru_acc = S2.access[0];
    for (int w = 1; w < kL2TlbWays; ++w) {
        if (S2.access[w] < lru_acc) { lru_acc = S2.access[w]; lru = w; }
    }
    S2.tags[lru] = page; S2.regions[lru] = r; S2.access[lru] = ++t_l2_ctr;

    // ── Fill L2 Shared TLB ────────────────────────────────────────────────────
    const int shard  = static_cast<int>(page % static_cast<uintptr_t>(kSharedL2Shards));
    const int s2_set = static_cast<int>((page / kSharedL2Shards) % kSharedL2Sets);
    // LRU eviction using atomic access counters
    uint64_t glob = g_sharedL2Ctr.fetch_add(1, std::memory_order_relaxed);
    int evict = 0;
    uint64_t min_acc = g_sharedL2[shard].sets[s2_set][0].access.load(std::memory_order_relaxed);
    for (int w = 1; w < kSharedL2Ways; ++w) {
        uint64_t a = g_sharedL2[shard].sets[s2_set][w].access.load(std::memory_order_relaxed);
        if (a < min_acc) { min_acc = a; evict = w; }
    }
    SharedTlbEntry& se = g_sharedL2[shard].sets[s2_set][evict];
    se.region.store(r,    std::memory_order_release);
    se.tag.store(page,    std::memory_order_release);
    se.access.store(glob, std::memory_order_relaxed);
}

inline void tlbEvict(uintptr_t addr, size_t size) {
    if (!kTlbEnabled || size == 0) return;
    const uintptr_t first = addr >> 12;
    const uintptr_t last  = (addr + size - 1) >> 12;
    for (uintptr_t page = first; page <= last; ++page) {
        // L1 eviction
        int l1s = static_cast<int>(page % kTlbSets);
        for (int w = 0; w < kTlbWays; ++w)
            if (t_tlb[l1s].tags[w] == page) {
                t_tlb[l1s].tags[w] = 0; t_tlb[l1s].regions[w] = nullptr;
            }
        // Thread-local L2 eviction
        int l2s = static_cast<int>(page % kL2TlbSets);
        for (int w = 0; w < kL2TlbWays; ++w)
            if (t_l2[l2s].tags[w] == page) {
                t_l2[l2s].tags[w] = 0; t_l2[l2s].regions[w] = nullptr;
            }
        // Shared L2 eviction
        int shard = static_cast<int>(page % kSharedL2Shards);
        int s2s   = static_cast<int>((page / kSharedL2Shards) % kSharedL2Sets);
        for (int w = 0; w < kSharedL2Ways; ++w) {
            SharedTlbEntry& e = g_sharedL2[shard].sets[s2s][w];
            uintptr_t expected = page;
            e.tag.compare_exchange_strong(expected, 0,
                std::memory_order_release, std::memory_order_relaxed);
        }
    }
}

// ── Resume original namespaces ────────────────────────────────────────────────

namespace vgre {
namespace core {

#if defined(_WIN32)
static PVOID& getVehHandler() { static PVOID h = nullptr; return h; }
#else
static struct sigaction& getOldSigaction() { static struct sigaction sa{}; return sa; }
#endif
static std::atomic<bool>& getHandlerInstalled() { static std::atomic<bool> v{false}; return v; }
static std::atomic<int>& getInstanceCount() { static std::atomic<int> v{0}; return v; }
static std::atomic<MemoryManager *>& getMemoryManagerId() { static std::atomic<MemoryManager *> v{nullptr}; return v; }

// Thread-local active device ID — set by CPUParallelExecutor before dispatching
// kernel blocks, read by the SIGSEGV handler to attribute page faults to a device.
// Default is 0 (host / device 0).
thread_local int t_currentDevice = 0;

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  if (getInstanceCount().fetch_add(1) == 0) {
    setupSignalHandler();
  }
  getMemoryManagerId().store(this, std::memory_order_release);

  // Initialize empty active tree before calibration and migration thread so
  // the SIGSEGV handler never sees a null tree pointer.
  activeTree_.store(new RegionTreeContainer{MemoryIntervalTree<ManagedRegion>(), 0}, std::memory_order_release);

  // Allocate pending-fault ring buffer (size configurable via env)
  const char* env = vgre_get_config("VGRE_UVM_MAX_PENDING_FAULTS");
  if (env) {
    try {
      size_t v = std::stoul(env);
      if (v >= 64) pendingFaultCapacity_ = v;
    } catch (...) {
      // ignore parse errors, keep default
    }
  }
  pendingRing_ = new PendingFault[pendingFaultCapacity_];
  pendingHead_.store(0);
  pendingTail_.store(0);
  pendingDropped_.store(0);
  startPendingDrainer();

  // Bandwidth calibration: run synchronously so we do not carry a dangling
  // background thread across the full object lifetime.  The 64 MB × 50-iter
  // benchmark takes ~300 ms on a cold cache — acceptable during initialization.
  calibrateBandwidth();

  // Start background UVM page-migration thread.
  startMigrationThread();
}

MemoryManager::~MemoryManager() {
  // Stop background threads first
  stopMigrationThread();
  stopPendingDrainer();
  
  if (getInstanceCount().fetch_sub(1) == 1) {
    teardownSignalHandler();
  }
  
  if (getMemoryManagerId().load() == this) {
    getMemoryManagerId().store(nullptr, std::memory_order_release);
  }
  
  for (auto const &[handle, alloc] : allocations_) {
    if (alloc.ptr && alloc.isManaged) {
      // MemoryManager::unregisterManagedRegion would have deleted dirtyPages
      // But if we are force-cleaning in destructor:
#if defined(_WIN32)
      VirtualFree(alloc.ptr, 0, MEM_RELEASE);
#else
      munmap(alloc.ptr, alloc.size);
#endif
    }
  }

  // Cleanup remaining dirty bitsets in master list
  for (auto& [key, region] : masterRegions_) {
      delete[] region.dirtyPages;
      region.dirtyPages = nullptr;
  }
  allocations_.clear();
  usedMemory_ = 0;

  // Cleanup RCU trees — wait until all in-flight signal handlers have
  // finished reading the tree (activeHandlers_ == 0), then free everything.
  // Avoid tight busy-waiting in destructor: use exponential backoff with
  // a bounded timeout to prevent long shutdown stalls under pathological
  // fault loads.  Signal handlers cannot call non-async-signal-safe APIs
  // (e.g., condition_variable::notify), so this is a best-effort wait.
  {
    using namespace std::chrono;
    auto start = steady_clock::now();
    auto deadline = start + seconds(5); // total wait budget
    size_t spin = 0;
    while (activeHandlers_.load(std::memory_order_acquire) != 0) {
      if (steady_clock::now() >= deadline) {
        VGRE_LOG_ERROR("MemoryManager", "RCU grace period timed out after 5s; proceeding with cleanup");
        break;
      }
      // Replace sleep_for with just yield, as signal handlers finish in microseconds
      std::this_thread::yield();
      ++spin;
    }
  }

  RegionTreeContainer* active = activeTree_.load(std::memory_order_relaxed);
  if (active) {
    delete active;
  }
  for (auto* tree : retiredTrees_) {
    delete tree;
  }
  
  delete[] pendingRing_;
  pendingRing_ = nullptr;
  
  VGRE_LOG_DEBUG("MemoryManager", "Destroyed — all allocations freed");
}

void MemoryManager::setupSignalHandler() {
#if defined(_WIN32)
  getVehHandler() = AddVectoredExceptionHandler(1, MemoryManager::vectoredHandler);
  if (!getVehHandler()) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup VEH handler for UVM");
  } else {
    getHandlerInstalled().store(true, std::memory_order_release);
  }
#else
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
#if defined(__APPLE__)
  // macOS: Add SA_RESTART to automatically restart interrupted system calls
  sa.sa_flags |= SA_RESTART;
#endif
  sigemptyset(&sa.sa_mask);
  // Block SIGSEGV during handler to prevent re-entrancy
  sigaddset(&sa.sa_mask, SIGSEGV);
  sa.sa_sigaction = MemoryManager::segfaultHandler;
  if (sigaction(SIGSEGV, &sa, &getOldSigaction()) == -1) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup SIGSEGV handler for UVM");
  } else {
    getHandlerInstalled().store(true, std::memory_order_release);
  }
#endif
}

void MemoryManager::teardownSignalHandler() {
#if defined(_WIN32)
  if (getHandlerInstalled().load(std::memory_order_acquire) && getVehHandler()) {
    RemoveVectoredExceptionHandler(getVehHandler());
    getVehHandler() = nullptr;
    getHandlerInstalled().store(false, std::memory_order_release);
  }
#else
  if (getHandlerInstalled().load(std::memory_order_acquire)) {
    sigaction(SIGSEGV, &getOldSigaction(), nullptr);
    getHandlerInstalled().store(false, std::memory_order_release);
  }
#if defined(__APPLE__)
  // macOS: Additional cleanup for signal handling
  sigemptyset(&getOldSigaction().sa_mask);
#endif
#endif
}

#if defined(_WIN32)
LONG
    CALLBACK MemoryManager::vectoredHandler(PEXCEPTION_POINTERS exceptionInfo) {
  // Guard against null/corrupt exception context (stack overflow, heap
  // corruption) before touching any fields of the exception record.
  if (!exceptionInfo || !exceptionInfo->ExceptionRecord)
    return EXCEPTION_CONTINUE_SEARCH;
  if (exceptionInfo->ExceptionRecord->ExceptionCode ==
      EXCEPTION_ACCESS_VIOLATION) {
    // EXCEPTION_ACCESS_VIOLATION always has NumberParameters >= 2:
    // [0] = access type (0=read, 1=write, 8=DEP), [1] = faulting address.
    if (exceptionInfo->ExceptionRecord->NumberParameters < 2)
      return EXCEPTION_CONTINUE_SEARCH;
    void *addr = reinterpret_cast<void *>(
        exceptionInfo->ExceptionRecord->ExceptionInformation[1]);

    MemoryManager *mgr = getMemoryManagerId().load(std::memory_order_acquire);
    if (!mgr)
      return EXCEPTION_CONTINUE_SEARCH;

    {
      uintptr_t target = reinterpret_cast<uintptr_t>(addr);

      // Security: Validate address is within managed region bounds before handling
      // This prevents handling faults on arbitrary addresses (security breach)
      ManagedRegion* regionPtr = tlbLookup(target);
      if (!regionPtr) {
          regionPtr = mgr->pageTable_.lookup(target);
          if (regionPtr) {
              // Additional validation: check address is within region bounds
              uintptr_t regionStart = reinterpret_cast<uintptr_t>(regionPtr->ptr);
              uintptr_t regionEnd = regionStart + regionPtr->size;
              if (target < regionStart || target >= regionEnd) {
                  // Address outside valid region bounds - security violation
                  regionPtr = nullptr;
              } else {
                  tlbFill(target, regionPtr);
              }
          }
      }
      if (regionPtr) {
          ManagedRegion &region = *regionPtr;
          DWORD oldProtect;
          // Precise Delta-Sync: Protect only the faulting page
          size_t pageSize = region.pageSize;
          uintptr_t pageMask = ~(static_cast<uintptr_t>(pageSize) - 1);
          void* pageAddr = reinterpret_cast<void*>(target & pageMask);
          if (VirtualProtect(pageAddr, static_cast<SIZE_T>(pageSize), PAGE_READWRITE, &oldProtect)) {
            region.isResidentOnHost.store(true, std::memory_order_relaxed);
            
            if (exceptionInfo->ExceptionRecord->ExceptionInformation[0] == 1) {
                // Enqueue for background processing to avoid heavy work in VEH
                mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
                // Also mark dirty immediately (signal-safe write into preallocated buffer)
                region.markDirty(addr);
            }
            
            mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
            return EXCEPTION_CONTINUE_EXECUTION;
          }
        }
      }
    }
  return EXCEPTION_CONTINUE_SEARCH;
}
#else
void MemoryManager::segfaultHandler(int sig, siginfo_t *si, void *unused) {
  if (si->si_code != SEGV_ACCERR) goto fallback;

  void *addr;
  MemoryManager *mgr;

  addr = si->si_addr;
  mgr = getMemoryManagerId().load(std::memory_order_acquire);
  if (!mgr)
    goto fallback;

  {
    mgr->activeHandlers_.fetch_add(1, std::memory_order_acquire);

    uintptr_t target = reinterpret_cast<uintptr_t>(addr);

    // Security: Validate address is within managed region bounds before handling
    // This prevents handling faults on arbitrary addresses (security breach)
    ManagedRegion* regionPtr = tlbLookup(target);
    if (!regionPtr) {
        regionPtr = mgr->pageTable_.lookup(target);
        if (regionPtr) {
            // Additional validation: check address is within region bounds
            uintptr_t regionStart = reinterpret_cast<uintptr_t>(regionPtr->ptr);
            uintptr_t regionEnd = regionStart + regionPtr->size;
            if (target < regionStart || target >= regionEnd) {
                // Address outside valid region bounds - security violation
                regionPtr = nullptr;
            } else {
                tlbFill(target, regionPtr);
            }
        }
    }
    if (regionPtr) {
        ManagedRegion &region = *regionPtr;
        // Precise Delta-Sync: Protect only the faulting page
        size_t pageSize = region.pageSize;
        uintptr_t pageMask = ~(static_cast<uintptr_t>(pageSize) - 1);
        void* pageAddr = reinterpret_cast<void*>(target & pageMask);
        if (vgre::os::mprotect_rw(pageAddr, pageSize)) {
          region.isResidentOnHost.store(true, std::memory_order_relaxed);
          
          // Phase 11: Authoritative UVM LRU Tracking
          region.lastAccessTime.store(static_cast<long long>(vgre::os::get_monotonic_time_ns()), std::memory_order_relaxed);
          region.accessCount.fetch_add(1, std::memory_order_relaxed);

          // Attribute this fault to the currently active device for adaptive
          // migration heuristics. t_currentDevice is a thread-local set by the
          // kernel dispatcher before executing each block.
          {
            int dev = t_currentDevice;
            if (dev >= 0 && dev < ManagedRegion::kMaxDevices) {
              region.deviceAccessCounts[dev].fetch_add(1, std::memory_order_relaxed);
            }
          }

          // Phase 12: Authoritative Memory Sync
          int pref = region.preferredLocation.load(std::memory_order_relaxed);
          if (pref > 0) { // Preferred on a Device, but faulting on Host
              uint32_t c = region.conflictCount.fetch_add(1, std::memory_order_relaxed);
              if (c == 100) { 
                  // Trigger threshold reached - note: we cannot log here safely
              }
          }
          
          // Mark dirty if it's a write fault — enqueue and also mark immediately
#if defined(__x86_64__) && !defined(_WIN32)
          ucontext_t *uc = (ucontext_t *)unused;
          // Bit 1 of error code is Write/Read (1=Write)
          if (uc->uc_mcontext.gregs[REG_ERR] & 0x2) {
              mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
              // Immediate marking still performed — signal-safe write
              region.markDirty(addr);
          }
#else
          // Fallback: enqueue for background drainer
          mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
          region.markDirty(addr);
#endif
          
          mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);

          // ── NUMA Node Bitmap update ─────────────────────────────────────
          // Record which NUMA node caused this fault (signal-safe syscall).
#ifdef SYS_getcpu
          { unsigned cpu_id = 0, node_id = 0;
            if (::syscall(SYS_getcpu, &cpu_id, &node_id, nullptr) == 0 &&
                node_id < 64) {
                region.nodeAccessBitmap.fetch_or(
                    1ULL << node_id, std::memory_order_relaxed);
            }
          }
#endif

          // ── Predictive Page Prefetch ────────────────────────────────────
          // Track the last 4 faulted pages. If a linear stride is detected,
          // prefetch the next predicted page by touching it with mprotect.
          {
            uintptr_t faultPage = target >> 12;
            // Shift history
            region.lastFaultPages[0] = region.lastFaultPages[1];
            region.lastFaultPages[1] = region.lastFaultPages[2];
            region.lastFaultPages[2] = region.lastFaultPages[3];
            region.lastFaultPages[3] = faultPage;
            // Detect stride from last 3 pages
            if (region.lastFaultPages[1] && region.lastFaultPages[2] && region.lastFaultPages[3]) {
                uintptr_t d1 = region.lastFaultPages[2] - region.lastFaultPages[1];
                uintptr_t d2 = region.lastFaultPages[3] - region.lastFaultPages[2];
                if (d1 == d2 && d1 >= 1 && d1 <= 64) {
                    if (d1 == region.prefetchStride) {
                        region.prefetchStrideConf = std::min(region.prefetchStrideConf + 1, 4);
                    } else {
                        region.prefetchStride    = d1;
                        region.prefetchStrideConf = 1;
                    }
                    // Prefetch if confidence >= 2
                    if (region.prefetchStrideConf >= 2) {
                        uintptr_t nextPage = faultPage + region.prefetchStride;
                        uintptr_t regionEnd = (reinterpret_cast<uintptr_t>(region.ptr) + region.size) >> 12;
                        if (nextPage < regionEnd) {
                            void* nextAddr = reinterpret_cast<void*>(nextPage << 12);
                            // Pre-fault: make the next page writable before the kernel faults on it
                            vgre::os::mprotect_rw(nextAddr, region.pageSize);
                        }
                    }
                } else {
                    region.prefetchStrideConf = 0;
                }
            }
          }

          mgr->activeHandlers_.fetch_sub(1, std::memory_order_release);
          return;
        }
      }
    // Tree lookup missed (not a managed region) — release the RCU counter
    // before falling through to the previous signal handler.
    mgr->activeHandlers_.fetch_sub(1, std::memory_order_release);
  }

fallback:
  auto& old_sa = getOldSigaction();
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

ManagedRegion* MemoryManager::findRegionByPtr(void* ptr) {
  // O(log n): std::map binary search by exact base address
  auto it = masterRegions_.find(reinterpret_cast<uintptr_t>(ptr));
  return (it != masterRegions_.end()) ? &it->second : nullptr;
}

ManagedRegion* MemoryManager::findRegionContaining(uintptr_t addr) {
  // O(log n): upper_bound then step back to find region whose key <= addr,
  // then verify addr < key + size to confirm containment.
  auto it = masterRegions_.upper_bound(addr);
  if (it == masterRegions_.begin()) return nullptr;
  --it;
  uintptr_t base = it->first;
  if (addr < base + it->second.size) return &it->second;
  return nullptr;
}

bool MemoryManager::registerManagedRegion(void *ptr, size_t size, bool hostAccessible) {
  // Note: lock must be held by caller (allocateManaged / allocateManagedAt)
  
  ManagedRegion region;
  region.ptr = ptr;
  region.size = size;
  region.isResidentOnHost.store(hostAccessible);
  region.hostAccessible.store(hostAccessible);
  
  // Initialize dirty page tracking
  size_t pageSize = 4096; // Default fallback
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pageSize = static_cast<size_t>(si.dwPageSize);
#else
  long ps = sysconf(_SC_PAGESIZE);
  if (ps > 0) pageSize = static_cast<size_t>(ps);
#endif
  region.pageSize = pageSize;
  region.pageCount = (size + pageSize - 1) / pageSize;
  region.dirtyPages = new uint8_t[region.pageCount];
  memset(region.dirtyPages, 0, region.pageCount);
  
  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto [ins_it, ok] = masterRegions_.emplace(key, region);

  // Create new active tree (RCU swap)
  RegionTreeContainer* newContainer = new RegionTreeContainer();
  newContainer->count = masterRegions_.size();
  for (auto &[k, r] : masterRegions_) {
    newContainer->tree.insert(reinterpret_cast<uintptr_t>(r.ptr),
                             reinterpret_cast<uintptr_t>(r.ptr) + r.size, &r);
  }

  // Insert into radix page table for O(1) lookup — use stable map node address
  pageTable_.insert(reinterpret_cast<uintptr_t>(ins_it->second.ptr), ins_it->second.size, &ins_it->second);
  
  RegionTreeContainer* oldContainer = activeTree_.exchange(newContainer, std::memory_order_acq_rel);
  if (oldContainer) {
    retiredTrees_.push_back(oldContainer);
  }
  if (activeHandlers_.load(std::memory_order_acquire) == 0 && !retiredTrees_.empty()) {
    for (auto* tree : retiredTrees_) {
      delete tree;
    }
    retiredTrees_.clear();
  }
  
  return true;
}

void MemoryManager::unregisterManagedRegion(void *ptr) {
  // Note: lock must be held by caller (free)

  // O(log n): std::map binary search by exact base address
  uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
  auto it = masterRegions_.find(key);

  if (it == masterRegions_.end()) return;

  // Owner deletes the shared dirty pages buffer
  delete[] it->second.dirtyPages;
  it->second.dirtyPages = nullptr; // Prevent double delete just in case

  // Remove from radix page table
  pageTable_.remove(reinterpret_cast<uintptr_t>(it->second.ptr), it->second.size);

  // Evict TLB entries covering this region so stale translations aren't used
  tlbEvict(reinterpret_cast<uintptr_t>(it->second.ptr), it->second.size);

  masterRegions_.erase(it);

  // Create new active tree
  RegionTreeContainer* newContainer = new RegionTreeContainer();
  newContainer->count = masterRegions_.size();
  for (auto &[k, r] : masterRegions_) {
    newContainer->tree.insert(reinterpret_cast<uintptr_t>(r.ptr),
                             reinterpret_cast<uintptr_t>(r.ptr) + r.size, &r);
  }
  
  RegionTreeContainer* oldContainer = activeTree_.exchange(newContainer, std::memory_order_acq_rel);
  if (oldContainer) {
    retiredTrees_.push_back(oldContainer);
  }
  if (activeHandlers_.load(std::memory_order_acquire) == 0 && !retiredTrees_.empty()) {
    for (auto* tree : retiredTrees_) {
      delete tree;
    }
    retiredTrees_.clear();
  }
}

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

// ── Memory pressure check (Linux) ─────────────────────────────────────────
// Returns approximate free system memory in bytes. Returns SIZE_MAX on error
// (treat as unlimited — never blocks a valid allocation on a read failure).
static size_t systemFreeMemoryBytes() {
#if defined(__linux__)
  std::ifstream f("/proc/meminfo");
  if (!f.is_open()) return SIZE_MAX;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      size_t kb = 0;
      if (std::sscanf(line.c_str(), "MemAvailable: %zu kB", &kb) == 1)
        return kb * 1024;
    }
  }
#endif
  return SIZE_MAX;
}

VGREResult MemoryManager::allocate(size_t size, MemoryHandle &outHandle,
                                   DeviceId deviceId) {
  if (size == 0)
    return VGREResult::ERR_INVALID_VALUE;
  constexpr size_t ALIGN = 64;
  size_t alignedSize = (size + ALIGN - 1) & ~(ALIGN - 1);

  // Memory pressure backoff: for large allocations (> 1 MB) check system free
  // memory before attempting. If < 10% of MemAvailable remains, retry with
  // exponential backoff rather than immediately returning OOM.
  constexpr size_t kPressureThresholdBytes = 1 * 1024 * 1024; // 1 MB
  if (alignedSize >= kPressureThresholdBytes) {
    constexpr int kMaxRetries = 3;
    const int kDelaysMs[kMaxRetries] = {50, 100, 200};
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
      size_t freeBytes = systemFreeMemoryBytes();
      if (freeBytes == SIZE_MAX) break; // can't read — proceed normally
      // Threshold: if < 10% free (freeBytes < total * 0.10), back off.
      // We don't know total easily, so use an absolute minimum: 64 MB must remain.
      constexpr size_t kMinFreeBytes = 64 * 1024 * 1024;
      if (freeBytes >= kMinFreeBytes + alignedSize) break; // enough room
      if (attempt == 0)
        VGRE_LOG_WARN("MemoryManager",
          "Memory pressure: only " + std::to_string(freeBytes / (1024*1024)) +
          " MB free, requested " + std::to_string(alignedSize / (1024*1024)) +
          " MB — backing off " + std::to_string(kDelaysMs[attempt]) + " ms");
      if (attempt == kMaxRetries - 1) {
        VGRE_LOG_ERROR("MemoryManager",
          "Memory pressure: allocation of " +
          std::to_string(alignedSize / (1024*1024)) + " MB failed after retries");
        return VGREResult::ERR_OUT_OF_MEMORY;
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDelaysMs[attempt]));
    }
  }

  // Atomic CAS reservation: eliminates TOCTOU race between check and add
  size_t current = usedMemory_.load(std::memory_order_relaxed);
  do {
    if (current + alignedSize > poolSize_)
      return VGREResult::ERR_OUT_OF_MEMORY;
  } while (!usedMemory_.compare_exchange_weak(current, current + alignedSize,
                                              std::memory_order_acq_rel));

  void *ptr = alignedAlloc(alignedSize, ALIGN);
  if (!ptr) {
    // Undo the reservation if allocation failed
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERR_OUT_OF_MEMORY;
  }

  memset(ptr, 0, alignedSize);

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = ALIGN;
  alloc.inUse = true;
  alloc.isManaged = false;
  alloc.isResidentOnHost = true;
  alloc.deviceId = deviceId;

  {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;
  }

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::free(MemoryHandle handle) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return VGREResult::ERR_INVALID_VALUE;

  size_t freed = it->second.size;
  allocRange_.erase(static_cast<uint8_t*>(it->second.ptr));
  if (it->second.isManaged) {
    unregisterManagedRegion(it->second.ptr);
#if defined(_WIN32)
    VirtualFree(it->second.ptr, 0, MEM_RELEASE);
#else
    munmap(it->second.ptr, it->second.size);
#endif
  } else {
    alignedFree(it->second.ptr);
  }
  allocations_.erase(it);
  usedMemory_.fetch_sub(freed, std::memory_order_acq_rel);
  return VGREResult::SUCCESS;
}


size_t MemoryManager::getTotalMemory() const { return poolSize_; }
size_t MemoryManager::getUsedMemory() const { return usedMemory_.load(); }
size_t MemoryManager::getFreeMemory() const {
  return poolSize_ - usedMemory_.load();
}

bool MemoryManager::isValidHandle(MemoryHandle handle) const {
  if (!handle) return false;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    if (target >= rit->first && target < rit->first + rit->second) return true;
  }
  return false;
}

size_t MemoryManager::getAllocationSize(MemoryHandle handle) const {
  if (!handle) return 0;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    if (target >= rit->first && target < rit->first + rit->second) return rit->second;
  }
  return 0;
}

size_t MemoryManager::getAllocationSizeFromPointer(void *ptr) const {
  if (!ptr) return 0;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(ptr);
  // O(log n) lookup via the sorted allocRange_ map
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    uint8_t* base = rit->first;
    size_t   sz   = rit->second;
    if (target >= base && target < base + sz) {
      return sz;
    }
  }
  return 0;
}

bool MemoryManager::getPointerAttributes(void *ptr, size_t &outSize,
                                          bool &outIsManaged,
                                          DeviceId &outDevice,
                                          unsigned int &outAttachmentFlags) const {
  if (!ptr) return false;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint8_t *target = static_cast<uint8_t *>(ptr);

  // O(log n) lookup via sorted allocRange_ map
  auto rit = allocRange_.upper_bound(target);
  if (rit == allocRange_.begin()) return false;
  --rit;
  uint8_t *base = rit->first;
  size_t   sz   = rit->second;
  if (target < base || target >= base + sz) return false;

  auto it = allocations_.find(static_cast<MemoryHandle>(static_cast<void *>(base)));
  if (it == allocations_.end()) return false;

  outSize = it->second.size;
  outIsManaged = it->second.isManaged;
  outDevice = it->second.deviceId;
  outAttachmentFlags = it->second.attachmentFlags;
  return true;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  if (!handle) return nullptr;
  std::shared_lock<std::shared_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);

  // O(log n) lookup via sorted allocRange_ map
  auto rit = allocRange_.upper_bound(target);
  if (rit == allocRange_.begin()) return nullptr;
  --rit;
  uint8_t* base = rit->first;
  size_t   sz   = rit->second;
  if (target < base || target >= base + sz) return nullptr;

  size_t offset = static_cast<size_t>(target - base);
  auto it = allocations_.find(static_cast<MemoryHandle>(static_cast<void*>(base)));
  if (it == allocations_.end()) return nullptr;

  if (it->second.isManaged) {
    vgre::os::mprotect_rw(it->second.ptr, it->second.size);
    const_cast<Allocation &>(it->second).isResidentOnHost = true;
  }

  return static_cast<uint8_t*>(it->second.ptr) + offset;
}

void MemoryManager::getPageResidency(uint8_t outMap[1024]) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  memset(outMap, 0, 1024);

  if (poolSize_ == 0) return;

  uint64_t virtualOffset = 0;
  for (auto const &[handle, alloc] : allocations_) {
    if (!alloc.ptr || alloc.size == 0)
      continue;

    // Map this allocation to its virtual position in the pool based on its entry order
    // This provides a stable "spatial" layout in the UI that represents the pool occupancy.
    size_t startCell = static_cast<size_t>((virtualOffset * 1024ULL) / poolSize_);
    uint64_t endOffset = virtualOffset + alloc.size;
    size_t endCell = static_cast<size_t>(((endOffset * 1024ULL + poolSize_ - 1ULL) / poolSize_) - 1ULL);
    
    if (startCell > 1023) startCell = 1023;
    if (endCell > 1023) endCell = 1023;
    if (endCell < startCell) endCell = startCell;

    // Feature 12: Real-time UVM Residency sync from lock-free tracking
    // Feature 12: Real-time UVM Residency sync from lock-free tracking
    bool isActuallyResident = alloc.isResidentOnHost;
    if (!isActuallyResident) {
      // O(log n): exact key lookup by base address
      auto rit = masterRegions_.find(reinterpret_cast<uintptr_t>(alloc.ptr));
      if (rit != masterRegions_.end()) {
        if (rit->second.isResidentOnHost.load(std::memory_order_relaxed)) {
          isActuallyResident = true;
        }
      }
    }

    uint8_t status = isActuallyResident ? 1 : 2; // 1=Resident, 2=Allocated but swapped/not-resident

    for (size_t cell = startCell; cell <= endCell; ++cell) {
      // If multiple allocations fall into one cell, prefer showing Resident status
      if (outMap[cell] == 0 || status == 1) {
          outMap[cell] = status;
      }
    }
    
    virtualOffset += alloc.size;
    if (virtualOffset >= poolSize_) break;
  }

  // ── Holt-Winters double exponential smoothing for page-fault rate ──────────
  // Level L_t = α * x_t + (1-α) * (L_{t-1} + T_{t-1})
  // Trend  T_t = β * (L_t - L_{t-1}) + (1-β) * T_{t-1}
  // Forecast for next interval: L_t + T_t
  // α=0.3 (smoothing), β=0.1 (trend damping) — chosen for 100-500 ms intervals
  static auto   hw_lastCheck = std::chrono::steady_clock::now();
  static float  hw_level  = 0.0f;  // current level (faults/s)
  static float  hw_trend  = 0.0f;  // current trend (Δ faults/s per interval)
  static constexpr float kAlpha = 0.30f;
  static constexpr float kBeta  = 0.10f;

  auto hw_now = std::chrono::steady_clock::now();
  double hw_dt = std::chrono::duration<double>(hw_now - hw_lastCheck).count();
  if (hw_dt > 0.1) {
    float x_t = static_cast<float>(faultCount_.exchange(0) / hw_dt); // observed rate
    float L_prev = hw_level;
    hw_level = kAlpha * x_t + (1.0f - kAlpha) * (hw_level + hw_trend);
    hw_trend = kBeta  * (hw_level - L_prev) + (1.0f - kBeta) * hw_trend;
    // pageFaultRate_ stores the one-step-ahead forecast: L + T
    pageFaultRate_.store(hw_level + hw_trend, std::memory_order_relaxed);
    hw_lastCheck = hw_now;
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

// ── Adaptive UVM Page Migration ────────────────────────────────────────────
void MemoryManager::calibrateBandwidth() {
  const size_t testSize = 64 * 1024 * 1024; // 64MB for better cache pressure
  void *hostPtr = alignedAlloc(testSize, 64);
  void *devicePtr = alignedAlloc(testSize, 64);

  if (!hostPtr || !devicePtr) {
    if (hostPtr) alignedFree(hostPtr);
    if (devicePtr) alignedFree(devicePtr);
    return;
  }

  auto runTest = [&](void* dst, void* src, size_t size, int iterations) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
      memcpy(dst, src, size);
    }
    
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      memcpy(dst, src, size);
    }
    auto end = std::chrono::steady_clock::now();
    
    double sec = std::chrono::duration<double>(end - start).count();
    if (sec <= 0.0) return 0.0;
    return (static_cast<double>(size) * iterations / (1024.0 * 1024.0 * 1024.0)) / sec;
  };

  double h2d = runTest(devicePtr, hostPtr, testSize, 50);
  double d2d = runTest(devicePtr, hostPtr, testSize, 50);

  h2dBandwidth_.store(h2d);
  d2dBandwidth_.store(d2d);
  d2hBandwidth_.store(h2d);

  alignedFree(hostPtr);
  alignedFree(devicePtr);

  // ── NUMA Latency Model calibration ─────────────────────────────────────────
  // Populate the per-node inter-node bandwidth matrix using the measured local
  // bandwidth as the baseline for same-node transfers.  Cross-NUMA penalties
  // are derived from published AMD EPYC / Intel Xeon characterisation data:
  //   • Same node: full measured bandwidth (h2d GB/s)
  //   • 1-hop (adjacent NUMA domain): ~50% of local bandwidth
  //   • 2-hop (remote NUMA domain):   ~25% of local bandwidth
  //
  // On a single-socket machine all nodes share the same interconnect so we
  // report the same bandwidth for all pairs (which degrades gracefully).
  {
    // Detect the actual NUMA topology: how many nodes are present?
#if defined(__linux__)
    int numNodes = 1;
    for (int n = 0; n < kMaxNumaNodes; ++n) {
        // /sys/devices/system/node/nodeN exists for each NUMA node
        char path[64];
        std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", n);
        struct stat st{};
        if (::stat(path, &st) == 0 && S_ISDIR(st.st_mode)) numNodes = n + 1;
        else break;
    }
#else
    int numNodes = 1;
#endif

    for (int src = 0; src < numNodes; ++src) {
        for (int dst = 0; dst < numNodes; ++dst) {
            double bw;
            int hops = std::abs(dst - src);
            if (hops == 0)      bw = h2d;               // same node
            else if (hops == 1) bw = h2d * 0.50;        // 1-hop
            else                bw = h2d * 0.25;        // 2-hop+
            g_numaBandwidthGBps[src][dst].store(bw, std::memory_order_relaxed);

            // Latency model: same-node ~100 ns, +75 ns per hop (typical EPYC/Xeon)
            double lat = 100.0 + hops * 75.0;
            g_numaLatencyNs[src][dst].store(lat, std::memory_order_relaxed);
        }
    }
    VGRE_LOG_INFO("MemoryManager",
        "NUMA latency model calibrated: " + std::to_string(numNodes) +
        " node(s), local BW=" + std::to_string(h2d) + " GB/s");
  }

  VGRE_LOG_INFO("MemoryManager", "Bandwidth Calibrated — Host-to-Device: " +
                                     std::to_string(h2dBandwidth_.load()) +
                                     " GB/s");
}

// ── Memory Pool APIs ──────────────────────────────────────────────────────


VGREResult MemoryManager::getDirtyPages(MemoryHandle handle, std::vector<std::pair<size_t, size_t>>& outDirtyRanges) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);

  // O(log n): std::map binary search by exact base address
  auto it = masterRegions_.find(reinterpret_cast<uintptr_t>(handle));

  if (it == masterRegions_.end()) return VGREResult::ERR_INVALID_VALUE;
  
  outDirtyRanges.clear();
  if (!it->second.dirtyPages) return VGREResult::SUCCESS;

  size_t startIdx = 0;
  bool inRange = false;
  for (size_t i = 0; i < it->second.pageCount; ++i) {
    if (it->second.dirtyPages[i]) {
      if (!inRange) {
        startIdx = i;
        inRange = true;
      }
    } else {
      if (inRange) {
        outDirtyRanges.push_back(std::make_pair(startIdx * it->second.pageSize, i * it->second.pageSize));
        inRange = false;
      }
    }
  }
  if (inRange) {
    outDirtyRanges.push_back(std::make_pair(startIdx * it->second.pageSize, it->second.pageCount * it->second.pageSize));
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::clearDirtyPages(MemoryHandle handle) {
  // Pending-fault drainer can still hold queued write-fault addresses that
  // occurred just before clearDirtyPages() was called. If we clear the bitmap
  // immediately, those stale queued faults can re-dirty old pages and produce
  // non-deterministic dirty ranges. Wait briefly for the queue to drain first.
  {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> allocLock(allocatorMutex_);
    allocatorCv_.wait_for(allocLock, 50ms, [this]() {
        return pendingTail_.load(std::memory_order_acquire) >= pendingHead_.load(std::memory_order_acquire);
    });
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);

  // O(log n): std::map binary search by exact base address
  auto it = masterRegions_.find(reinterpret_cast<uintptr_t>(handle));

  if (it == masterRegions_.end()) return VGREResult::ERR_INVALID_VALUE;

  if (it->second.dirtyPages) {
    memset(it->second.dirtyPages, 0, it->second.pageCount);
  }

  // Re-protect to PROT_READ to catch the next write, but ONLY for regions
  // that were originally allocated with PROT_NONE (flags != 2). Host-accessible
  // regions (flags == 2) stay writable for application-level managed memory.
  if (!it->second.hostAccessible.load(std::memory_order_relaxed)) {
    vgre::os::mprotect_ro(it->second.ptr, it->second.size);
  }

  return VGREResult::SUCCESS;
}

// ── Singleton accessor ────────────────────────────────────────────────────
MemoryManager &MemoryManager::instance() {
  return RuntimeEngine::instance().getMemoryManager();
}

// ── TLB telemetry (Fix 3) ────────────────────────────────────────────────
MemoryManager::TlbStats MemoryManager::getTlbStats() {
    TlbStats s;
    s.l1Hits   = g_tlbStats.l1Hits.load(std::memory_order_relaxed);
    s.l1Misses = g_tlbStats.l1Misses.load(std::memory_order_relaxed);
    s.l2Hits   = g_tlbStats.l2Hits.load(std::memory_order_relaxed);
    s.l2Misses = g_tlbStats.l2Misses.load(std::memory_order_relaxed);
    s.l1HitRate = g_tlbStats.l1HitRate();
    s.l2HitRate = g_tlbStats.l2HitRate();
    return s;
}

} // namespace core
} // namespace vgre
