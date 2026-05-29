// CUPTI shim — software-proxy implementation backed by RuntimeProfiler.
//
// Hardware PMU counters are read via platform-native APIs:
//   Linux:   perf_event_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)
//   macOS:   mach_absolute_time() + QueryThreadCycleTime equivalent via
//            thread_info(THREAD_BASIC_INFO) for user-time instruction estimate
//   Windows: QueryThreadCycleTime() for actual TSC cycle deltas
//
// All platforms fall back to RuntimeProfiler instruction-mix proxies when
// hardware counters are unavailable (paranoid>1, sandbox, missing privilege).

#include "vgre/api/cupti_shim.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <chrono>
#include <cmath>
#include <string>
#include <atomic>
#include <algorithm>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_info.h>
#include <mach/mach_time.h>
#endif

#if defined(_WIN32)
#include "vgre/common/os_backend.h"
#endif

// ── Hardware PMU sampler ──────────────────────────────────────────────────────
// Thin RAII wrapper for per-thread hardware instruction/cycle counter.
// Instances are thread_local; start/stop bracket the region to measure.
namespace {

struct HwPmuSampler {
#if defined(__linux__)
    int fd = -1;

    HwPmuSampler() {
        struct perf_event_attr attr{};
        attr.type           = PERF_TYPE_HARDWARE;
        attr.size           = sizeof(attr);
        attr.config         = PERF_COUNT_HW_INSTRUCTIONS;
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        fd = static_cast<int>(syscall(SYS_perf_event_open, &attr,
                                      0, -1, -1, 0));
    }
    ~HwPmuSampler() { if (fd >= 0) { ::close(fd); fd = -1; } }

    bool valid() const { return fd >= 0; }
    void start() {
        if (fd < 0) return;
        ioctl(fd, PERF_EVENT_IOC_RESET,  0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }
    uint64_t stop() {
        if (fd < 0) return 0;
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t n = 0;
        if (::read(fd, &n, sizeof(n)) != (ssize_t)sizeof(n)) n = 0;
        return n;
    }

#elif defined(__APPLE__)
    // macOS: measure user-mode CPU time via mach thread_basic_info.
    // thread_basic_info::user_time gives microseconds of CPU time for this thread.
    // Not instruction counts, but proportional and hardware-derived.
    uint64_t startNs = 0;
    bool available = false;

    HwPmuSampler() {
        // Check if thread_info is accessible (always is on macOS user processes)
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        available = (thread_info(mach_thread_self(), THREAD_BASIC_INFO,
                                 reinterpret_cast<thread_info_t>(&info),
                                 &count) == KERN_SUCCESS);
    }
    ~HwPmuSampler() = default;

    bool valid() const { return available; }
    void start() {
        if (!available) return;
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        if (thread_info(mach_thread_self(), THREAD_BASIC_INFO,
                        reinterpret_cast<thread_info_t>(&info), &count) == KERN_SUCCESS) {
            startNs = static_cast<uint64_t>(info.user_time.seconds) * 1000000000ULL
                    + static_cast<uint64_t>(info.user_time.microseconds) * 1000ULL;
        }
    }
    // Returns elapsed user-mode nanoseconds as a proxy for instruction count
    // (CPU-time-proportional, not raw instructions, but hardware-measured).
    uint64_t stop() {
        if (!available) return 0;
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        if (thread_info(mach_thread_self(), THREAD_BASIC_INFO,
                        reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS)
            return 0;
        uint64_t nowNs = static_cast<uint64_t>(info.user_time.seconds) * 1000000000ULL
                       + static_cast<uint64_t>(info.user_time.microseconds) * 1000ULL;
        return (nowNs > startNs) ? (nowNs - startNs) : 0;
    }

#elif defined(_WIN32)
    // Windows: QueryThreadCycleTime reads the actual CPU TSC for this thread.
    uint64_t startCycles = 0;
    bool available = false;

    HwPmuSampler() {
        ULONG64 dummy = 0;
        available = (QueryThreadCycleTime(GetCurrentThread(), &dummy) != 0);
    }
    ~HwPmuSampler() = default;

    bool valid() const { return available; }
    void start() {
        if (!available) return;
        ULONG64 c = 0;
        if (QueryThreadCycleTime(GetCurrentThread(), &c)) startCycles = c;
    }
    uint64_t stop() {
        if (!available) return 0;
        ULONG64 c = 0;
        if (!QueryThreadCycleTime(GetCurrentThread(), &c)) return 0;
        return (c > startCycles) ? (c - startCycles) : 0;
    }

#else
    bool valid() const { return false; }
    void start() {}
    uint64_t stop() { return 0; }
#endif

    HwPmuSampler(const HwPmuSampler&) = delete;
    HwPmuSampler& operator=(const HwPmuSampler&) = delete;
};

static thread_local HwPmuSampler* t_pmuSampler = nullptr;
static HwPmuSampler& getThreadPmu() {
    if (!t_pmuSampler) t_pmuSampler = new HwPmuSampler();
    return *t_pmuSampler;
}

} // anonymous namespace — HwPmuSampler

// ── Internal state ────────────────────────────────────────────────────────────

namespace {

struct Subscriber {
    CUpti_CallbackFunc func     = nullptr;
    void*              userdata = nullptr;
    bool               active   = false;
    std::vector<std::pair<CUpti_CallbackDomain, uint32_t>> enabled;
};

struct EventGroup {
    std::vector<CUpti_EventID> events;
    // Hardware PMU snapshot: captured on cuptiEventGroupDisable().
    // Zero means no hardware read has completed yet for this group.
    uint64_t hwInstrCount = 0;
    bool     hwValid      = false;
};

struct ActivityRecord {
    CUpti_ActivityKernel5 kernel;
};

static std::mutex                                    g_mutex;
static std::unordered_map<uintptr_t, Subscriber>    g_subscribers;
static std::atomic<uintptr_t>                        g_nextSubId{1};
static bool                                          g_kernelActivityEnabled = false;

static CUpti_BuffersCallbackRequestFunc  g_bufReq  = nullptr;
static CUpti_BuffersCallbackCompleteFunc g_bufComp = nullptr;

static std::vector<ActivityRecord>  g_pending;
static std::mutex                   g_pendingMu;

// Event group state: maps handle → EventGroup
static std::mutex                                       g_egMutex;
static std::unordered_map<uintptr_t, EventGroup>        g_eventGroups;

// ── Epoch for nanosecond timestamps ──────────────────────────────────────────
static std::chrono::steady_clock::time_point g_epoch =
    std::chrono::steady_clock::now();

static uint64_t nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - g_epoch).count());
}

// ── Flush one batch of activity records to the registered buffer callbacks ───
static void flushRecords() {
    if (!g_bufReq || !g_bufComp) return;

    std::vector<ActivityRecord> snap;
    {
        std::lock_guard<std::mutex> lk(g_pendingMu);
        snap.swap(g_pending);
    }
    if (snap.empty()) return;

    uint8_t* buf  = nullptr;
    size_t   size = 0;
    size_t   maxN = 0;
    g_bufReq(&buf, &size, &maxN);
    if (!buf || size == 0) return;

    size_t written = 0;
    for (auto& rec : snap) {
        if (written + sizeof(CUpti_ActivityKernel5) > size) break;
        memcpy(buf + written, &rec.kernel, sizeof(CUpti_ActivityKernel5));
        written += sizeof(CUpti_ActivityKernel5);
    }
    g_bufComp(nullptr, 0, buf, size, written);
}

// ── Compute achieved occupancy from instruction mix ───────────────────────────
// Rationale: ALU-heavy kernels hide latency well (high occupancy).
// Memory-heavy kernels stall on cache misses (lower occupancy).
// Formula: base=0.50, +0.38×alu_fraction, -0.18×mem_fraction, clamped [0.10, 0.95].
// Time-weighted across all profiled kernels.
static double computeAchievedOccupancy(
    const std::vector<vgre::advanced::KernelStats>& allStats)
{
    double weightedOcc  = 0.0;
    double totalTimeMs  = 0.0;

    for (const auto& ks : allStats) {
        uint64_t total = ks.instructionMix.total();
        double occ;
        if (total == 0 || ks.totalTimeMs <= 0.0) {
            occ = 0.50;
        } else {
            double aluFrac =
                static_cast<double>(ks.instructionMix.aluCount) / total;
            double memFrac =
                static_cast<double>(
                    ks.instructionMix.loadCount + ks.instructionMix.storeCount)
                / total;
            // ALU-bound → good warp latency hiding → higher occupancy
            // Memory-bound → cache stalls → lower achieved occupancy
            occ = 0.50 + 0.38 * aluFrac - 0.18 * memFrac;
            occ = std::max(0.10, std::min(0.95, occ));
        }
        weightedOcc  += occ * ks.totalTimeMs;
        totalTimeMs  += ks.totalTimeMs;
    }

    return (totalTimeMs > 0.0) ? weightedOcc / totalTimeMs : 0.0;
}

// ── Aggregate instruction-mix counters from all profiled kernels ──────────────
static vgre::advanced::InstructionSample aggregateInstructionMix(
    const std::vector<vgre::advanced::KernelStats>& allStats)
{
    vgre::advanced::InstructionSample agg{};
    for (const auto& ks : allStats) {
        agg.loadCount    += ks.instructionMix.loadCount;
        agg.storeCount   += ks.instructionMix.storeCount;
        agg.aluCount     += ks.instructionMix.aluCount;
        agg.barrierCount += ks.instructionMix.barrierCount;
        agg.branchCount  += ks.instructionMix.branchCount;
        agg.otherCount   += ks.instructionMix.otherCount;
    }
    return agg;
}

} // anonymous namespace

// ── Helper: convert RuntimeProfiler stats to an activity record ──────────────
static void pushKernelActivity(const vgre::advanced::KernelStats& ks) {
    if (!g_kernelActivityEnabled) return;

    ActivityRecord rec{};
    rec.kernel.kind          = CUPTI_ACTIVITY_KIND_KERNEL;
    rec.kernel.correlationId = 0;
    rec.kernel.deviceId      = 0;
    rec.kernel.contextId     = 0;
    rec.kernel.streamId      = 0;
    rec.kernel.gridX         = 1;
    rec.kernel.gridY         = 1;
    rec.kernel.gridZ         = 1;
    rec.kernel.blockX        = 1;
    rec.kernel.blockY        = 1;
    rec.kernel.blockZ        = 1;

    uint64_t durationNs  = static_cast<uint64_t>(ks.avgTimeMs * 1e6);
    rec.kernel.start     = nowNs();
    rec.kernel.end       = rec.kernel.start + durationNs;
    rec.kernel.completed = rec.kernel.end;

    uint64_t total = ks.instructionMix.total();
    rec.kernel.registersPerThread = 32;
    if (total > 0) {
        rec.kernel.aluActivePct =
            static_cast<float>(ks.instructionMix.aluCount)
            / static_cast<float>(total) * 100.0f;
        rec.kernel.srcAccessPct =
            static_cast<float>(
                ks.instructionMix.loadCount + ks.instructionMix.storeCount)
            / static_cast<float>(total) * 100.0f;
    }

    static std::unordered_map<std::string, std::string> nameStore;
    static std::mutex nameMu;
    {
        std::lock_guard<std::mutex> lk(nameMu);
        auto it = nameStore.emplace(ks.kernelName, ks.kernelName).first;
        rec.kernel.name = it->second.c_str();
    }

    std::lock_guard<std::mutex> lk(g_pendingMu);
    g_pending.push_back(rec);
}

// ── Public API ────────────────────────────────────────────────────────────────

extern "C" {

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                           CUpti_CallbackFunc callback,
                           void* userdata) {
    if (!subscriber || !callback) return CUPTI_ERROR_INVALID_PARAMETER;
    uintptr_t id = g_nextSubId.fetch_add(1);
    Subscriber sub;
    sub.func     = callback;
    sub.userdata = userdata;
    sub.active   = true;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_subscribers[id] = sub;
    }
    *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(id);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_subscribers.erase(reinterpret_cast<uintptr_t>(subscriber));
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_subscribers.find(reinterpret_cast<uintptr_t>(subscriber));
    if (it == g_subscribers.end()) return CUPTI_ERROR_INVALID_PARAMETER;
    auto& sub = it->second;
    auto key = std::make_pair(domain, static_cast<uint32_t>(cbid));
    if (enable) {
        sub.enabled.push_back(key);
    } else {
        auto& v = sub.enabled;
        v.erase(std::remove(v.begin(), v.end(), key), v.end());
    }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableDomain(uint32_t /*enable*/,
                              CUpti_SubscriberHandle /*subscriber*/,
                              CUpti_CallbackDomain /*domain*/) {
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
    if (kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        g_kernelActivityEnabled = true;
        vgre::advanced::RuntimeProfiler::instance().setEnabled(true);
    }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
    if (kind == CUPTI_ACTIVITY_KIND_KERNEL) g_kernelActivityEnabled = false;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
    if (!funcBufferRequested || !funcBufferCompleted)
        return CUPTI_ERROR_INVALID_PARAMETER;
    g_bufReq  = funcBufferRequested;
    g_bufComp = funcBufferCompleted;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityFlushAll(uint32_t /*flag*/) {
    auto allStats = vgre::advanced::RuntimeProfiler::instance().getAllStats();
    for (const auto& ks : allStats) pushKernelActivity(ks);
    flushRecords();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity** record) {
    if (!buffer || !record) return CUPTI_ERROR_INVALID_PARAMETER;
    static size_t s_offset = 0;
    if (s_offset + sizeof(CUpti_ActivityKernel5) > validBufferSizeBytes) {
        s_offset = 0;
        return CUPTI_ERROR_QUEUE_EMPTY;
    }
    *record = reinterpret_cast<CUpti_Activity*>(buffer + s_offset);
    s_offset += sizeof(CUpti_ActivityKernel5);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiGetTimestamp(uint64_t* timestamp) {
    if (!timestamp) return CUPTI_ERROR_INVALID_PARAMETER;
    *timestamp = nowNs();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiMetricGetIdFromName(CUdevice /*device*/,
                                     const char* metricName,
                                     CUpti_MetricID* metric) {
    if (!metricName || !metric) return CUPTI_ERROR_INVALID_PARAMETER;
    static const struct { const char* name; CUpti_MetricID id; } kTable[] = {
        { "ipc",                     CUPTI_METRIC_ID_IPC                    },
        { "achieved_occupancy",      CUPTI_METRIC_ID_ACHIEVED_OCCUPANCY     },
        { "flop_count_sp",           CUPTI_METRIC_ID_FLOP_COUNT_SP          },
        { "dram_read_throughput",    CUPTI_METRIC_ID_DRAM_READ_THROUGHPUT   },
        { "dram_write_throughput",   CUPTI_METRIC_ID_DRAM_WRITE_THROUGHPUT  },
        { "l1_global_load_hit_rate", CUPTI_METRIC_ID_L1_GLOBAL_LOAD_HIT    },
        { "branch_efficiency",       CUPTI_METRIC_ID_BRANCH_EFFICIENCY      },
        { "kernel_duration",         CUPTI_METRIC_ID_KERNEL_DURATION_NS     },
        { nullptr, 0 }
    };
    for (int i = 0; kTable[i].name; ++i) {
        if (std::strcmp(kTable[i].name, metricName) == 0) {
            *metric = kTable[i].id;
            return CUPTI_SUCCESS;
        }
    }
    return CUPTI_ERROR_INVALID_PARAMETER;
}

CUptiResult cuptiMetricGetValue(CUdevice /*device*/,
                                CUpti_MetricID metric,
                                uint32_t /*numEventSpecs*/,
                                void* /*eventSpecArray*/,
                                uint32_t /*numEvents*/,
                                CUpti_EventID* /*eventIdArray*/,
                                uint64_t* /*eventValueArray*/,
                                uint64_t /*timeDuration*/,
                                CUpti_MetricValue* metricValue) {
    if (!metricValue) return CUPTI_ERROR_INVALID_PARAMETER;

    auto allStats = vgre::advanced::RuntimeProfiler::instance().getAllStats();

    double   totalTimeMs       = 0.0;
    double   totalGBps         = 0.0;
    uint64_t totalFlops        = 0;
    uint64_t totalInst         = 0;
    uint64_t totalBranch       = 0;
    uint64_t totalBranchTaken  = 0;
    int      count             = 0;

    for (const auto& ks : allStats) {
        totalTimeMs        += ks.totalTimeMs;
        totalGBps          += ks.avgThroughputGBps;
        totalFlops         += static_cast<uint64_t>(
                                  ks.avgGflops * 1e9 * ks.avgTimeMs / 1000.0);
        totalInst          += ks.totalInstructions;
        totalBranch        += ks.instructionMix.branchCount;
        // Conservative: assume all branches are taken (no divergence data)
        totalBranchTaken   += ks.instructionMix.branchCount;
        ++count;
    }

    switch (metric) {
    case CUPTI_METRIC_ID_IPC:
        // instructions / (cycles ≈ totalTimeMs × assumed 1 GHz emulated clk)
        metricValue->metricValueDouble =
            (totalTimeMs > 0.0 && totalInst > 0)
            ? static_cast<double>(totalInst) / (totalTimeMs * 1e6)
            : 0.0;
        break;

    case CUPTI_METRIC_ID_ACHIEVED_OCCUPANCY:
        // Derived from ALU vs memory instruction fraction (see computeAchievedOccupancy).
        metricValue->metricValueDouble = computeAchievedOccupancy(allStats);
        break;

    case CUPTI_METRIC_ID_FLOP_COUNT_SP:
        metricValue->metricValueUint64 = totalFlops;
        break;

    case CUPTI_METRIC_ID_DRAM_READ_THROUGHPUT:
        metricValue->metricValueDouble =
            (count > 0) ? totalGBps / count * 0.6 : 0.0;
        break;

    case CUPTI_METRIC_ID_DRAM_WRITE_THROUGHPUT:
        metricValue->metricValueDouble =
            (count > 0) ? totalGBps / count * 0.4 : 0.0;
        break;

    case CUPTI_METRIC_ID_L1_GLOBAL_LOAD_HIT: {
        // Estimate from ALU:memory ratio: more ALU → better reuse → higher hit
        auto agg   = aggregateInstructionMix(allStats);
        uint64_t t = agg.total();
        double hitRate = 60.0;  // conservative default
        if (t > 0) {
            double aluFrac = static_cast<double>(agg.aluCount) / t;
            double memFrac = static_cast<double>(agg.loadCount + agg.storeCount) / t;
            // More ALU relative to memory → temporal reuse → higher L1 hit rate
            hitRate = 40.0 + 50.0 * aluFrac - 20.0 * memFrac;
            hitRate = std::max(10.0, std::min(95.0, hitRate));
        }
        metricValue->metricValueDouble = hitRate;
        break;
    }

    case CUPTI_METRIC_ID_BRANCH_EFFICIENCY:
        // Ratio of non-divergent branches. Proxy: assume all branches taken.
        metricValue->metricValueDouble =
            (totalBranch > 0)
            ? static_cast<double>(totalBranchTaken) / totalBranch * 100.0
            : 100.0;
        break;

    case CUPTI_METRIC_ID_KERNEL_DURATION_NS:
        metricValue->metricValueUint64 =
            static_cast<uint64_t>(totalTimeMs * 1e6);
        break;

    default:
        return CUPTI_ERROR_INVALID_PARAMETER;
    }
    return CUPTI_SUCCESS;
}

// ── Event group API ───────────────────────────────────────────────────────────
// Hardware PMU registers are not accessible; we return software-proxy values
// derived from the RuntimeProfiler instruction-mix counters so that profiling
// tools receive non-zero, proportional counts rather than silence.

CUptiResult cuptiEventGroupCreate(CUcontext /*ctx*/,
                                  CUpti_EventGroupHandle* eg,
                                  uint32_t /*flags*/) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;
    auto* handle = new EventGroup();
    {
        std::lock_guard<std::mutex> lk(g_egMutex);
        g_eventGroups[reinterpret_cast<uintptr_t>(handle)] = EventGroup{};
    }
    *eg = reinterpret_cast<CUpti_EventGroupHandle>(handle);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupDestroy(CUpti_EventGroupHandle eg) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;
    uintptr_t key = reinterpret_cast<uintptr_t>(eg);
    {
        std::lock_guard<std::mutex> lk(g_egMutex);
        g_eventGroups.erase(key);
    }
    delete reinterpret_cast<EventGroup*>(eg);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupAddEvent(CUpti_EventGroupHandle eg,
                                    CUpti_EventID event) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lk(g_egMutex);
    auto it = g_eventGroups.find(reinterpret_cast<uintptr_t>(eg));
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_PARAMETER;
    it->second.events.push_back(event);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupEnable(CUpti_EventGroupHandle eg) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;
    // Start the per-thread hardware PMU counter so we can snapshot it on disable.
    HwPmuSampler& pmu = getThreadPmu();
    if (pmu.valid()) pmu.start();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupDisable(CUpti_EventGroupHandle eg) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;
    // Stop hardware counter and cache the reading in the event group.
    HwPmuSampler& pmu = getThreadPmu();
    uint64_t hwCount = pmu.valid() ? pmu.stop() : 0;
    if (hwCount > 0) {
        std::lock_guard<std::mutex> lk(g_egMutex);
        auto it = g_eventGroups.find(reinterpret_cast<uintptr_t>(eg));
        if (it != g_eventGroups.end()) {
            it->second.hwInstrCount = hwCount;
            it->second.hwValid      = true;
        }
    }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupReadAllEvents(CUpti_EventGroupHandle eg,
                                         uint32_t /*flags*/,
                                         size_t* sizeBytes,
                                         uint64_t* buf,
                                         size_t* idSizeBytes,
                                         CUpti_EventID* idArray,
                                         size_t* numRead) {
    if (!eg) return CUPTI_ERROR_INVALID_PARAMETER;

    std::vector<CUpti_EventID> events;
    uint64_t hwTotal = 0;
    bool hwValid = false;
    {
        std::lock_guard<std::mutex> lk(g_egMutex);
        auto it = g_eventGroups.find(reinterpret_cast<uintptr_t>(eg));
        if (it != g_eventGroups.end()) {
            events   = it->second.events;
            hwTotal  = it->second.hwInstrCount;
            hwValid  = it->second.hwValid;
        }
    }

    size_t n = events.size();

    // Prefer hardware PMU reading (from cuptiEventGroupDisable snapshot).
    // If hardware is not available, fall back to instruction-mix software proxies.
    uint64_t counterTable[7];
    size_t tableSize;

    if (hwValid && hwTotal > 0) {
        // Distribute the hardware total across buckets using the instruction-mix
        // fractions as weights, keeping them hardware-calibrated.
        auto allStats = vgre::advanced::RuntimeProfiler::instance().getAllStats();
        auto agg = aggregateInstructionMix(allStats);
        uint64_t softTotal = agg.total();
        auto scale = [&](uint64_t softBucket) -> uint64_t {
            if (softTotal == 0) return hwTotal / 7;
            return static_cast<uint64_t>(
                static_cast<double>(hwTotal) *
                (static_cast<double>(softBucket) / static_cast<double>(softTotal)));
        };
        counterTable[0] = scale(agg.aluCount);
        counterTable[1] = scale(agg.loadCount);
        counterTable[2] = scale(agg.storeCount);
        counterTable[3] = scale(agg.branchCount);
        counterTable[4] = scale(agg.barrierCount);
        counterTable[5] = scale(agg.otherCount);
        counterTable[6] = hwTotal;
        tableSize = 7;
    } else {
        // Software-proxy fallback: pure instruction-mix counters.
        auto allStats = vgre::advanced::RuntimeProfiler::instance().getAllStats();
        auto agg = aggregateInstructionMix(allStats);
        uint64_t aggTotal = agg.total();
        counterTable[0] = agg.aluCount;
        counterTable[1] = agg.loadCount;
        counterTable[2] = agg.storeCount;
        counterTable[3] = agg.branchCount;
        counterTable[4] = agg.barrierCount;
        counterTable[5] = agg.otherCount;
        counterTable[6] = aggTotal;
        tableSize = 7;
    }

    if (sizeBytes)   *sizeBytes   = n * sizeof(uint64_t);
    if (idSizeBytes) *idSizeBytes = n * sizeof(CUpti_EventID);
    if (numRead)     *numRead     = n;

    for (size_t i = 0; i < n; ++i) {
        if (buf)     buf[i]     = (tableSize > 0) ? counterTable[i % tableSize] : 0;
        if (idArray) idArray[i] = events[i];
    }

    return CUPTI_SUCCESS;
}

CUptiResult cuptiProfilerInitialize(void)   { return CUPTI_SUCCESS; }
CUptiResult cuptiProfilerDeInitialize(void) { return CUPTI_SUCCESS; }

} // extern "C"
