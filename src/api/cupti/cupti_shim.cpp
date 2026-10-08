// VGRE profiling implementation. Its virtual counters are derived from recorded
// launch events; it never labels host estimates as physical GPU measurements.

#include "vgre/api/cupti_shim.h"
#include "vgre/advanced/runtime_profiler.h"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <chrono>
#include <string>
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <memory>
#include <limits>
#include <cmath>
#include <utility>
#include <new>

struct CUpti_Subscriber_st {
    CUpti_CallbackFunc callback = nullptr;
    void* userdata = nullptr;
    bool driverDomainEnabled = false;
    bool runtimeDomainEnabled = false;
    bool driverLaunchCallbackEnabled = false;
    bool runtimeLaunchCallbackEnabled = false;
    bool removing = false;
    size_t callbacksInFlight = 0;
    std::condition_variable callbacksFinished;
};

struct CUpti_EventGroup_st {};

// ── Internal state ────────────────────────────────────────────────────────────

namespace {

struct ActivityRecord {
    CUpti_ActivityKernel5 kernel;
};

struct RecordCursor {
    size_t offset = 0;
    size_t validSize = 0;
};

static std::atomic<bool> g_kernelActivityEnabled{false};
static std::mutex g_activityFlushMutex;
static size_t g_activityFlushedEventCount = 0;
static uint64_t g_activityGeneration = 0;
static std::mutex g_recordCursorMutex;
static std::unordered_map<const uint8_t*, RecordCursor> g_recordCursors;

static std::mutex g_bufferCallbackMutex;
static CUpti_BuffersCallbackRequestFunc  g_bufReq  = nullptr;
static CUpti_BuffersCallbackCompleteFunc g_bufComp = nullptr;

static std::vector<ActivityRecord>  g_pending;
static std::mutex                   g_pendingMu;

static std::mutex g_profilerLeaseMutex;
static size_t g_profilerLeaseCount = 0;
static bool g_profilerWasEnabled = false;
static std::mutex g_explicitProfilerInitMutex;
static uint32_t g_explicitProfilerInitCount = 0;
static std::atomic<bool> g_activityBufferCallbackInFlight{false};

struct ActivityBufferCallbackScope {
    ActivityBufferCallbackScope() {
        g_activityBufferCallbackInFlight.store(true, std::memory_order_release);
    }
    ~ActivityBufferCallbackScope() {
        g_activityBufferCallbackInFlight.store(false, std::memory_order_release);
    }
};

struct EventGroupState {
    std::vector<CUpti_EventID> eventIds;
    bool enabled = false;
    size_t startEventCount = 0;
    uint64_t startGeneration = 0;
    uint64_t values[3] = {0, 0, 0};
};
static std::mutex g_eventGroupMutex;
static std::unordered_map<CUpti_EventGroupHandle, EventGroupState> g_eventGroups;

static std::mutex g_subscriberMutex;
static std::unordered_map<CUpti_SubscriberHandle,
                          std::shared_ptr<CUpti_Subscriber_st>> g_subscribers;
static vgre::advanced::RuntimeProfiler::EventListenerId g_callbackListenerId = 0;
static thread_local bool g_insideCUPTICallback = false;

static void acquireProfiler() {
    std::lock_guard<std::mutex> lock(g_profilerLeaseMutex);
    auto& profiler = vgre::advanced::RuntimeProfiler::instance();
    if (g_profilerLeaseCount++ == 0) {
        g_profilerWasEnabled = profiler.isEnabled();
        if (!g_profilerWasEnabled) profiler.setEnabled(true);
    }
}

static void releaseProfiler() {
    std::lock_guard<std::mutex> lock(g_profilerLeaseMutex);
    if (g_profilerLeaseCount == 0) return;
    if (--g_profilerLeaseCount == 0 && !g_profilerWasEnabled)
        vgre::advanced::RuntimeProfiler::instance().setEnabled(false);
}

static bool isVirtualEvent(CUpti_EventID event) {
    return event == VGRE_CUPTI_EVENT_KERNEL_LAUNCHES ||
           event == VGRE_CUPTI_EVENT_KERNEL_DURATION_NS ||
           event == VGRE_CUPTI_EVENT_VIRTUAL_THREADS;
}

static size_t eventIndex(CUpti_EventID event) {
    if (event == VGRE_CUPTI_EVENT_KERNEL_LAUNCHES) return 0;
    if (event == VGRE_CUPTI_EVENT_KERNEL_DURATION_NS) return 1;
    return 2;
}

static bool checkedAdd(uint64_t& target, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - target) return false;
    target += value;
    return true;
}

static bool virtualThreadCount(const vgre::advanced::ProfileEvent& event,
                               uint64_t& count) {
    const uint64_t dimensions[] = {event.gridDim.x, event.gridDim.y, event.gridDim.z,
                                   event.blockDim.x, event.blockDim.y, event.blockDim.z};
    count = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension != 0 && count > std::numeric_limits<uint64_t>::max() / dimension)
            return false;
        count *= dimension;
    }
    return true;
}

static bool durationNs(const vgre::advanced::ProfileEvent& event, uint64_t& value) {
    if (!std::isfinite(event.durationMs) || event.durationMs < 0.0) return false;
    const long double ns = static_cast<long double>(event.durationMs) * 1000000.0L;
    if (ns >= static_cast<long double>(std::numeric_limits<uint64_t>::max()))
        return false;
    value = static_cast<uint64_t>(ns);
    return true;
}

static bool aggregateEvents(const std::vector<vgre::advanced::ProfileEvent>& events,
                            size_t begin, uint64_t* values) {
    if (begin > events.size()) begin = 0;
    uint64_t aggregate[3] = {0, 0, 0};
    for (size_t i = begin; i < events.size(); ++i) {
        if (!events[i].isKernelLaunch) continue;
        uint64_t threads = 0, duration = 0;
        if (!virtualThreadCount(events[i], threads) || !durationNs(events[i], duration) ||
            !checkedAdd(aggregate[0], 1) || !checkedAdd(aggregate[1], duration) ||
            !checkedAdd(aggregate[2], threads))
            return false;
    }
    std::copy(aggregate, aggregate + 3, values);
    return true;
}

static bool supportedLaunchCallback(CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid) {
    return (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
            cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) ||
           (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
            cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
}

static void dispatchLaunchCallbacks(const vgre::advanced::ProfileEvent& event) {
    if (!event.isKernelLaunch) return;
    struct Call { std::shared_ptr<CUpti_Subscriber_st> subscriber;
                  CUpti_CallbackDomain domain; CUpti_CallbackId cbid; };
    std::vector<Call> calls;
    CUpti_CallbackDomain domain;
    CUpti_CallbackId cbid;
    const char* functionName;
    if (event.api == vgre::advanced::ProfileApi::CudaRuntime) {
        domain = CUPTI_CB_DOMAIN_RUNTIME_API;
        cbid = CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
        functionName = "cudaLaunchKernel";
    } else if (event.api == vgre::advanced::ProfileApi::CudaDriver) {
        domain = CUPTI_CB_DOMAIN_DRIVER_API;
        cbid = CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
        functionName = "cuLaunchKernel";
    } else {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_subscriberMutex);
        calls.reserve(g_subscribers.size());
        for (const auto& entry : g_subscribers) {
            const auto& subscriber = entry.second;
            if (subscriber->removing) continue;
            const bool domainEnabled = domain == CUPTI_CB_DOMAIN_RUNTIME_API
                ? subscriber->runtimeDomainEnabled : subscriber->driverDomainEnabled;
            const bool callbackEnabled = domain == CUPTI_CB_DOMAIN_RUNTIME_API
                ? subscriber->runtimeLaunchCallbackEnabled
                : subscriber->driverLaunchCallbackEnabled;
            if (domainEnabled || callbackEnabled) {
                ++subscriber->callbacksInFlight;
                calls.push_back({subscriber, domain, cbid});
            }
        }
    }
    for (const Call& call : calls) {
        CUpti_CallbackData data{};
        data.size = sizeof(data);
        data.callbackSite = VGRE_CUPTI_API_COMPLETION;
        data.functionName = functionName;
        data.symbolName = event.kernelName.c_str();
        // No API/activity correlation is tracked by the virtual runtime.
        data.correlationId = 0;
        const bool previousCallbackState = g_insideCUPTICallback;
        g_insideCUPTICallback = true;
        try {
            call.subscriber->callback(call.subscriber->userdata,
                                      call.domain, call.cbid, &data);
        }
        catch (...) { /* C callbacks must not unwind through the runtime. */ }
        g_insideCUPTICallback = previousCallbackState;
        {
            std::lock_guard<std::mutex> lock(g_subscriberMutex);
            --call.subscriber->callbacksInFlight;
            call.subscriber->callbacksFinished.notify_all();
        }
    }
}

static void ensureCallbackListener() {
    if (g_callbackListenerId != 0) return;
    g_callbackListenerId = vgre::advanced::RuntimeProfiler::instance().addEventListener(
        [](const vgre::advanced::ProfileEvent& event) {
            dispatchLaunchCallbacks(event);
        });
}

static uint64_t encodeSteadyTimestampNs(
    std::chrono::steady_clock::time_point timestamp) {
    const int64_t nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
        timestamp.time_since_epoch()).count();
    return static_cast<uint64_t>(nanoseconds) ^ (uint64_t{1} << 63);
}

static uint64_t nowNs() {
    return encodeSteadyTimestampNs(std::chrono::steady_clock::now());
}

// ── Flush one batch of activity records to the registered buffer callbacks ───
static CUptiResult flushRecords() {
    CUpti_BuffersCallbackRequestFunc request = nullptr;
    CUpti_BuffersCallbackCompleteFunc complete = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_bufferCallbackMutex);
        request = g_bufReq;
        complete = g_bufComp;
    }
    std::vector<ActivityRecord> snap;
    {
        std::lock_guard<std::mutex> lk(g_pendingMu);
        try { snap = g_pending; }
        catch (const std::bad_alloc&) { return CUPTI_ERROR_OUT_OF_MEMORY; }
    }
    if (snap.empty()) return CUPTI_SUCCESS;
    if (!request || !complete) return CUPTI_ERROR_NOT_INITIALIZED;

    uint8_t* buf  = nullptr;
    size_t   size = 0;
    size_t   maxN = 0;
    try {
        ActivityBufferCallbackScope callbackScope;
        request(&buf, &size, &maxN);
    }
    catch (...) { return CUPTI_ERROR_UNKNOWN; }
    if (!buf || size < sizeof(CUpti_ActivityKernel5))
        return CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT;

    size_t recordLimit = size / sizeof(CUpti_ActivityKernel5);
    if (maxN != 0) recordLimit = std::min(recordLimit, maxN);
    const size_t recordsToWrite = std::min(recordLimit, snap.size());
    const size_t written = recordsToWrite * sizeof(CUpti_ActivityKernel5);
    for (size_t i = 0; i < recordsToWrite; ++i)
        memcpy(buf + i * sizeof(CUpti_ActivityKernel5), &snap[i].kernel,
               sizeof(CUpti_ActivityKernel5));

    {
        std::lock_guard<std::mutex> lk(g_recordCursorMutex);
        g_recordCursors.erase(buf);
    }
    try {
        ActivityBufferCallbackScope callbackScope;
        complete(nullptr, 0, buf, size, written);
    }
    catch (...) { return CUPTI_ERROR_UNKNOWN; }
    {
        std::lock_guard<std::mutex> lk(g_pendingMu);
        const size_t consumed = std::min(recordsToWrite, g_pending.size());
        g_pending.erase(g_pending.begin(), g_pending.begin() + consumed);
    }
    return recordsToWrite == snap.size()
        ? CUPTI_SUCCESS : CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT;
}

} // anonymous namespace

// ── Helper: convert one recorded CUDA launch to an activity record ───────────
static CUptiResult pushKernelActivity(const vgre::advanced::ProfileEvent& event) {
    if (!g_kernelActivityEnabled) return CUPTI_SUCCESS;
    if (!event.isKernelLaunch) return CUPTI_SUCCESS;
    if (event.gridDim.x == 0 || event.gridDim.y == 0 || event.gridDim.z == 0 ||
        event.blockDim.x == 0 || event.blockDim.y == 0 || event.blockDim.z == 0)
        return CUPTI_SUCCESS;

    ActivityRecord rec{};
    rec.kernel.kind          = CUPTI_ACTIVITY_KIND_KERNEL;
    // VGRE's profiler does not currently retain hardware device/context/stream
    // IDs or a CUDA correlation ID, so their documented zero values mean absent.
    rec.kernel.correlationId = 0;
    rec.kernel.deviceId      = 0;
    rec.kernel.contextId     = 0;
    rec.kernel.streamId      = 0;
    rec.kernel.gridX         = event.gridDim.x;
    rec.kernel.gridY         = event.gridDim.y;
    rec.kernel.gridZ         = event.gridDim.z;
    rec.kernel.blockX        = event.blockDim.x;
    rec.kernel.blockY        = event.blockDim.y;
    rec.kernel.blockZ        = event.blockDim.z;
    if (event.staticSharedMemoryBytes >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        event.dynamicSharedMemoryBytes >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return CUPTI_ERROR_INVALID_EVENT_VALUE;
    rec.kernel.staticSharedMemory =
        static_cast<int32_t>(event.staticSharedMemoryBytes);
    rec.kernel.dynamicSharedMemory =
        static_cast<int32_t>(event.dynamicSharedMemoryBytes);

    uint64_t eventDurationNs = 0;
    if (!durationNs(event, eventDurationNs)) return CUPTI_ERROR_INVALID_EVENT_VALUE;
    rec.kernel.end       = encodeSteadyTimestampNs(event.timestamp);
    rec.kernel.start     = rec.kernel.end - eventDurationNs;
    rec.kernel.completed = rec.kernel.end;
    // VGRE does not track hardware register usage or utilization percentages.
    rec.kernel.registersPerThread = 0;
    rec.kernel.aluActivePct = 0.0f;
    rec.kernel.srcAccessPct = 0.0f;

    static std::unordered_map<std::string, std::string> nameStore;
    static std::mutex nameMu;
    try {
        {
            std::lock_guard<std::mutex> lk(nameMu);
            auto it = nameStore.emplace(event.kernelName, event.kernelName).first;
            rec.kernel.name = it->second.c_str();
        }
        std::lock_guard<std::mutex> lk(g_pendingMu);
        g_pending.push_back(rec);
    } catch (const std::bad_alloc&) {
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    return CUPTI_SUCCESS;
}

static bool validSubscriber(CUpti_SubscriberHandle subscriber) {
    return subscriber && g_subscribers.find(subscriber) != g_subscribers.end();
}

// ── Public API ────────────────────────────────────────────────────────────────

extern "C" {

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                           CUpti_CallbackFunc callback,
                           void* userdata) {
    if (!subscriber || !callback) return CUPTI_ERROR_INVALID_PARAMETER;
    *subscriber = nullptr;
    std::shared_ptr<CUpti_Subscriber_st> state;
    try {
        state = std::make_shared<CUpti_Subscriber_st>();
        state->callback = callback;
        state->userdata = userdata;
    } catch (const std::bad_alloc&) {
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    CUpti_SubscriberHandle handle = state.get();
    bool firstSubscriber = false;
    vgre::advanced::RuntimeProfiler::EventListenerId listenerToRemove = 0;
    try {
        std::lock_guard<std::mutex> lock(g_subscriberMutex);
        firstSubscriber = g_subscribers.empty();
        if (firstSubscriber) {
            ensureCallbackListener();
            if (g_callbackListenerId == 0) return CUPTI_ERROR_OUT_OF_MEMORY;
            acquireProfiler();
        }
        try {
            g_subscribers.emplace(handle, state);
        } catch (...) {
            if (firstSubscriber) {
                listenerToRemove = g_callbackListenerId;
                g_callbackListenerId = 0;
                releaseProfiler();
            }
            throw;
        }
    } catch (const std::bad_alloc&) {
        if (listenerToRemove)
            vgre::advanced::RuntimeProfiler::instance().removeEventListener(listenerToRemove);
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    *subscriber = handle;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
    // Dispatch reserves in-flight references for every subscriber before it
    // invokes any callback. Waiting here can therefore deadlock when one
    // callback synchronously unsubscribes itself or another subscriber.
    if (g_insideCUPTICallback) return CUPTI_ERROR_INVALID_OPERATION;
    bool lastSubscriber = false;
    std::shared_ptr<CUpti_Subscriber_st> state;
    {
        std::unique_lock<std::mutex> lock(g_subscriberMutex);
        auto it = g_subscribers.find(subscriber);
        if (it == g_subscribers.end()) return CUPTI_ERROR_INVALID_HANDLE;
        state = it->second;
        state->removing = true;
        g_subscribers.erase(subscriber);
        lastSubscriber = g_subscribers.empty();
        if (lastSubscriber) {
            const auto listenerId = g_callbackListenerId;
            g_callbackListenerId = 0;
            if (listenerId)
                vgre::advanced::RuntimeProfiler::instance().removeEventListener(listenerId);
        }
        state->callbacksFinished.wait(lock, [&] {
            return state->callbacksInFlight == 0;
        });
    }
    if (lastSubscriber) releaseProfiler();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
    if (enable > 1) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_subscriberMutex);
    if (!validSubscriber(subscriber)) return CUPTI_ERROR_INVALID_HANDLE;
    if (!supportedLaunchCallback(domain, cbid)) return CUPTI_ERROR_NOT_SUPPORTED;
    auto* state = g_subscribers[subscriber].get();
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
        state->runtimeLaunchCallbackEnabled = enable != 0;
    else
        state->driverLaunchCallbackEnabled = enable != 0;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableDomain(uint32_t enable,
                              CUpti_SubscriberHandle subscriber,
                              CUpti_CallbackDomain domain) {
    if (enable > 1) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_subscriberMutex);
    if (!validSubscriber(subscriber)) return CUPTI_ERROR_INVALID_HANDLE;
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API && domain != CUPTI_CB_DOMAIN_RUNTIME_API)
        return CUPTI_ERROR_NOT_SUPPORTED;
    auto* state = g_subscribers[subscriber].get();
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
        state->runtimeDomainEnabled = enable != 0;
    else
        state->driverDomainEnabled = enable != 0;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
    if (kind != CUPTI_ACTIVITY_KIND_KERNEL) return CUPTI_ERROR_NOT_SUPPORTED;
    if (g_activityBufferCallbackInFlight.load(std::memory_order_acquire))
        return CUPTI_ERROR_INVALID_OPERATION;
    std::lock_guard<std::mutex> flushLock(g_activityFlushMutex);
    if (!g_kernelActivityEnabled.load(std::memory_order_acquire)) {
        std::pair<std::vector<vgre::advanced::ProfileEvent>, uint64_t> snapshot;
        try {
            snapshot = vgre::advanced::RuntimeProfiler::instance().getEventSnapshot();
        } catch (const std::bad_alloc&) {
            return CUPTI_ERROR_OUT_OF_MEMORY;
        }
        g_activityFlushedEventCount = snapshot.first.size();
        g_activityGeneration = snapshot.second;
        g_kernelActivityEnabled.store(true, std::memory_order_release);
        acquireProfiler();
    }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
    if (kind != CUPTI_ACTIVITY_KIND_KERNEL) return CUPTI_ERROR_NOT_SUPPORTED;
    if (g_activityBufferCallbackInFlight.load(std::memory_order_acquire))
        return CUPTI_ERROR_INVALID_OPERATION;
    std::lock_guard<std::mutex> flushLock(g_activityFlushMutex);
    if (g_kernelActivityEnabled.exchange(false, std::memory_order_acq_rel))
        releaseProfiler();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
    if (!funcBufferRequested || !funcBufferCompleted)
        return CUPTI_ERROR_INVALID_PARAMETER;
    {
        std::lock_guard<std::mutex> lock(g_bufferCallbackMutex);
        g_bufReq  = funcBufferRequested;
        g_bufComp = funcBufferCompleted;
    }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
    if (flag & ~CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)
        return CUPTI_ERROR_INVALID_PARAMETER;
    if (g_activityBufferCallbackInFlight.load(std::memory_order_acquire))
        return CUPTI_ERROR_INVALID_OPERATION;
    std::lock_guard<std::mutex> flushLock(g_activityFlushMutex);
    std::pair<std::vector<vgre::advanced::ProfileEvent>, uint64_t> snapshot;
    try {
        snapshot = vgre::advanced::RuntimeProfiler::instance().getEventSnapshot();
    } catch (const std::bad_alloc&) {
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    const auto& events = snapshot.first;
    if (snapshot.second != g_activityGeneration ||
        events.size() < g_activityFlushedEventCount) {
        g_activityFlushedEventCount = 0;
        g_activityGeneration = snapshot.second;
    }
    for (size_t i = g_activityFlushedEventCount; i < events.size(); ++i) {
        const CUptiResult result = pushKernelActivity(events[i]);
        if (result != CUPTI_SUCCESS) return result;
        g_activityFlushedEventCount = i + 1;
    }
    return flushRecords();
}

CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity** record) {
    if (!buffer || !record) return CUPTI_ERROR_INVALID_PARAMETER;
    *record = nullptr;
    std::lock_guard<std::mutex> lk(g_recordCursorMutex);
    auto it = g_recordCursors.find(buffer);
    if (it == g_recordCursors.end()) {
        try { it = g_recordCursors.emplace(buffer, RecordCursor{}).first; }
        catch (const std::bad_alloc&) { return CUPTI_ERROR_OUT_OF_MEMORY; }
    }
    RecordCursor& cursor = it->second;
    if (cursor.validSize != validBufferSizeBytes)
        cursor = RecordCursor{0, validBufferSizeBytes};
    if (cursor.offset > validBufferSizeBytes ||
        sizeof(CUpti_ActivityKernel5) > validBufferSizeBytes - cursor.offset)
        return CUPTI_ERROR_QUEUE_EMPTY;
    *record = reinterpret_cast<CUpti_Activity*>(buffer + cursor.offset);
    cursor.offset += sizeof(CUpti_ActivityKernel5);
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
    if (std::strcmp(metricName, "vgre_kernel_duration_ns") == 0 ||
        std::strcmp(metricName, "kernel_duration_ns") == 0 ||
        std::strcmp(metricName, "kernel_duration") == 0)
        *metric = VGRE_CUPTI_METRIC_KERNEL_DURATION_NS;
    else if (std::strcmp(metricName, "vgre_kernel_launches") == 0 ||
             std::strcmp(metricName, "kernel_launches") == 0)
        *metric = VGRE_CUPTI_METRIC_KERNEL_LAUNCHES;
    else if (std::strcmp(metricName, "vgre_virtual_threads") == 0 ||
             std::strcmp(metricName, "virtual_threads_launched") == 0)
        *metric = VGRE_CUPTI_METRIC_VIRTUAL_THREADS;
    else if (std::strcmp(metricName, "vgre_average_kernel_duration_ns") == 0 ||
             std::strcmp(metricName, "average_kernel_duration_ns") == 0)
        *metric = VGRE_CUPTI_METRIC_AVG_DURATION_NS;
    else if (std::strcmp(metricName, "ipc") == 0 ||
             std::strcmp(metricName, "occupancy") == 0 ||
             std::strcmp(metricName, "achieved_occupancy") == 0 ||
             std::strcmp(metricName, "flop_count") == 0 ||
             std::strcmp(metricName, "flop_count_sp") == 0 ||
             std::strcmp(metricName, "dram_read_bytes") == 0 ||
             std::strcmp(metricName, "dram_read_throughput") == 0 ||
             std::strcmp(metricName, "dram_write_bytes") == 0 ||
             std::strcmp(metricName, "dram_write_throughput") == 0 ||
             std::strcmp(metricName, "l1_hit_rate") == 0 ||
             std::strcmp(metricName, "l1_global_load_hit") == 0 ||
             std::strcmp(metricName, "l1_global_load_hit_rate") == 0 ||
             std::strcmp(metricName, "branch_efficiency") == 0)
        return CUPTI_ERROR_NOT_SUPPORTED;
    else
        return CUPTI_ERROR_INVALID_METRIC_NAME;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiMetricGetValue(CUdevice /*device*/,
                                CUpti_MetricID metric,
                                uint32_t numEventSpecs,
                                void* eventSpecArray,
                                uint32_t numEvents,
                                CUpti_EventID* eventIdArray,
                                uint64_t* eventValueArray,
                                uint64_t timeDuration,
                                CUpti_MetricValue* metricValue) {
    if (!metricValue) return CUPTI_ERROR_INVALID_PARAMETER;
    if (timeDuration != 0 || (numEventSpecs != 0 && !eventSpecArray) ||
        (numEvents != 0 && (!eventIdArray || !eventValueArray)))
        return CUPTI_ERROR_INVALID_PARAMETER;
    if (metric >= VGRE_CUPTI_METRIC_IPC && metric <= VGRE_CUPTI_METRIC_BRANCH_EFFICIENCY)
        return CUPTI_ERROR_NOT_SUPPORTED;
    if (metric < VGRE_CUPTI_METRIC_KERNEL_DURATION_NS ||
        metric > VGRE_CUPTI_METRIC_AVG_DURATION_NS)
        return CUPTI_ERROR_INVALID_METRIC_ID;

    uint64_t values[3] = {0, 0, 0};
    bool present[3] = {false, false, false};
    for (uint32_t i = 0; i < numEvents; ++i) {
        if (!isVirtualEvent(eventIdArray[i])) return CUPTI_ERROR_INVALID_EVENT_ID;
        const size_t index = eventIndex(eventIdArray[i]);
        if (present[index]) return CUPTI_ERROR_INVALID_EVENT_VALUE;
        values[index] = eventValueArray[i];
        present[index] = true;
    }
    const auto* specs = static_cast<const CUpti_EventID*>(eventSpecArray);
    bool seenSpecs[3] = {false, false, false};
    for (uint32_t i = 0; i < numEventSpecs; ++i) {
        if (!isVirtualEvent(specs[i])) return CUPTI_ERROR_INVALID_EVENT_ID;
        const size_t index = eventIndex(specs[i]);
        if (seenSpecs[index] || !present[index])
            return CUPTI_ERROR_INVALID_EVENT_VALUE;
        seenSpecs[index] = true;
    }

    CUpti_EventID required[2]{};
    size_t requiredCount = 1;
    switch (metric) {
    case VGRE_CUPTI_METRIC_KERNEL_DURATION_NS:
        required[0] = VGRE_CUPTI_EVENT_KERNEL_DURATION_NS; break;
    case VGRE_CUPTI_METRIC_KERNEL_LAUNCHES:
        required[0] = VGRE_CUPTI_EVENT_KERNEL_LAUNCHES; break;
    case VGRE_CUPTI_METRIC_VIRTUAL_THREADS:
        required[0] = VGRE_CUPTI_EVENT_VIRTUAL_THREADS; break;
    case VGRE_CUPTI_METRIC_AVG_DURATION_NS:
        required[0] = VGRE_CUPTI_EVENT_KERNEL_DURATION_NS;
        required[1] = VGRE_CUPTI_EVENT_KERNEL_LAUNCHES;
        requiredCount = 2; break;
    default: return CUPTI_ERROR_INVALID_METRIC_ID;
    }
    for (size_t i = 0; i < requiredCount; ++i)
        if (!present[eventIndex(required[i])])
            return CUPTI_ERROR_INVALID_EVENT_VALUE;
    for (uint32_t i = 0; i < numEventSpecs; ++i) {
        bool requiredSpec = false;
        for (size_t j = 0; j < requiredCount; ++j)
            requiredSpec = requiredSpec || specs[i] == required[j];
        if (!requiredSpec) return CUPTI_ERROR_INVALID_EVENT_VALUE;
    }
    if (numEventSpecs != 0) {
        for (size_t i = 0; i < requiredCount; ++i)
            if (!seenSpecs[eventIndex(required[i])])
                return CUPTI_ERROR_INVALID_EVENT_VALUE;
    }

    switch (metric) {
    case VGRE_CUPTI_METRIC_KERNEL_DURATION_NS:
        metricValue->metricValueUint64 = values[eventIndex(VGRE_CUPTI_EVENT_KERNEL_DURATION_NS)]; break;
    case VGRE_CUPTI_METRIC_KERNEL_LAUNCHES:
        metricValue->metricValueUint64 = values[eventIndex(VGRE_CUPTI_EVENT_KERNEL_LAUNCHES)]; break;
    case VGRE_CUPTI_METRIC_VIRTUAL_THREADS:
        metricValue->metricValueUint64 = values[eventIndex(VGRE_CUPTI_EVENT_VIRTUAL_THREADS)]; break;
    case VGRE_CUPTI_METRIC_AVG_DURATION_NS: {
        const uint64_t launches = values[eventIndex(VGRE_CUPTI_EVENT_KERNEL_LAUNCHES)];
        if (launches == 0) return CUPTI_ERROR_INVALID_EVENT_VALUE;
        metricValue->metricValueDouble =
            static_cast<double>(values[eventIndex(VGRE_CUPTI_EVENT_KERNEL_DURATION_NS)]) /
            static_cast<double>(launches);
        break;
    }
    default: return CUPTI_ERROR_INVALID_METRIC_ID;
    }
    return CUPTI_SUCCESS;
}

// These counters describe the actual virtual work submitted to the runtime.
CUptiResult cuptiEventGroupCreate(CUcontext ctx,
                                  CUpti_EventGroupHandle* eventGroup,
                                  uint32_t flags) {
    if (!eventGroup) return CUPTI_ERROR_INVALID_PARAMETER;
    *eventGroup = nullptr;
    if (flags != 0) return CUPTI_ERROR_INVALID_PARAMETER;
    if (ctx != nullptr) return CUPTI_ERROR_INVALID_CONTEXT;
    auto* handle = new (std::nothrow) CUpti_EventGroup_st;
    if (!handle) return CUPTI_ERROR_OUT_OF_MEMORY;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    try {
        EventGroupState state;
        g_eventGroups.emplace(handle, std::move(state));
    } catch (const std::bad_alloc&) {
        delete handle;
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    *eventGroup = handle;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupDestroy(CUpti_EventGroupHandle eventGroup) {
    if (!eventGroup) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    auto it = g_eventGroups.find(eventGroup);
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_HANDLE;
    if (it->second.enabled) return CUPTI_ERROR_INVALID_OPERATION;
    g_eventGroups.erase(it);
    delete eventGroup;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupAddEvent(CUpti_EventGroupHandle eventGroup,
                                    CUpti_EventID event) {
    if (!eventGroup) return CUPTI_ERROR_INVALID_PARAMETER;
    if (!isVirtualEvent(event)) return CUPTI_ERROR_INVALID_EVENT_ID;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    auto it = g_eventGroups.find(eventGroup);
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_HANDLE;
    if (it->second.enabled) return CUPTI_ERROR_INVALID_OPERATION;
    if (std::find(it->second.eventIds.begin(), it->second.eventIds.end(), event) !=
        it->second.eventIds.end()) return CUPTI_ERROR_INVALID_EVENT_ID;
    try { it->second.eventIds.push_back(event); }
    catch (const std::bad_alloc&) { return CUPTI_ERROR_OUT_OF_MEMORY; }
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupEnable(CUpti_EventGroupHandle eventGroup) {
    if (!eventGroup) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    auto it = g_eventGroups.find(eventGroup);
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_HANDLE;
    EventGroupState& group = it->second;
    if (group.enabled || group.eventIds.empty()) return CUPTI_ERROR_INVALID_OPERATION;
    auto& profiler = vgre::advanced::RuntimeProfiler::instance();
    std::pair<std::vector<vgre::advanced::ProfileEvent>, uint64_t> snapshot;
    try { snapshot = profiler.getEventSnapshot(); }
    catch (const std::bad_alloc&) { return CUPTI_ERROR_OUT_OF_MEMORY; }
    group.startEventCount = snapshot.first.size();
    group.startGeneration = snapshot.second;
    group.enabled = true;
    acquireProfiler();
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupDisable(CUpti_EventGroupHandle eventGroup) {
    if (!eventGroup) return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    auto it = g_eventGroups.find(eventGroup);
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_HANDLE;
    EventGroupState& group = it->second;
    if (!group.enabled) return CUPTI_ERROR_INVALID_OPERATION;
    std::pair<std::vector<vgre::advanced::ProfileEvent>, uint64_t> snapshot;
    try {
        snapshot = vgre::advanced::RuntimeProfiler::instance().getEventSnapshot();
    } catch (const std::bad_alloc&) {
        return CUPTI_ERROR_OUT_OF_MEMORY;
    }
    uint64_t values[3] = {0, 0, 0};
    const size_t begin = snapshot.second == group.startGeneration
        ? group.startEventCount : 0;
    bool valid = aggregateEvents(snapshot.first, begin, values);
    for (size_t i = 0; valid && i < 3; ++i)
        valid = checkedAdd(values[i], group.values[i]);
    group.enabled = false;
    releaseProfiler();
    if (!valid) return CUPTI_ERROR_INVALID_EVENT_VALUE;
    std::copy(values, values + 3, group.values);
    return CUPTI_SUCCESS;
}

CUptiResult cuptiEventGroupReadAllEvents(CUpti_EventGroupHandle eventGroup,
                                         uint32_t flags,
                                         size_t* eventValueBufferSizeBytes,
                                         uint64_t* eventValueBuffer,
                                         size_t* eventIdArraySizeBytes,
                                         CUpti_EventID* eventIdArray,
                                         size_t* numEventIdsRead) {
    if (!eventGroup || !eventValueBufferSizeBytes || !eventIdArraySizeBytes ||
        !numEventIdsRead || flags != 0)
        return CUPTI_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> lock(g_eventGroupMutex);
    auto it = g_eventGroups.find(eventGroup);
    if (it == g_eventGroups.end()) return CUPTI_ERROR_INVALID_HANDLE;
    EventGroupState& group = it->second;
    if (group.enabled) return CUPTI_ERROR_INVALID_OPERATION;
    const size_t requiredValues = group.eventIds.size() * sizeof(uint64_t);
    const size_t requiredIds = group.eventIds.size() * sizeof(CUpti_EventID);
    *numEventIdsRead = 0;
    if (*eventValueBufferSizeBytes < requiredValues ||
        *eventIdArraySizeBytes < requiredIds) {
        *eventValueBufferSizeBytes = requiredValues;
        *eventIdArraySizeBytes = requiredIds;
        return CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT;
    }
    if (requiredValues && (!eventValueBuffer || !eventIdArray))
        return CUPTI_ERROR_INVALID_PARAMETER;
    for (size_t i = 0; i < group.eventIds.size(); ++i) {
        const CUpti_EventID id = group.eventIds[i];
        eventIdArray[i] = id;
        eventValueBuffer[i] = group.values[eventIndex(id)];
    }
    *eventValueBufferSizeBytes = requiredValues;
    *eventIdArraySizeBytes = requiredIds;
    *numEventIdsRead = group.eventIds.size();
    std::fill(group.values, group.values + 3, uint64_t{0});
    return CUPTI_SUCCESS;
}

CUptiResult cuptiProfilerInitialize(void) {
    std::lock_guard<std::mutex> lock(g_explicitProfilerInitMutex);
    if (g_explicitProfilerInitCount == std::numeric_limits<uint32_t>::max())
        return CUPTI_ERROR_OUT_OF_MEMORY;
    if (g_explicitProfilerInitCount == 0) acquireProfiler();
    ++g_explicitProfilerInitCount;
    return CUPTI_SUCCESS;
}

CUptiResult cuptiProfilerDeInitialize(void) {
    std::lock_guard<std::mutex> lock(g_explicitProfilerInitMutex);
    if (g_explicitProfilerInitCount == 0) return CUPTI_ERROR_NOT_INITIALIZED;
    --g_explicitProfilerInitCount;
    if (g_explicitProfilerInitCount == 0) releaseProfiler();
    return CUPTI_SUCCESS;
}

} // extern "C"
