// VGRE virtual CUPTI counters are derived from recorded runtime launch data.

#include "vgre/api/cupti_shim.h"
#include "vgre/api/cuda_interceptor.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/core/runtime_engine.h"

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <string>
#include <atomic>
#include <thread>
#include <vector>

extern "C" int cuLaunchKernel(vgre::api::CUfunction function,
                               unsigned int gridX, unsigned int gridY,
                               unsigned int gridZ, unsigned int blockX,
                               unsigned int blockY, unsigned int blockZ,
                               unsigned int sharedBytes,
                               vgre::api::cudaStream_t stream,
                               void** arguments, void** extra);

namespace {

int passed = 0;
int failed = 0;
alignas(CUpti_ActivityKernel5)
uint8_t activityBuffer[4 * sizeof(CUpti_ActivityKernel5)]{};
size_t activityValidBytes = 0;
int callbackCount = 0;
std::string callbackKernelName;
std::string callbackFunctionName;
CUpti_SubscriberHandle callbackSubscriber = nullptr;
bool attemptSelfUnsubscribe = false;
CUptiResult selfUnsubscribeResult = CUPTI_SUCCESS;
uint32_t callbackCorrelationId = 0;
bool testReentrantActivityFlush = false;
CUptiResult reentrantActivityFlushResult = CUPTI_SUCCESS;

void requestActivityBuffer(uint8_t** buffer, size_t* size, size_t* maxRecords) {
    *buffer = activityBuffer;
    *size = sizeof(activityBuffer);
    *maxRecords = 4;
}

void completeActivityBuffer(CUcontext, uint32_t, uint8_t*, size_t, size_t validSize) {
    activityValidBytes = validSize;
    if (testReentrantActivityFlush) {
        testReentrantActivityFlush = false;
        std::thread nestedFlush([&] {
            reentrantActivityFlushResult = cuptiActivityFlushAll(0);
        });
        nestedFlush.join();
    }
}

void launchCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                    const CUpti_CallbackData* info) {
    if (!userdata || !info || info->callbackSite != VGRE_CUPTI_API_COMPLETION) return;
    const bool runtimeLaunch = domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
    const bool driverLaunch = domain == CUPTI_CB_DOMAIN_DRIVER_API &&
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
    if (!runtimeLaunch && !driverLaunch) return;
    ++*static_cast<int*>(userdata);
    callbackKernelName = info->symbolName ? info->symbolName : "";
    callbackFunctionName = info->functionName ? info->functionName : "";
    callbackCorrelationId = info->correlationId;
    if (attemptSelfUnsubscribe) {
        attemptSelfUnsubscribe = false;
        selfUnsubscribeResult = cuptiUnsubscribe(callbackSubscriber);
    }
}

#define CHECK(condition, message) do { \
    if (condition) { std::printf("  PASS: %s\n", message); ++passed; } \
    else { std::printf("  FAIL: %s\n", message); ++failed; } \
} while (false)

vgre::advanced::ProfileEvent launch(const char* name, double durationMs,
                                    vgre::dim3 grid, vgre::dim3 block) {
    vgre::advanced::ProfileEvent event;
    event.kernelName = name;
    event.durationMs = durationMs;
    event.gridDim = grid;
    event.blockDim = block;
    event.api = vgre::advanced::ProfileApi::CudaRuntime;
    event.isKernelLaunch = true;
    event.timestamp = std::chrono::steady_clock::now();
    return event;
}

vgre::advanced::ProfileEvent annotation(const char* name, double durationMs,
                                        vgre::dim3 grid, vgre::dim3 block) {
    auto event = launch(name, durationMs, grid, block);
    event.isKernelLaunch = false;
    return event;
}

} // namespace

int main() {
    std::printf("=== CUPTI virtual launch profiling ===\n");
    auto& profiler = vgre::advanced::RuntimeProfiler::instance();
    profiler.setEnabled(false);
    profiler.clear();

    CUpti_EventGroupHandle group = nullptr;
    CHECK(cuptiEventGroupCreate(nullptr, nullptr, 0) == CUPTI_ERROR_INVALID_PARAMETER,
          "null event-group output is rejected");
    CHECK(cuptiEventGroupCreate(nullptr, &group, 1) == CUPTI_ERROR_INVALID_PARAMETER,
          "unsupported creation flags are rejected");
    CHECK(cuptiEventGroupCreate(reinterpret_cast<CUcontext>(uintptr_t{1}),
                                &group, 0) == CUPTI_ERROR_INVALID_CONTEXT,
          "unmapped opaque CUDA contexts are rejected rather than mixed together");
    CHECK(cuptiEventGroupCreate(nullptr, &group, 0) == CUPTI_SUCCESS && group,
          "event group receives a live handle");
    const auto fakeGroup = reinterpret_cast<CUpti_EventGroupHandle>(uintptr_t{1});
    CHECK(cuptiEventGroupAddEvent(fakeGroup, VGRE_CUPTI_EVENT_KERNEL_LAUNCHES) ==
              CUPTI_ERROR_INVALID_HANDLE,
          "fabricated event-group handles are rejected");
    CHECK(cuptiEventGroupAddEvent(group, 99) == CUPTI_ERROR_INVALID_EVENT_ID,
          "unknown virtual event IDs are rejected");
    CHECK(cuptiEventGroupAddEvent(group, VGRE_CUPTI_EVENT_KERNEL_LAUNCHES) == CUPTI_SUCCESS &&
          cuptiEventGroupAddEvent(group, VGRE_CUPTI_EVENT_KERNEL_DURATION_NS) == CUPTI_SUCCESS &&
          cuptiEventGroupAddEvent(group, VGRE_CUPTI_EVENT_VIRTUAL_THREADS) == CUPTI_SUCCESS,
          "launch count, measured duration, and virtual-thread events register");
    CHECK(cuptiEventGroupAddEvent(group, VGRE_CUPTI_EVENT_KERNEL_LAUNCHES) ==
              CUPTI_ERROR_INVALID_EVENT_ID,
          "duplicate events are rejected");

    CHECK(cuptiEventGroupEnable(group) == CUPTI_SUCCESS && profiler.isEnabled(),
          "enabling the group captures actual runtime profiling events");
    size_t valueBytes = sizeof(uint64_t) * 3;
    size_t idBytes = sizeof(CUpti_EventID) * 3;
    size_t count = 99;
    uint64_t eventValues[3]{};
    CUpti_EventID eventIds[3]{};
    CHECK(cuptiEventGroupReadAllEvents(group, 0, &valueBytes, eventValues,
                                       &idBytes, eventIds, &count) ==
              CUPTI_ERROR_INVALID_OPERATION,
          "event reads require a disabled group");

    profiler.recordEvent(launch("cupti-first", 1.25, vgre::dim3(2, 3, 4),
                                 vgre::dim3(5, 6, 7)));
    profiler.recordEvent(annotation("non-kernel-annotation", 9.0,
                                   vgre::dim3(99), vgre::dim3(99)));
    profiler.recordEvent(launch("cupti-second", 0.75, vgre::dim3(1, 1, 1),
                                 vgre::dim3(2, 1, 1)));
    CHECK(cuptiEventGroupDisable(group) == CUPTI_SUCCESS && !profiler.isEnabled(),
          "disabling the group closes capture and restores profiler state");

    valueBytes = 0;
    idBytes = 0;
    CHECK(cuptiEventGroupReadAllEvents(group, 0, &valueBytes, nullptr,
                                       &idBytes, nullptr, &count) ==
              CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT &&
          valueBytes == sizeof(uint64_t) * 3 && idBytes == sizeof(CUpti_EventID) * 3,
          "short buffers report exact required capacities without partial writes");
    valueBytes = sizeof(eventValues);
    idBytes = sizeof(eventIds);
    CHECK(cuptiEventGroupReadAllEvents(group, 0, &valueBytes, eventValues,
                                       &idBytes, eventIds, &count) == CUPTI_SUCCESS &&
          count == 3,
          "disabled event group returns all requested counters");
    CHECK(eventValues[0] == 2 && eventValues[1] == 2000000 && eventValues[2] == 5042,
          "counters equal two launches, measured nanoseconds, and launched threads");
    CHECK(eventIds[0] == VGRE_CUPTI_EVENT_KERNEL_LAUNCHES &&
          eventIds[1] == VGRE_CUPTI_EVENT_KERNEL_DURATION_NS &&
          eventIds[2] == VGRE_CUPTI_EVENT_VIRTUAL_THREADS,
          "event values retain their registered ID order");
    valueBytes = sizeof(eventValues);
    idBytes = sizeof(eventIds);
    CHECK(cuptiEventGroupReadAllEvents(group, 0, &valueBytes, eventValues,
                                       &idBytes, eventIds, &count) == CUPTI_SUCCESS &&
          eventValues[0] == 0 && eventValues[1] == 0 && eventValues[2] == 0,
          "a successful event read resets the collected interval");

    CUpti_MetricID metric = 0;
    CHECK(cuptiMetricGetIdFromName(nullptr, "vgre_kernel_launches", &metric) ==
              CUPTI_SUCCESS && metric == VGRE_CUPTI_METRIC_KERNEL_LAUNCHES,
          "virtual metric lookup resolves the launch-count metric");
    CHECK(cuptiMetricGetIdFromName(nullptr, "kernel_duration", &metric) ==
              CUPTI_SUCCESS && metric == VGRE_CUPTI_METRIC_KERNEL_DURATION_NS,
          "legacy kernel-duration lookup resolves the measured virtual metric");
    CHECK(cuptiMetricGetIdFromName(nullptr, "ipc", &metric) == CUPTI_ERROR_NOT_SUPPORTED,
          "physical GPU metrics remain unavailable instead of being fabricated");
    CHECK(cuptiMetricGetIdFromName(nullptr, "achieved_occupancy", &metric) ==
              CUPTI_ERROR_NOT_SUPPORTED &&
          cuptiMetricGetIdFromName(nullptr, "flop_count_sp", &metric) ==
              CUPTI_ERROR_NOT_SUPPORTED &&
          cuptiMetricGetIdFromName(nullptr, "l1_global_load_hit_rate", &metric) ==
              CUPTI_ERROR_NOT_SUPPORTED,
          "legacy physical metric names report unsupported instead of invalid names");
    CHECK(cuptiMetricGetIdFromName(nullptr, "missing_metric", &metric) ==
              CUPTI_ERROR_INVALID_METRIC_NAME,
          "unknown metric names are distinguished from unavailable hardware metrics");
    CUpti_MetricValue metricValue{};
    bool allPhysicalMetricIdsUnsupported = true;
    for (CUpti_MetricID metricId = VGRE_CUPTI_METRIC_IPC;
         metricId <= VGRE_CUPTI_METRIC_BRANCH_EFFICIENCY; ++metricId) {
        allPhysicalMetricIdsUnsupported = allPhysicalMetricIdsUnsupported &&
            cuptiMetricGetValue(nullptr, metricId, 0, nullptr, 0,
                                nullptr, nullptr, 0, &metricValue) ==
                CUPTI_ERROR_NOT_SUPPORTED;
    }
    CHECK(allPhysicalMetricIdsUnsupported,
          "all reserved physical metric IDs report unsupported without estimates");
    eventValues[0] = 2;
    eventValues[1] = 2000000;
    eventValues[2] = 5042;
    CHECK(cuptiMetricGetValue(nullptr, VGRE_CUPTI_METRIC_KERNEL_LAUNCHES,
                              0, nullptr, 3, eventIds, eventValues, 0,
                              &metricValue) == CUPTI_SUCCESS &&
          metricValue.metricValueUint64 == 2,
          "metric evaluation uses the caller-provided launch event value");
    CUpti_EventID averageSpecs[] = {
        VGRE_CUPTI_EVENT_KERNEL_DURATION_NS, VGRE_CUPTI_EVENT_KERNEL_LAUNCHES};
    eventValues[0] = eventValues[1] = eventValues[2] = 0;
    CHECK(cuptiMetricGetValue(nullptr, VGRE_CUPTI_METRIC_AVG_DURATION_NS,
                              2, averageSpecs, 3,
                              eventIds, eventValues, 0, &metricValue) ==
              CUPTI_ERROR_INVALID_EVENT_VALUE,
          "metrics reject incomplete or reset event inputs");
    eventValues[0] = 2;
    eventValues[1] = 2000000;
    eventValues[2] = 5042;
    CHECK(cuptiMetricGetValue(nullptr, VGRE_CUPTI_METRIC_AVG_DURATION_NS,
                              2, averageSpecs, 3,
                              eventIds, eventValues, 0, &metricValue) == CUPTI_SUCCESS &&
          std::fabs(metricValue.metricValueDouble - 1000000.0) < 0.001,
          "average duration is derived from measured time and launch count");
    CHECK(cuptiMetricGetValue(nullptr, VGRE_CUPTI_METRIC_AVG_DURATION_NS,
                              1, averageSpecs, 3,
                              eventIds, eventValues, 0, &metricValue) ==
              CUPTI_ERROR_INVALID_EVENT_VALUE,
          "metric evaluation requires every declared input event");
    CHECK(cuptiEventGroupEnable(group) == CUPTI_SUCCESS,
          "event group can start a second collection interval");
    profiler.recordEvent(launch("cupti-third", 0.5, vgre::dim3(1), vgre::dim3(3)));
    CHECK(cuptiEventGroupDisable(group) == CUPTI_SUCCESS &&
          cuptiEventGroupEnable(group) == CUPTI_SUCCESS,
          "unread event values remain available across collection intervals");
    profiler.recordEvent(launch("cupti-fourth", 0.25, vgre::dim3(1), vgre::dim3(4)));
    CHECK(cuptiEventGroupDisable(group) == CUPTI_SUCCESS,
          "second interval can be disabled");
    valueBytes = sizeof(eventValues);
    idBytes = sizeof(eventIds);
    CHECK(cuptiEventGroupReadAllEvents(group, 0, &valueBytes, eventValues,
                                       &idBytes, eventIds, &count) == CUPTI_SUCCESS &&
          eventValues[0] == 2 && eventValues[1] == 750000 && eventValues[2] == 7,
          "unread counters accumulate exact values across successive intervals");
    CHECK(cuptiMetricGetValue(nullptr, 999, 0, nullptr, 0, nullptr, nullptr, 0,
                              &metricValue) == CUPTI_ERROR_INVALID_METRIC_ID,
          "unknown metric IDs are rejected");
    CHECK(cuptiEventGroupDestroy(group) == CUPTI_SUCCESS,
          "disabled event group can be destroyed");
    CHECK(cuptiEventGroupDestroy(group) == CUPTI_ERROR_INVALID_HANDLE,
          "destroyed handles cannot be reused");

    CHECK(cuptiActivityRegisterCallbacks(requestActivityBuffer,
                                         completeActivityBuffer) == CUPTI_SUCCESS,
          "activity buffers register");
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS,
          "kernel activity enables profiling");
    CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY) == CUPTI_ERROR_NOT_SUPPORTED,
          "unimplemented activity kinds report unsupported");
    auto activityLaunch = launch("activity-regression-kernel", 1.25,
                                 vgre::dim3(2, 3, 4), vgre::dim3(5, 6, 7));
    activityLaunch.staticSharedMemoryBytes = 32;
    activityLaunch.dynamicSharedMemoryBytes = 64;
    profiler.recordEvent(activityLaunch);
    profiler.recordEvent(annotation("activity-annotation", 9.0,
                                    vgre::dim3(99), vgre::dim3(99)));
    testReentrantActivityFlush = true;
    CHECK(cuptiActivityFlushAll(0) == CUPTI_SUCCESS,
          "flush emits recorded launch activity");
    CHECK(reentrantActivityFlushResult == CUPTI_ERROR_INVALID_OPERATION,
          "activity callbacks cannot deadlock by recursively flushing");
    CHECK(activityValidBytes == sizeof(CUpti_ActivityKernel5),
          "flush writes one record for the one recorded launch");

    CUpti_Activity* activity = nullptr;
    CHECK(cuptiActivityGetNextRecord(activityBuffer, activityValidBytes, &activity) ==
              CUPTI_SUCCESS,
          "activity record iteration returns the record");
    if (activity) {
        auto* kernel = reinterpret_cast<CUpti_ActivityKernel5*>(activity);
        CHECK(kernel->kind == CUPTI_ACTIVITY_KIND_KERNEL &&
              kernel->gridX == 2 && kernel->gridY == 3 && kernel->gridZ == 4 &&
              kernel->blockX == 5 && kernel->blockY == 6 && kernel->blockZ == 7,
              "activity preserves launch dimensions");
        CHECK(kernel->end >= kernel->start && kernel->end - kernel->start == 1250000,
              "activity uses measured duration");
        CHECK(kernel->staticSharedMemory == 32 &&
              kernel->dynamicSharedMemory == 64,
              "activity preserves static and dynamic shared-memory launch values");
        CHECK(kernel->registersPerThread == 0 && kernel->aluActivePct == 0.0f &&
              kernel->srcAccessPct == 0.0f,
              "unmeasured hardware fields stay explicitly unavailable");
    }
    CHECK(cuptiActivityGetNextRecord(activityBuffer, activityValidBytes, &activity) ==
              CUPTI_ERROR_QUEUE_EMPTY,
          "activity iteration reports end of buffer");
    CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS &&
          !profiler.isEnabled(),
          "disabling activity restores the prior profiler state");

    profiler.clear();
    callbackCount = 0;
    CUpti_SubscriberHandle subscriber = nullptr;
    CHECK(cuptiSubscribe(&subscriber, launchCallback, &callbackCount) == CUPTI_SUCCESS &&
          subscriber,
          "callback subscription registers a live observer");
    CHECK(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, 1) ==
              CUPTI_ERROR_NOT_SUPPORTED &&
          cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX) ==
              CUPTI_ERROR_NOT_SUPPORTED,
          "unsupported callback domains and IDs report unsupported");
    callbackSubscriber = subscriber;
    attemptSelfUnsubscribe = true;
    CHECK(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                             CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) == CUPTI_SUCCESS,
          "runtime kernel-launch callback enables");
    profiler.recordEvent(launch("callback-kernel", 0.5, vgre::dim3(1), vgre::dim3(8)));
    CHECK(callbackCount == 1 && callbackKernelName == "callback-kernel" &&
          callbackFunctionName == "cudaLaunchKernel" && callbackCorrelationId == 0,
          "callback reports the actual runtime launch entry point and kernel name");
    profiler.recordEvent(annotation("callback-annotation", 0.5,
                                    vgre::dim3(1), vgre::dim3(8)));
    CHECK(callbackCount == 1,
          "non-kernel profile events do not trigger CUDA launch callbacks");
    CHECK(selfUnsubscribeResult == CUPTI_ERROR_INVALID_OPERATION,
          "a callback cannot deadlock by unsubscribing itself synchronously");
    CHECK(cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                              CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) == CUPTI_SUCCESS,
          "runtime launch callback disables");
    profiler.recordEvent(launch("callback-disabled", 0.5, vgre::dim3(1), vgre::dim3(8)));
    CHECK(callbackCount == 1, "disabled callback stops receiving launch records");
    CHECK(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API) == CUPTI_SUCCESS,
          "runtime callback domain enables");
    profiler.recordEvent(launch("callback-domain", 0.5, vgre::dim3(1), vgre::dim3(8)));
    CHECK(callbackCount == 2 && callbackKernelName == "callback-domain",
          "domain callback receives subsequent runtime launches");
    CHECK(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API) == CUPTI_SUCCESS,
          "driver callback domain enables");
    auto driverLaunch = launch("driver-callback", 0.5, vgre::dim3(1), vgre::dim3(8));
    driverLaunch.api = vgre::advanced::ProfileApi::CudaDriver;
    profiler.recordEvent(driverLaunch);
    CHECK(callbackCount == 3 && callbackKernelName == "driver-callback" &&
          callbackFunctionName == "cuLaunchKernel",
          "driver callback reports the actual driver launch entry point");
    CHECK(cuptiUnsubscribe(subscriber) == CUPTI_SUCCESS && !profiler.isEnabled(),
          "unsubscribe removes the observer and restores profiler state");
    CHECK(cuptiUnsubscribe(subscriber) == CUPTI_ERROR_INVALID_HANDLE,
          "stale subscriber handles are rejected");

    auto& engine = vgre::core::RuntimeEngine::instance();
    const auto engineInit = engine.initialize();
    CHECK(engineInit == vgre::VGREResult::SUCCESS,
          "runtime engine initializes for an end-to-end driver launch");
    if (engineInit == vgre::VGREResult::SUCCESS) {
        const std::string liveKernelSource =
            "extern \"C\" __global__ void cupti_live_probe(float* x) { "
            "int i = threadIdx.x; x[i] += 1.0f; }";
        vgre::KernelId liveKernel = 0;
        const auto registration = engine.registerKernel(
            "cupti_live_probe", liveKernelSource, liveKernel);
        CHECK(registration == vgre::VGREResult::SUCCESS && liveKernel != 0,
              "real CUDA kernel source registers through the runtime engine");
        if (registration == vgre::VGREResult::SUCCESS && liveKernel != 0) {
            profiler.clear();
            std::vector<float> output(4, 0.0f);
            void* deviceOutput = output.data();
            void* liveArgs[] = {&deviceOutput};
            CUpti_EventGroupHandle liveGroup = nullptr;
            const bool liveGroupReady =
                cuptiEventGroupCreate(nullptr, &liveGroup, 0) == CUPTI_SUCCESS &&
                cuptiEventGroupAddEvent(liveGroup,
                    VGRE_CUPTI_EVENT_KERNEL_LAUNCHES) == CUPTI_SUCCESS &&
                cuptiEventGroupAddEvent(liveGroup,
                    VGRE_CUPTI_EVENT_KERNEL_DURATION_NS) == CUPTI_SUCCESS &&
                cuptiEventGroupAddEvent(liveGroup,
                    VGRE_CUPTI_EVENT_VIRTUAL_THREADS) == CUPTI_SUCCESS;
            CHECK(liveGroupReady, "live driver launch counters configure");

            callbackCount = 0;
            callbackKernelName.clear();
            callbackFunctionName.clear();
            CUpti_SubscriberHandle liveSubscriber = nullptr;
            const bool liveSubscriberReady =
                cuptiSubscribe(&liveSubscriber, launchCallback,
                              &callbackCount) == CUPTI_SUCCESS &&
                cuptiEnableDomain(1, liveSubscriber,
                                  CUPTI_CB_DOMAIN_DRIVER_API) == CUPTI_SUCCESS &&
                cuptiEnableDomain(1, liveSubscriber,
                                  CUPTI_CB_DOMAIN_RUNTIME_API) == CUPTI_SUCCESS;
            CHECK(liveSubscriberReady,
                  "live driver callback subscribes to actual launch completion");
            callbackSubscriber = liveSubscriber;
            attemptSelfUnsubscribe = true;
            activityValidBytes = 0;
            const bool captureEnabled = liveGroupReady && liveSubscriberReady &&
                cuptiEventGroupEnable(liveGroup) == CUPTI_SUCCESS &&
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS;
            CHECK(captureEnabled, "live counters and kernel activity begin capture");

            int launchResult = -1;
            vgre::VGREResult syncResult = vgre::VGREResult::ERR_UNKNOWN;
            if (captureEnabled) {
                launchResult = cuLaunchKernel(liveKernel, 1, 1, 1, 4, 1, 1,
                                              64, 0, liveArgs, nullptr);
                syncResult = engine.synchronize();
            }
            CHECK(launchResult == 0 && syncResult == vgre::VGREResult::SUCCESS,
                  "public cuLaunchKernel executes and synchronizes a real kernel");
            bool outputMatches = true;
            for (float value : output) outputMatches = outputMatches && value == 1.0f;
            CHECK(outputMatches,
                  "real kernel writes the expected values to caller memory");
            CHECK(callbackCount == 1 &&
                  callbackKernelName == "cupti_live_probe" &&
                  callbackFunctionName == "cuLaunchKernel",
                  "driver callback reports the completed real driver launch");
            CHECK(selfUnsubscribeResult == CUPTI_ERROR_INVALID_OPERATION,
                  "real launch callback self-unsubscribe returns without deadlock");

            const auto runtimeLaunchResult =
                vgre::api::CUDAInterceptor::instance().launchKernel(
                    "cupti_live_probe", liveKernelSource,
                    vgre::dim3(1), vgre::dim3(4), liveArgs);
            const auto runtimeSyncResult = engine.synchronize();
            CHECK(runtimeLaunchResult == vgre::api::cudaSuccess &&
                  runtimeSyncResult == vgre::VGREResult::SUCCESS,
                  "public runtime interceptor executes and synchronizes a real kernel");
            outputMatches = true;
            for (float value : output) outputMatches = outputMatches && value == 2.0f;
            CHECK(outputMatches,
                  "runtime kernel launch updates caller memory a second time");
            CHECK(callbackCount == 2 &&
                  callbackKernelName == "cupti_live_probe" &&
                  callbackFunctionName == "cudaLaunchKernel",
                  "runtime callback reports the completed real runtime launch");

            CHECK(cuptiEventGroupDisable(liveGroup) == CUPTI_SUCCESS,
                  "live driver counter interval stops");
            valueBytes = sizeof(eventValues);
            idBytes = sizeof(eventIds);
            const bool liveCountersRead =
                cuptiEventGroupReadAllEvents(liveGroup, 0, &valueBytes,
                    eventValues, &idBytes, eventIds, &count) == CUPTI_SUCCESS;
            CHECK(liveCountersRead && eventValues[0] == 2 && eventValues[2] == 8,
                  "virtual counters reflect both real launches and their eight threads");
            CHECK(liveCountersRead && eventValues[1] > 0,
                  "real launch counter reports measured nonzero execution time");
            const bool flushSucceeded =
                cuptiActivityFlushAll(0) == CUPTI_SUCCESS &&
                activityValidBytes == 2 * sizeof(CUpti_ActivityKernel5);
            CHECK(flushSucceeded,
                  "activity flush emits the real kernel's launch record");
            CUpti_Activity* liveActivity = nullptr;
            const bool liveRecordRead = flushSucceeded &&
                cuptiActivityGetNextRecord(activityBuffer, activityValidBytes,
                                           &liveActivity) == CUPTI_SUCCESS;
            CUpti_Activity* runtimeActivity = nullptr;
            const bool runtimeRecordRead = liveRecordRead &&
                cuptiActivityGetNextRecord(activityBuffer, activityValidBytes,
                                           &runtimeActivity) == CUPTI_SUCCESS;
            bool liveRecordMatches = false;
            if (liveRecordRead && liveActivity) {
                const auto* kernel = reinterpret_cast<CUpti_ActivityKernel5*>(
                    liveActivity);
                liveRecordMatches = kernel->kind == CUPTI_ACTIVITY_KIND_KERNEL &&
                    kernel->blockX == 4 && kernel->gridX == 1 && kernel->name &&
                    kernel->dynamicSharedMemory == 64 &&
                    std::string(kernel->name) == "cupti_live_probe";
            }
            bool runtimeRecordMatches = false;
            if (runtimeRecordRead && runtimeActivity) {
                const auto* kernel = reinterpret_cast<CUpti_ActivityKernel5*>(
                    runtimeActivity);
                runtimeRecordMatches = kernel->kind == CUPTI_ACTIVITY_KIND_KERNEL &&
                    kernel->blockX == 4 && kernel->gridX == 1 && kernel->name &&
                    kernel->dynamicSharedMemory == 0 &&
                    std::string(kernel->name) == "cupti_live_probe";
            }
            CHECK(liveRecordMatches && runtimeRecordMatches,
                  "activity preserves real names, dimensions, and shared-memory sizes");

            CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL) == CUPTI_SUCCESS,
                  "live kernel activity capture disables");
            CHECK(cuptiUnsubscribe(liveSubscriber) == CUPTI_SUCCESS,
                  "live driver callback unsubscribes cleanly");
            CHECK(cuptiEventGroupDestroy(liveGroup) == CUPTI_SUCCESS,
                  "live driver event group is destroyed");
            CHECK(engine.shutdown() == vgre::VGREResult::SUCCESS,
                  "runtime engine shuts down after the end-to-end launch");
        } else {
            CHECK(engine.shutdown() == vgre::VGREResult::SUCCESS,
                  "runtime engine shuts down after failed kernel registration");
        }
    }

    CHECK(cuptiProfilerInitialize() == CUPTI_SUCCESS && profiler.isEnabled(),
          "profiler initialization acquires a runtime profiling lease");
    CHECK(cuptiProfilerDeInitialize() == CUPTI_SUCCESS && !profiler.isEnabled(),
          "profiler deinitialization releases the lease");
    CHECK(cuptiProfilerDeInitialize() == CUPTI_ERROR_NOT_INITIALIZED,
          "unbalanced profiler deinitialization is rejected");

    std::atomic<bool> profilerLeaseStressSucceeded{true};
    std::atomic<unsigned int> readyThreads{0};
    std::atomic<bool> startThreads{false};
    std::vector<std::thread> profilerLeaseThreads;
    for (unsigned int threadIndex = 0; threadIndex < 8; ++threadIndex) {
        profilerLeaseThreads.emplace_back([&] {
            readyThreads.fetch_add(1, std::memory_order_release);
            while (!startThreads.load(std::memory_order_acquire))
                std::this_thread::yield();
            for (unsigned int iteration = 0; iteration < 250; ++iteration) {
                if (cuptiProfilerInitialize() != CUPTI_SUCCESS ||
                    cuptiProfilerDeInitialize() != CUPTI_SUCCESS) {
                    profilerLeaseStressSucceeded.store(false,
                                                       std::memory_order_release);
                    return;
                }
            }
        });
    }
    while (readyThreads.load(std::memory_order_acquire) != 8)
        std::this_thread::yield();
    startThreads.store(true, std::memory_order_release);
    for (auto& thread : profilerLeaseThreads) thread.join();
    CHECK(profilerLeaseStressSucceeded.load(std::memory_order_acquire) &&
              !profiler.isEnabled(),
          "concurrent profiler init/deinit keeps its lease count balanced");

    std::atomic<bool> subscriberLeaseStressSucceeded{true};
    readyThreads.store(0, std::memory_order_release);
    startThreads.store(false, std::memory_order_release);
    std::vector<std::thread> subscriberLeaseThreads;
    for (unsigned int threadIndex = 0; threadIndex < 8; ++threadIndex) {
        subscriberLeaseThreads.emplace_back([&] {
            readyThreads.fetch_add(1, std::memory_order_release);
            while (!startThreads.load(std::memory_order_acquire))
                std::this_thread::yield();
            for (unsigned int iteration = 0; iteration < 50; ++iteration) {
                CUpti_SubscriberHandle handle = nullptr;
                if (cuptiSubscribe(&handle, launchCallback, nullptr) != CUPTI_SUCCESS ||
                    cuptiUnsubscribe(handle) != CUPTI_SUCCESS) {
                    subscriberLeaseStressSucceeded.store(false,
                                                         std::memory_order_release);
                    return;
                }
            }
        });
    }
    while (readyThreads.load(std::memory_order_acquire) != 8)
        std::this_thread::yield();
    startThreads.store(true, std::memory_order_release);
    for (auto& thread : subscriberLeaseThreads) thread.join();
    CHECK(subscriberLeaseStressSucceeded.load(std::memory_order_acquire) &&
              !profiler.isEnabled(),
          "concurrent subscriber creation and removal keeps profiler leases balanced");
    std::printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
