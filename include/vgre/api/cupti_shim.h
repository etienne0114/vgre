// VGRE's virtual profiling interface.
//
// This header is not ABI-compatible with NVIDIA CUPTI. It declares a VGRE-owned
// subset of similarly named entry points for the software runtime. Its event
// groups expose measured kernel launches, elapsed time, and virtual thread counts.
// Physical GPU PMU metrics are not available on the CPU-backed virtual device.

#ifndef VGRE_CUPTI_SHIM_H
#define VGRE_CUPTI_SHIM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Error codes ───────────────────────────────────────────────────────────────
typedef enum {
    CUPTI_SUCCESS                    = 0,
    CUPTI_ERROR_INVALID_PARAMETER    = 1,
    CUPTI_ERROR_INVALID_DEVICE       = 2,
    CUPTI_ERROR_INVALID_CONTEXT      = 3,
    CUPTI_ERROR_INVALID_EVENT_ID     = 5,
    CUPTI_ERROR_INVALID_OPERATION    = 7,
    CUPTI_ERROR_OUT_OF_MEMORY        = 8,
    CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT = 10,
    CUPTI_ERROR_NOT_READY            = 13,
    CUPTI_ERROR_NOT_COMPATIBLE       = 14,
    CUPTI_ERROR_NOT_INITIALIZED      = 15,
    CUPTI_ERROR_INVALID_METRIC_ID    = 16,
    CUPTI_ERROR_INVALID_METRIC_NAME  = 17,
    CUPTI_ERROR_QUEUE_EMPTY          = 18,
    CUPTI_ERROR_INVALID_HANDLE       = 19,
    CUPTI_ERROR_INVALID_KIND         = 21,
    CUPTI_ERROR_INVALID_EVENT_VALUE  = 22,
    CUPTI_ERROR_NOT_SUPPORTED        = 27,
    CUPTI_ERROR_UNKNOWN              = 999,
} CUptiResult;

// ── Handles ───────────────────────────────────────────────────────────────────
typedef struct CUpti_Subscriber_st* CUpti_SubscriberHandle;
typedef struct CUpti_EventGroup_st* CUpti_EventGroupHandle;

typedef uint32_t CUpti_EventID;
typedef uint32_t CUpti_MetricID;
typedef void*    CUcontext;
typedef void*    CUdevice;

// VGRE virtual event IDs. Values come from completed RuntimeProfiler launch
// records and reset after a successful disabled-group read.
#define VGRE_CUPTI_EVENT_KERNEL_LAUNCHES       ((CUpti_EventID)1)
#define VGRE_CUPTI_EVENT_KERNEL_DURATION_NS    ((CUpti_EventID)2)
#define VGRE_CUPTI_EVENT_VIRTUAL_THREADS       ((CUpti_EventID)3)

// Metric IDs 1–7 remain reserved for physical GPU counters from older VGRE
// headers. They are never calculated from host estimates.
#define VGRE_CUPTI_METRIC_IPC                  ((CUpti_MetricID)1)
#define VGRE_CUPTI_METRIC_OCCUPANCY            ((CUpti_MetricID)2)
#define VGRE_CUPTI_METRIC_FLOP_COUNT           ((CUpti_MetricID)3)
#define VGRE_CUPTI_METRIC_DRAM_READ_BYTES      ((CUpti_MetricID)4)
#define VGRE_CUPTI_METRIC_DRAM_WRITE_BYTES     ((CUpti_MetricID)5)
#define VGRE_CUPTI_METRIC_L1_HIT_RATE          ((CUpti_MetricID)6)
#define VGRE_CUPTI_METRIC_BRANCH_EFFICIENCY    ((CUpti_MetricID)7)
#define VGRE_CUPTI_METRIC_KERNEL_DURATION_NS   ((CUpti_MetricID)8)
#define VGRE_CUPTI_METRIC_KERNEL_LAUNCHES      ((CUpti_MetricID)9)
#define VGRE_CUPTI_METRIC_VIRTUAL_THREADS      ((CUpti_MetricID)10)
#define VGRE_CUPTI_METRIC_AVG_DURATION_NS      ((CUpti_MetricID)11)

// Source-compatibility aliases from the original VGRE header. IDs 1–7 are
// reserved and return NOT_SUPPORTED; ID 8 is measured virtual duration.
#define CUPTI_METRIC_ID_IPC                    VGRE_CUPTI_METRIC_IPC
#define CUPTI_METRIC_ID_ACHIEVED_OCCUPANCY     VGRE_CUPTI_METRIC_OCCUPANCY
#define CUPTI_METRIC_ID_FLOP_COUNT_SP          VGRE_CUPTI_METRIC_FLOP_COUNT
#define CUPTI_METRIC_ID_DRAM_READ_THROUGHPUT   VGRE_CUPTI_METRIC_DRAM_READ_BYTES
#define CUPTI_METRIC_ID_DRAM_WRITE_THROUGHPUT  VGRE_CUPTI_METRIC_DRAM_WRITE_BYTES
#define CUPTI_METRIC_ID_L1_GLOBAL_LOAD_HIT     VGRE_CUPTI_METRIC_L1_HIT_RATE
#define CUPTI_METRIC_ID_BRANCH_EFFICIENCY      VGRE_CUPTI_METRIC_BRANCH_EFFICIENCY
#define CUPTI_METRIC_ID_KERNEL_DURATION_NS     VGRE_CUPTI_METRIC_KERNEL_DURATION_NS

#define CUPTI_ACTIVITY_FLAG_FLUSH_FORCED       1u

// ── Callback domain / ID ─────────────────────────────────────────────────────
typedef enum {
    CUPTI_CB_DOMAIN_INVALID         = 0,
    CUPTI_CB_DOMAIN_DRIVER_API      = 1,
    CUPTI_CB_DOMAIN_RUNTIME_API     = 2,
    CUPTI_CB_DOMAIN_RESOURCE        = 3,
    CUPTI_CB_DOMAIN_SYNCHRONIZE     = 4,
    CUPTI_CB_DOMAIN_NVTX            = 5,
} CUpti_CallbackDomain;

typedef uint32_t CUpti_CallbackId;

// Common driver/runtime callback IDs for kernel launches
#define CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel         33
#define CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020      1
#define CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 160

typedef enum {
    CUPTI_API_ENTER = 0,
    CUPTI_API_EXIT  = 1,
    VGRE_CUPTI_API_COMPLETION = 2,
} CUpti_ApiCallbackSite;

typedef struct {
    uint32_t                 size;
    CUpti_ApiCallbackSite    callbackSite;
    CUcontext                context;
    CUdevice                 deviceId;
    const char*              functionName;
    const void*              functionParams;  // points to kernel params struct
    void*                    functionReturnValue;
    const char*              symbolName;
    void*                    correlationData;
    uint32_t                 correlationId;
} CUpti_CallbackData;

typedef void (*CUpti_CallbackFunc)(void* userdata,
                                   CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid,
                                   const CUpti_CallbackData* cbInfo);

// ── Activity records ──────────────────────────────────────────────────────────
typedef enum {
    CUPTI_ACTIVITY_KIND_INVALID      = 0,
    CUPTI_ACTIVITY_KIND_MEMCPY       = 1,
    CUPTI_ACTIVITY_KIND_MEMSET       = 2,
    CUPTI_ACTIVITY_KIND_KERNEL       = 3,
    CUPTI_ACTIVITY_KIND_DRIVER       = 4,
    CUPTI_ACTIVITY_KIND_RUNTIME      = 5,
    CUPTI_ACTIVITY_KIND_NAME         = 6,
    CUPTI_ACTIVITY_KIND_MARKER       = 7,
    CUPTI_ACTIVITY_KIND_SYNCHRONIZE  = 8,
} CUpti_ActivityKind;

typedef struct {
    CUpti_ActivityKind kind;           // CUPTI_ACTIVITY_KIND_KERNEL
    uint32_t           correlationId;
    uint32_t           deviceId;       // zero: VGRE does not track this ID
    uint32_t           contextId;      // zero: VGRE does not track this ID
    uint32_t           streamId;       // zero: VGRE does not track this ID
    uint32_t           gridX, gridY, gridZ;
    uint32_t           blockX, blockY, blockZ;
    int32_t            staticSharedMemory;  // launch metadata when reported
    int32_t            dynamicSharedMemory; // launch metadata when reported
    uint64_t           start;          // sign-biased steady_clock nanoseconds
    uint64_t           end;
    const char*        name;           // kernel name (pointer into static storage)
    uint64_t           completed;
    uint32_t           registersPerThread; // zero: unavailable
    float              srcAccessPct;   // zero: hardware utilization is unavailable
    float              aluActivePct;   // zero: hardware utilization is unavailable
} CUpti_ActivityKernel5;

typedef struct {
    CUpti_ActivityKind kind;           // generic header for casting
    uint32_t           correlationId;
} CUpti_Activity;

// ── Metric value ──────────────────────────────────────────────────────────────
typedef enum {
    CUPTI_METRIC_VALUE_KIND_DOUBLE        = 0,
    CUPTI_METRIC_VALUE_KIND_UINT64        = 1,
    CUPTI_METRIC_VALUE_KIND_PERCENT       = 2,
    CUPTI_METRIC_VALUE_KIND_THROUGHPUT    = 3,
    CUPTI_METRIC_VALUE_KIND_INT64         = 4,
    CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL = 5,
} CUpti_MetricValueKind;

typedef union {
    double   metricValueDouble;
    uint64_t metricValueUint64;
    int64_t  metricValueInt64;
    double   metricValuePercent;
    double   metricValueThroughput;
    int      metricValueUtilizationLevel; // 0-10
} CUpti_MetricValue;

// ── Subscriber API ────────────────────────────────────────────────────────────
// VGRE dispatches supported cudaLaunchKernel/cuLaunchKernel notifications after
// completed work is recorded with callbackSite=VGRE_CUPTI_API_COMPLETION.
// Callbacks run on the execution worker; context,
// device, raw parameter structures, and cross-tool correlation IDs are
// unavailable and null/zero. Unsubscribe waits for in-flight calls and returns
// INVALID_OPERATION from that callback. Other callback domains and IDs return
// NOT_SUPPORTED.
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                           CUpti_CallbackFunc callback,
                           void* userdata);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid);
CUptiResult cuptiEnableDomain(uint32_t enable,
                              CUpti_SubscriberHandle subscriber,
                              CUpti_CallbackDomain domain);

// ── Activity API ──────────────────────────────────────────────────────────────
// VGRE records kernel launches only. Other activity kinds return NOT_SUPPORTED
// because copies, synchronization, and other activity records are not tracked.
// Shared-memory values use launch metadata where available and otherwise zero.
// While either buffer callback runs, enable/disable/flush calls return
// INVALID_OPERATION so callback code cannot deadlock by recursively flushing.
CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind);
CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind);

typedef void (*CUpti_BuffersCallbackRequestFunc)(uint8_t** buffer,
                                                  size_t* size,
                                                  size_t* maxNumRecords);
typedef void (*CUpti_BuffersCallbackCompleteFunc)(CUcontext ctx,
                                                   uint32_t streamId,
                                                   uint8_t* buffer,
                                                   size_t size,
                                                   size_t validSize);

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);

CUptiResult cuptiActivityFlushAll(uint32_t flag);

CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity** record);

// ── Timestamp ────────────────────────────────────────────────────────────────
CUptiResult cuptiGetTimestamp(uint64_t* timestamp);

// ── Metric API ───────────────────────────────────────────────────────────────
CUptiResult cuptiMetricGetIdFromName(CUdevice device, const char* metricName,
                                     CUpti_MetricID* metric);
CUptiResult cuptiMetricGetValue(CUdevice device,
                                CUpti_MetricID metric,
                                uint32_t numEventSpecs,
                                void* eventSpecArray,
                                uint32_t numEvents,
                                CUpti_EventID* eventIdArray,
                                uint64_t* eventValueArray,
                                uint64_t timeDuration,
                                CUpti_MetricValue* metricValue);

// `eventSpecArray`, when supplied, is an array of CUpti_EventID values that
// declares the event inputs expected by the caller. Metric values are computed
// from `eventIdArray` / `eventValueArray`, which must contain required event IDs.
// `timeDuration` is reserved and must be zero for these cumulative metrics.

// ── VGRE virtual event group API ─────────────────────────────────────────────
// Event groups aggregate the process-wide default virtual context. VGRE cannot
// filter by opaque CUDA contexts, so non-null contexts return INVALID_CONTEXT.
CUptiResult cuptiEventGroupCreate(CUcontext context,
                                  CUpti_EventGroupHandle* eventGroup,
                                  uint32_t flags);
CUptiResult cuptiEventGroupDestroy(CUpti_EventGroupHandle eventGroup);
CUptiResult cuptiEventGroupAddEvent(CUpti_EventGroupHandle eventGroup,
                                    CUpti_EventID event);
CUptiResult cuptiEventGroupEnable(CUpti_EventGroupHandle eventGroup);
CUptiResult cuptiEventGroupDisable(CUpti_EventGroupHandle eventGroup);
CUptiResult cuptiEventGroupReadAllEvents(CUpti_EventGroupHandle eventGroup,
                                         uint32_t flags,
                                         size_t* eventValueBufferSizeBytes,
                                         uint64_t* eventValueBuffer,
                                         size_t* eventIdArraySizeBytes,
                                         CUpti_EventID* eventIdArray,
                                         size_t* numEventIdsRead);

// ── Profiler context API ──────────────────────────────────────────────────────
CUptiResult cuptiProfilerInitialize(void);
CUptiResult cuptiProfilerDeInitialize(void);

#ifdef __cplusplus
}
#endif

#endif // VGRE_CUPTI_SHIM_H
