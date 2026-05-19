// CUPTI shim — VGRE software-proxy implementation of the NVIDIA CUDA
// Profiling Tools Interface (CUPTI).
//
// Hardware performance counters are approximated from the RuntimeProfiler
// instruction-mix data and kernel timing metrics collected during execution.
// All types match the CUPTI 12.x public API so that applications compiled
// against real CUDA/CUPTI headers link and run against VGRE unchanged.

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
    CUPTI_ERROR_NOT_INITIALIZED      = 4,
    CUPTI_ERROR_NOT_SUPPORTED        = 5,
    CUPTI_ERROR_QUEUE_EMPTY          = 6,
    CUPTI_ERROR_OUT_OF_MEMORY        = 7,
} CUptiResult;

// ── Handles ───────────────────────────────────────────────────────────────────
typedef struct CUpti_Subscriber_st* CUpti_SubscriberHandle;
typedef struct CUpti_EventGroup_st* CUpti_EventGroupHandle;

typedef uint32_t CUpti_EventID;
typedef uint32_t CUpti_MetricID;
typedef void*    CUcontext;
typedef void*    CUdevice;

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
} CUpti_ApiCallbackSite;

typedef struct {
    uint32_t                 size;
    CUpti_CallbackDomain     callbackSite; // ENTER or EXIT (stored here for layout compat)
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
    uint32_t           deviceId;
    uint32_t           contextId;
    uint32_t           streamId;
    int32_t            gridX, gridY, gridZ;
    int32_t            blockX, blockY, blockZ;
    int32_t            staticSharedMemory;
    int32_t            dynamicSharedMemory;
    uint64_t           start;          // nanoseconds (steady_clock)
    uint64_t           end;
    const char*        name;           // kernel name (pointer into static storage)
    uint64_t           completed;
    uint32_t           registersPerThread;
    float              srcAccessPct;   // estimated memory access (software proxy)
    float              aluActivePct;   // estimated ALU utilization (software proxy)
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

// ── Known software-proxy metric IDs ──────────────────────────────────────────
// IDs < 0x1000 are VGRE virtual metrics mapped from RuntimeProfiler data.
#define CUPTI_METRIC_ID_IPC                    0x01  // instructions per cycle (proxy)
#define CUPTI_METRIC_ID_ACHIEVED_OCCUPANCY     0x02  // achieved occupancy (proxy)
#define CUPTI_METRIC_ID_FLOP_COUNT_SP          0x03  // single-precision FLOPs
#define CUPTI_METRIC_ID_DRAM_READ_THROUGHPUT   0x04  // estimated read GB/s
#define CUPTI_METRIC_ID_DRAM_WRITE_THROUGHPUT  0x05  // estimated write GB/s
#define CUPTI_METRIC_ID_L1_GLOBAL_LOAD_HIT     0x06  // L1 load hit rate (proxy)
#define CUPTI_METRIC_ID_BRANCH_EFFICIENCY      0x07  // branch efficiency (proxy)
#define CUPTI_METRIC_ID_KERNEL_DURATION_NS     0x08  // kernel duration in ns

// ── Subscriber API ────────────────────────────────────────────────────────────
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

// ── Event group API (stub — metrics only) ────────────────────────────────────
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
