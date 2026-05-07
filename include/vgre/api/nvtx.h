#pragma once
// NVTX v3 shim for VGRE.
// Provides the full NVIDIA Tools Extension (NVTX) v3 API so that
// applications instrumented with NVTX (PyTorch, TensorFlow, etc.) link and
// run correctly against VGRE.  All ranges are forwarded to RuntimeProfiler
// for correlation with kernel traces and OTLP export.

#include <stdint.h>
#include "vgre/common/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Handle types ──────────────────────────────────────────────────────────────
typedef uint64_t nvtxRangeId_t;
typedef void*    nvtxDomainHandle_t;
typedef void*    nvtxStringHandle_t;

// ── Color / category ─────────────────────────────────────────────────────────
#define NVTX_COLOR_ARGB         0x00000001u
#define NVTX_COLOR_UNKNOWN      0x00000000u
#define NVTX_PAYLOAD_UNKNOWN    0x00000000u
#define NVTX_PAYLOAD_TYPE_INT64 0x00000001u
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 0x00000002u
#define NVTX_PAYLOAD_TYPE_DOUBLE        0x00000003u

// ── Event attributes ─────────────────────────────────────────────────────────
typedef struct nvtxEventAttributes_v2 {
    uint16_t version;            // must be NVTX_VERSION
    uint16_t size;               // sizeof(nvtxEventAttributes_t)
    uint32_t category;
    int32_t  colorType;
    uint32_t color;
    int32_t  payloadType;
    int32_t  reserved0;
    union {
        uint64_t ullValue;
        int64_t  llValue;
        double   dValue;
        float    fValue;
    } payload;
    int32_t  messageType;
    union {
        const char*          ascii;
        const wchar_t*       unicode;
        nvtxStringHandle_t   registered;
    } message;
} nvtxEventAttributes_v2;
typedef nvtxEventAttributes_v2 nvtxEventAttributes_t;
#define NVTX_VERSION 3
#define NVTX_MESSAGE_TYPE_ASCII   1
#define NVTX_MESSAGE_TYPE_UNICODE 2
#define NVTX_MESSAGE_TYPE_REGISTERED 3

// ── Core marker / range APIs ─────────────────────────────────────────────────
VGRE_PUBLIC_API void            nvtxMarkA(const char* message);
VGRE_PUBLIC_API void            nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API nvtxRangeId_t   nvtxRangeStartA(const char* message);
VGRE_PUBLIC_API nvtxRangeId_t   nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API void            nvtxRangeEnd(nvtxRangeId_t id);
VGRE_PUBLIC_API int             nvtxRangePushA(const char* message);
VGRE_PUBLIC_API int             nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API int             nvtxRangePop(void);

// ── Domain APIs (isolated profiling namespaces) ──────────────────────────────
VGRE_PUBLIC_API nvtxDomainHandle_t nvtxDomainCreateA(const char* name);
VGRE_PUBLIC_API nvtxDomainHandle_t nvtxDomainCreateW(const wchar_t* name);
VGRE_PUBLIC_API nvtxDomainHandle_t nvtxDomainCreate(const char* name);
VGRE_PUBLIC_API void               nvtxDomainDestroy(nvtxDomainHandle_t domain);
VGRE_PUBLIC_API void               nvtxDomainMarkEx(nvtxDomainHandle_t domain,
                                                     const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API nvtxRangeId_t      nvtxDomainRangeStartEx(nvtxDomainHandle_t domain,
                                                           const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API void               nvtxDomainRangeEnd(nvtxDomainHandle_t domain,
                                                       nvtxRangeId_t id);
VGRE_PUBLIC_API int                nvtxDomainRangePushEx(nvtxDomainHandle_t domain,
                                                          const nvtxEventAttributes_t* eventAttrib);
VGRE_PUBLIC_API int                nvtxDomainRangePop(nvtxDomainHandle_t domain);

// ── String registration ───────────────────────────────────────────────────────
VGRE_PUBLIC_API nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t domain,
                                                              const char* string);
VGRE_PUBLIC_API nvtxStringHandle_t nvtxDomainRegisterStringW(nvtxDomainHandle_t domain,
                                                              const wchar_t* string);

// ── Resource naming ───────────────────────────────────────────────────────────
VGRE_PUBLIC_API void nvtxNameCategoryA(uint32_t category, const char* name);
VGRE_PUBLIC_API void nvtxNameOsThreadA(uint32_t tid, const char* name);

// CUDA-specific resource naming (links NVTX ranges to CUDA streams/devices)
typedef void* cudaStream_t_nvtx;
VGRE_PUBLIC_API void nvtxNameCudaDeviceA(int device, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaStreamA(cudaStream_t_nvtx stream, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaEventA(void* event, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaDeviceEx(nvtxDomainHandle_t domain, int device, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaStreamEx(nvtxDomainHandle_t domain, cudaStream_t_nvtx stream, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaEventEx(nvtxDomainHandle_t domain, void* event, const char* name);
VGRE_PUBLIC_API void nvtxNameCudaThreadEx(nvtxDomainHandle_t domain, uint32_t tid, const char* name);

#ifdef __cplusplus
} // extern "C"
#endif
