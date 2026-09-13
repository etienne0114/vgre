// VGRE HIP/ROCm compatibility layer — docs/missingFeatures.md §7.2.
//
// AMD HIP is a near-1:1 mirror of the CUDA runtime/driver API, so HIP programs
// run on VGRE by mapping hip* → VGRE's cuda*/cu* implementations (all of which
// are real, tested code paths — this layer adds no emulation logic of its own).
//
// This is a SOURCE-level compatibility header: HIP host code is compiled
// against it and linked with libvgre. Handle types are opaque and carry VGRE's
// internal ids; hipError_t values are numerically identical to CUDA runtime
// error codes (hipSuccess=0, hipErrorInvalidValue=1, hipErrorOutOfMemory=2, …),
// which HIP itself also guarantees for these basics.
//
// Device code path: load HIP/CUDA kernel source (or PTX) with
// hipModuleLoadData(), resolve with hipModuleGetFunction(), and launch with
// hipModuleLaunchKernel() — the source is JIT-compiled by VGRE's LLVM pipeline
// exactly like a CUDA kernel (HIP device syntax is CUDA device syntax).
#ifndef VGRE_HIP_HIP_RUNTIME_H
#define VGRE_HIP_HIP_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Error codes (numerically identical to CUDA runtime error codes) ─────────
typedef int hipError_t;
enum { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2,
       hipErrorNotInitialized = 3, hipErrorDeinitialized = 4,
       hipErrorInvalidDevice = 10, hipErrorInvalidResourceHandle = 400,
       hipErrorNotSupported = 801 };

// ── Opaque handle types ──────────────────────────────────────────────────────
// Pointer-typed so `hipStream_t s = nullptr` (the HIP idiom for the default
// stream) compiles; internally the value carries VGRE's integer StreamId.
typedef struct ihipStream_t*  hipStream_t;
typedef struct ihipEvent_t*   hipEvent_t;
typedef struct ihipModule_t*  hipModule_t;
typedef struct ihipFunction_t* hipFunction_t;

// ── Memcpy kinds (share CUDA's numeric values) ──────────────────────────────
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

// ── Flags ────────────────────────────────────────────────────────────────────
#define hipStreamDefault      0x00u
#define hipStreamNonBlocking  0x01u
#define hipEventDefault       0x00u
#define hipEventBlockingSync  0x01u
#define hipEventDisableTiming 0x02u
#define hipHostMallocDefault  0x00u
#define hipHostMallocPortable 0x01u
#define hipHostMallocMapped   0x02u
#define hipMemAttachGlobal    0x01u

// ── Device attributes ────────────────────────────────────────────────────────
// Source-compatible names; values chosen to match VGRE's CUDA attribute ids so
// the query delegates directly to cudaDeviceGetAttribute.
typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock            = 1,
    hipDeviceAttributeMaxBlockDimX                  = 2,
    hipDeviceAttributeMaxBlockDimY                  = 3,
    hipDeviceAttributeMaxBlockDimZ                  = 4,
    hipDeviceAttributeMaxGridDimX                   = 5,
    hipDeviceAttributeMaxGridDimY                   = 6,
    hipDeviceAttributeMaxGridDimZ                   = 7,
    hipDeviceAttributeMaxSharedMemoryPerBlock       = 8,
    hipDeviceAttributeTotalConstantMemory           = 9,
    hipDeviceAttributeWarpSize                      = 10,
    hipDeviceAttributeMaxRegistersPerBlock          = 12,
    hipDeviceAttributeClockRate                     = 13,
    hipDeviceAttributeMultiprocessorCount           = 16,
    hipDeviceAttributeIntegrated                    = 18,
    hipDeviceAttributeCanMapHostMemory              = 19,
    hipDeviceAttributeComputeMode                   = 20,
    hipDeviceAttributeConcurrentKernels             = 31,
    hipDeviceAttributeMemoryClockRate               = 36,
    hipDeviceAttributeMemoryBusWidth                = 37,
    hipDeviceAttributeL2CacheSize                   = 38,
    hipDeviceAttributeMaxThreadsPerMultiProcessor   = 39,
    hipDeviceAttributeComputeCapabilityMajor        = 75,
    hipDeviceAttributeComputeCapabilityMinor        = 76,
    hipDeviceAttributeManagedMemory                 = 83,
    hipDeviceAttributeCooperativeLaunch             = 95
} hipDeviceAttribute_t;

// ── Device properties ────────────────────────────────────────────────────────
typedef struct hipDeviceProp_t {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    warpSize;
    int    multiProcessorCount;
    int    major;
    int    minor;
    int    clockRate;
    size_t totalConstMem;
    int    maxThreadsPerMultiProcessor;
    int    l2CacheSize;
    int    memoryClockRate;
    int    memoryBusWidth;
    int    integrated;
    int    canMapHostMemory;
    int    concurrentKernels;
    int    computeMode;
    int    pciBusID;
    int    pciDeviceID;
    char   gcnArchName[256];   // HIP-specific; VGRE reports its emulated arch
} hipDeviceProp_t;

// ── Error handling ───────────────────────────────────────────────────────────
hipError_t  hipGetLastError(void);
hipError_t  hipPeekAtLastError(void);
const char* hipGetErrorString(hipError_t err);
const char* hipGetErrorName(hipError_t err);

// ── Device management ────────────────────────────────────────────────────────
hipError_t hipGetDeviceCount(int* count);
hipError_t hipSetDevice(int deviceId);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceReset(void);
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int deviceId);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipRuntimeGetVersion(int* runtimeVersion);

// ── Memory management ────────────────────────────────────────────────────────
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyHtoD(void* dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoH(void* dst, void* src, size_t sizeBytes);
hipError_t hipMemcpyDtoD(void* dst, void* src, size_t sizeBytes);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemGetInfo(size_t* freeBytes, size_t* totalBytes);

// ── Stream management ────────────────────────────────────────────────────────
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamQuery(hipStream_t stream);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);

// ── Event management ─────────────────────────────────────────────────────────
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
hipError_t hipEventDestroy(hipEvent_t event);

// ── Module / kernel launch (driver-style; the VGRE-native HIP path) ─────────
// image: HIP/CUDA C++ kernel source, PTX text, or an LLVM-bitcode/fatbinary
// blob — anything VGRE's cuModuleLoadData accepts.
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);

#ifdef __cplusplus
}
#endif

#endif // VGRE_HIP_HIP_RUNTIME_H
