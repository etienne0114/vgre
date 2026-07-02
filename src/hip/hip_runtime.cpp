// VGRE HIP/ROCm compatibility layer (impl).  See include/vgre/hip/hip_runtime.h.
//
// Every hip* entry delegates to the corresponding VGRE cuda*/cu* symbol — all
// real, tested implementations; this file adds no emulation logic of its own.
// cudaError_t is `int`, hipMemcpyKind mirrors cudaMemcpyKind numerically, and
// hipError_t values are the CUDA runtime error codes, so status passthrough is
// direct.  Handle marshalling: VGRE's cudaStream_t is an integer StreamId and
// CUfunction is an integer KernelId — both are carried inside the opaque HIP
// pointer types; cudaEvent_t and CUmodule are pointers and pass through as-is.

#include "vgre/hip/hip_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

// ── VGRE's CUDA runtime/driver exports (ABI-exact prototypes) ───────────────
extern "C" {
// cudaError_t == int; cudaStream_t == uint64_t; cudaEvent_t == void*.
int         cudaMalloc(void** devPtr, size_t size);
int         cudaFree(void* devPtr);
int         cudaMallocManaged(void** devPtr, size_t size, unsigned int flags);
int         cudaMallocAsync(void** devPtr, size_t size, uint64_t stream);
int         cudaFreeAsync(void* devPtr, uint64_t stream);
int         cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
int         cudaFreeHost(void* ptr);
int         cudaMemcpy(void* dst, const void* src, size_t count, int kind);
int         cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, uint64_t stream);
int         cudaMemset(void* devPtr, int value, size_t count);
int         cudaMemsetAsync(void* devPtr, int value, size_t count, uint64_t stream);
int         cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes);
int         cudaDeviceSynchronize(void);
int         cudaDeviceReset(void);
int         cudaGetDeviceCount(int* count);
int         cudaSetDevice(int device);
int         cudaGetDevice(int* device);
int         cudaDeviceGetAttribute(int* value, int attr, int device);
int         cudaDriverGetVersion(int* version);
int         cudaRuntimeGetVersion(int* version);
int         cudaGetLastError(void);
int         cudaPeekAtLastError(void);
const char* cudaGetErrorString(int err);
const char* cudaGetErrorName(int err);

int         cudaStreamCreate(uint64_t* pStream);
int         cudaStreamCreateWithFlags(uint64_t* pStream, unsigned int flags);
int         cudaStreamCreateWithPriority(uint64_t* pStream, unsigned int flags, int priority);
int         cudaStreamDestroy(uint64_t stream);
int         cudaStreamSynchronize(uint64_t stream);
int         cudaStreamQuery(uint64_t stream);
int         cudaStreamWaitEvent(uint64_t stream, void* event, unsigned int flags);

int         cudaEventCreate(void** event);
int         cudaEventCreateWithFlags(void** event, unsigned int flags);
int         cudaEventRecord(void* event, uint64_t stream);
int         cudaEventSynchronize(void* event);
int         cudaEventQuery(void* event);
int         cudaEventElapsedTime(float* ms, void* start, void* end);
int         cudaEventDestroy(void* event);

// VGRE's internal cudaDeviceProp_t layout (include/vgre/api/cuda_interceptor.h).
struct vgre_cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int warpSize;
    int multiProcessorCount;
    int major;
    int minor;
    int clockRate;
    size_t totalConstMem;
};
int         cudaGetDeviceProperties(vgre_cudaDeviceProp* prop, int device);

// Driver API (CUresult == int; CUmodule == void*; CUfunction == uint64_t;
// CUstream == uint64_t).
int         cuModuleLoad(void** module, const char* fname);
int         cuModuleLoadData(void** module, const void* image);
int         cuModuleGetFunction(uint64_t* hfunc, void* hmod, const char* name);
int         cuModuleUnload(void* hmod);
int         cuLaunchKernel(uint64_t f,
                           unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                           unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                           unsigned int sharedMemBytes, uint64_t hStream,
                           void** kernelParams, void** extra);
} // extern "C"

namespace {
// hipStream_t ⇄ VGRE StreamId (integer carried in the opaque pointer value;
// nullptr ⇒ 0 ⇒ the default stream, matching the HIP idiom).
inline uint64_t    toStreamId(hipStream_t s) { return reinterpret_cast<uint64_t>(s); }
inline hipStream_t toHipStream(uint64_t id)  { return reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(id)); }
// CUDA driver-error → runtime-error normalisation for the module path: VGRE's
// cu* entry points return CUresult codes whose basics coincide with the
// runtime codes used by hipError_t (0 success, 1 invalid value, 2 OOM).
inline hipError_t fromCu(int r) { return r; }
} // namespace

extern "C" {

// ── Error handling ───────────────────────────────────────────────────────────
hipError_t  hipGetLastError(void)             { return cudaGetLastError(); }
hipError_t  hipPeekAtLastError(void)          { return cudaPeekAtLastError(); }
const char* hipGetErrorString(hipError_t err) { return cudaGetErrorString(err); }
const char* hipGetErrorName(hipError_t err)   { return cudaGetErrorName(err); }

// ── Device management ────────────────────────────────────────────────────────
hipError_t hipGetDeviceCount(int* count)     { return cudaGetDeviceCount(count); }
hipError_t hipSetDevice(int deviceId)        { return cudaSetDevice(deviceId); }
hipError_t hipGetDevice(int* deviceId)       { return cudaGetDevice(deviceId); }
hipError_t hipDeviceSynchronize(void)        { return cudaDeviceSynchronize(); }
hipError_t hipDeviceReset(void)              { return cudaDeviceReset(); }
hipError_t hipDriverGetVersion(int* v)       { return cudaDriverGetVersion(v); }
hipError_t hipRuntimeGetVersion(int* v)      { return cudaRuntimeGetVersion(v); }

hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int deviceId) {
    // hipDeviceAttribute_t values are defined to coincide with VGRE's CUDA
    // attribute ids (see the header), so this is a direct delegation.
    return cudaDeviceGetAttribute(value, static_cast<int>(attr), deviceId);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    if (!prop) return hipErrorInvalidValue;
    vgre_cudaDeviceProp cp;
    std::memset(&cp, 0, sizeof(cp));
    if (int r = cudaGetDeviceProperties(&cp, deviceId); r != 0) return r;

    std::memset(prop, 0, sizeof(*prop));
    std::memcpy(prop->name, cp.name, sizeof(prop->name));
    prop->totalGlobalMem      = cp.totalGlobalMem;
    prop->sharedMemPerBlock   = cp.sharedMemPerBlock;
    prop->maxThreadsPerBlock  = cp.maxThreadsPerBlock;
    for (int i = 0; i < 3; ++i) {
        prop->maxThreadsDim[i] = cp.maxThreadsDim[i];
        prop->maxGridSize[i]   = cp.maxGridSize[i];
    }
    prop->warpSize            = cp.warpSize;
    prop->multiProcessorCount = cp.multiProcessorCount;
    prop->major               = cp.major;
    prop->minor               = cp.minor;
    prop->clockRate           = cp.clockRate;
    prop->totalConstMem       = cp.totalConstMem;

    // Fields beyond VGRE's compact prop struct come from the attribute path
    // (each backed by the same real device model; unknown attrs read as 0).
    cudaDeviceGetAttribute(&prop->maxThreadsPerMultiProcessor,
                           hipDeviceAttributeMaxThreadsPerMultiProcessor, deviceId);
    cudaDeviceGetAttribute(&prop->l2CacheSize, hipDeviceAttributeL2CacheSize, deviceId);
    cudaDeviceGetAttribute(&prop->memoryClockRate, hipDeviceAttributeMemoryClockRate, deviceId);
    cudaDeviceGetAttribute(&prop->memoryBusWidth, hipDeviceAttributeMemoryBusWidth, deviceId);
    cudaDeviceGetAttribute(&prop->integrated, hipDeviceAttributeIntegrated, deviceId);
    cudaDeviceGetAttribute(&prop->canMapHostMemory, hipDeviceAttributeCanMapHostMemory, deviceId);
    cudaDeviceGetAttribute(&prop->concurrentKernels, hipDeviceAttributeConcurrentKernels, deviceId);
    cudaDeviceGetAttribute(&prop->computeMode, hipDeviceAttributeComputeMode, deviceId);

    // VGRE is a CPU-backed emulated device; report an honest arch string.
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName),
                  "vgre-cpu-emulated (sm_%d%d equivalent)", cp.major, cp.minor);
    return hipSuccess;
}

// ── Memory management ────────────────────────────────────────────────────────
hipError_t hipMalloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
hipError_t hipFree(void* ptr)                 { return cudaFree(ptr); }
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    return cudaMallocManaged(ptr, size, flags ? flags : hipMemAttachGlobal);
}
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
    return cudaMallocAsync(ptr, size, toStreamId(stream));
}
hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
    return cudaFreeAsync(ptr, toStreamId(stream));
}
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    return cudaHostAlloc(ptr, size, flags);
}
hipError_t hipHostFree(void* ptr) { return cudaFreeHost(ptr); }

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    return cudaMemcpy(dst, src, sizeBytes, static_cast<int>(kind));
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream) {
    return cudaMemcpyAsync(dst, src, sizeBytes, static_cast<int>(kind), toStreamId(stream));
}
hipError_t hipMemcpyHtoD(void* dst, void* src, size_t sizeBytes) {
    return cudaMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}
hipError_t hipMemcpyDtoH(void* dst, void* src, size_t sizeBytes) {
    return cudaMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}
hipError_t hipMemcpyDtoD(void* dst, void* src, size_t sizeBytes) {
    return cudaMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}
hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
    return cudaMemset(dst, value, sizeBytes);
}
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream) {
    return cudaMemsetAsync(dst, value, sizeBytes, toStreamId(stream));
}
hipError_t hipMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    return cudaMemGetInfo(freeBytes, totalBytes);
}

// ── Stream management ────────────────────────────────────────────────────────
hipError_t hipStreamCreate(hipStream_t* stream) {
    if (!stream) return hipErrorInvalidValue;
    uint64_t id = 0;
    int r = cudaStreamCreate(&id);
    if (r == hipSuccess) *stream = toHipStream(id);
    return r;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    if (!stream) return hipErrorInvalidValue;
    uint64_t id = 0;
    int r = cudaStreamCreateWithFlags(&id, flags);
    if (r == hipSuccess) *stream = toHipStream(id);
    return r;
}
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    if (!stream) return hipErrorInvalidValue;
    uint64_t id = 0;
    int r = cudaStreamCreateWithPriority(&id, flags, priority);
    if (r == hipSuccess) *stream = toHipStream(id);
    return r;
}
hipError_t hipStreamDestroy(hipStream_t stream)     { return cudaStreamDestroy(toStreamId(stream)); }
hipError_t hipStreamSynchronize(hipStream_t stream) { return cudaStreamSynchronize(toStreamId(stream)); }
hipError_t hipStreamQuery(hipStream_t stream)       { return cudaStreamQuery(toStreamId(stream)); }
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    return cudaStreamWaitEvent(toStreamId(stream), event, flags);
}

// ── Event management ─────────────────────────────────────────────────────────
hipError_t hipEventCreate(hipEvent_t* event) {
    return cudaEventCreate(reinterpret_cast<void**>(event));
}
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
    return cudaEventCreateWithFlags(reinterpret_cast<void**>(event), flags);
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    return cudaEventRecord(event, toStreamId(stream));
}
hipError_t hipEventSynchronize(hipEvent_t event) { return cudaEventSynchronize(event); }
hipError_t hipEventQuery(hipEvent_t event)       { return cudaEventQuery(event); }
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    return cudaEventElapsedTime(ms, start, stop);
}
hipError_t hipEventDestroy(hipEvent_t event)     { return cudaEventDestroy(event); }

// ── Module / kernel launch ───────────────────────────────────────────────────
hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    return fromCu(cuModuleLoad(reinterpret_cast<void**>(module), fname));
}
hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    return fromCu(cuModuleLoadData(reinterpret_cast<void**>(module), image));
}
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) {
    if (!function) return hipErrorInvalidValue;
    uint64_t id = 0;
    int r = cuModuleGetFunction(&id, module, kname);
    if (r == 0) *function = reinterpret_cast<hipFunction_t>(static_cast<uintptr_t>(id));
    return fromCu(r);
}
hipError_t hipModuleUnload(hipModule_t module) {
    return fromCu(cuModuleUnload(module));
}
hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra) {
    return fromCu(cuLaunchKernel(reinterpret_cast<uintptr_t>(f),
                                 gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, toStreamId(stream),
                                 kernelParams, extra));
}

} // extern "C"
