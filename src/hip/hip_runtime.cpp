// VGRE HIP/ROCm compatibility layer (impl).  See include/vgre/hip/hip_runtime.h.
// Each hip* entry delegates to the corresponding VGRE cuda* runtime symbol.
// cudaError_t is `int` and hipMemcpyKind mirrors cudaMemcpyKind numerically, so
// the mapping is a direct passthrough.

#include "vgre/hip/hip_runtime.h"

// VGRE's CUDA runtime exports (cudaError_t == int).
extern "C" {
int         cudaMalloc(void** devPtr, size_t size);
int         cudaFree(void* devPtr);
int         cudaMemcpy(void* dst, const void* src, size_t count, int kind);
int         cudaMemset(void* devPtr, int value, size_t count);
int         cudaDeviceSynchronize(void);
int         cudaGetDeviceCount(int* count);
int         cudaGetLastError(void);
int         cudaPeekAtLastError(void);
const char* cudaGetErrorString(int err);
}

extern "C" {

hipError_t hipMalloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
hipError_t hipFree(void* ptr) { return cudaFree(ptr); }
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    return cudaMemcpy(dst, src, sizeBytes, static_cast<int>(kind));
}
hipError_t hipMemset(void* dst, int value, size_t sizeBytes) { return cudaMemset(dst, value, sizeBytes); }
hipError_t hipDeviceSynchronize(void) { return cudaDeviceSynchronize(); }
hipError_t hipGetDeviceCount(int* count) { return cudaGetDeviceCount(count); }
hipError_t hipGetLastError(void) { return cudaGetLastError(); }
hipError_t hipPeekAtLastError(void) { return cudaPeekAtLastError(); }
const char* hipGetErrorString(hipError_t err) { return cudaGetErrorString(err); }

} // extern "C"
