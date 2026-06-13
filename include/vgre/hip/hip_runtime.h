// VGRE HIP/ROCm compatibility layer — docs/missingFeatures.md §7.2.
// AMD HIP is a near-1:1 mirror of the CUDA runtime API, so HIP programs run on
// VGRE by mapping hip* → VGRE's cuda* implementations. This header provides the
// HIP types/enums and entry points; src/hip/hip_runtime.cpp delegates to the
// CUDA runtime (hipMemcpyKind values are numerically identical to CUDA's).
#ifndef VGRE_HIP_HIP_RUNTIME_H
#define VGRE_HIP_HIP_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int hipError_t;
enum { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorOutOfMemory = 2,
       hipErrorNotInitialized = 3, hipErrorInvalidDevice = 10 };

// HIP memcpy kinds share CUDA's numeric values.
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
} hipMemcpyKind;

hipError_t  hipMalloc(void** ptr, size_t size);
hipError_t  hipFree(void* ptr);
hipError_t  hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t  hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t  hipDeviceSynchronize(void);
hipError_t  hipGetDeviceCount(int* count);
hipError_t  hipGetLastError(void);
hipError_t  hipPeekAtLastError(void);
const char* hipGetErrorString(hipError_t err);

#ifdef __cplusplus
}
#endif

#endif // VGRE_HIP_HIP_RUNTIME_H
