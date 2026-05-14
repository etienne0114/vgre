// CUDA Driver API — error strings, profiler

#include "cuda_driver_internal.h"

extern "C" {

CUresult cuGetErrorName(CUresult error, const char **pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case 0:   *pStr = "CUDA_SUCCESS"; break;
    case 1:   *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
    case 3:   *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
    case 101: *pStr = "CUDA_ERROR_INVALID_DEVICE"; break;
    case 801: *pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
    case 999: *pStr = "CUDA_ERROR_UNKNOWN"; break;
    default:  *pStr = "<unknown>"; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **pStr) {
  if (!pStr) return CUDA_ERROR_INVALID_VALUE;
  switch (error) {
    case 0:   *pStr = "no error"; break;
    case 1:   *pStr = "invalid argument"; break;
    case 3:   *pStr = "driver not initialized"; break;
    case 101: *pStr = "invalid device ordinal"; break;
    case 801: *pStr = "operation not supported"; break;
    case 999: *pStr = "unknown error"; break;
    default:  *pStr = "unknown error"; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuProfilerStart(void) {
  // VGRE profiler is always on in TRACE/DEBUG builds; this is a no-op
  return CUDA_SUCCESS;
}

CUresult cuProfilerStop(void) {
  // No-op; profiling data is flushed at process exit via RuntimeProfiler
  return CUDA_SUCCESS;
}

} // extern "C"
