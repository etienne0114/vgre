// CUDA Driver API — error strings, profiler

#include "cuda_driver_internal.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <string>

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
  auto &prof = vgre::advanced::RuntimeProfiler::instance();
  prof.setEnabled(true);
  const char *dumpPath = vgre_get_config("VGRE_PROFILER_DUMP");
  VGRE_LOG_INFO("CUDriver", std::string("cuProfilerStart(): profiling enabled") +
                (dumpPath && dumpPath[0] ? std::string(" (dump=") + dumpPath + ")" : ""));
  return CUDA_SUCCESS;
}

CUresult cuProfilerStop(void) {
  auto &prof = vgre::advanced::RuntimeProfiler::instance();
  prof.setEnabled(false);
  const char *dumpPath = vgre_get_config("VGRE_PROFILER_DUMP");
  if (dumpPath && dumpPath[0]) {
    prof.exportToFile(std::string(dumpPath));
    VGRE_LOG_INFO("CUDriver", std::string("cuProfilerStop(): profiler dumped to ") + dumpPath);
  } else {
    VGRE_LOG_INFO("CUDriver", "cuProfilerStop(): profiling disabled");
  }
  return CUDA_SUCCESS;
}

} // extern "C"
