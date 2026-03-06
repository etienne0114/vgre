#ifndef VGRE_API_CUDA_INTERCEPTOR_H
#define VGRE_API_CUDA_INTERCEPTOR_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/event.h"

#include <cstddef>
#include <string>

namespace vgre {
namespace api {

// ── CUDA-compatible error type ─────────────────────────────────────────────
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidDevice = 10;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorLaunchFailure = 4;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr cudaError_t cudaErrorInvalidDeviceFunction = 8;
constexpr cudaError_t cudaErrorFileNotFound = 301;

// ── CUDA Driver API Types ──────────────────────────────────────────────────
using CUmodule = ModuleHandle;
using CUfunction = KernelId;

// ── CUDA memcpy direction ──────────────────────────────────────────────────
enum cudaMemcpyKind_t {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};

// ── CUDA-compatible device properties ──────────────────────────────────────
struct cudaDeviceProp_t {
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

using cudaStream_t = StreamId;
using cudaGraph_t = GraphId;
using cudaGraphExec_t = GraphExecId;

// ── CUDA event handle ─────────────────────────────────────────────────────
using cudaEvent_t = vgre::core::Event *;

// ── CUDA Runtime API Interceptor ──────────────────────────────────────────
class CUDAInterceptor {
public:
  CUDAInterceptor();
  ~CUDAInterceptor();

  // Initialize the interceptor (call before any CUDA API)
  cudaError_t init();

  // ── Device Management ──────────────────────────────────────────────────
  cudaError_t getDeviceCount(int *count);
  cudaError_t setDevice(int device);
  cudaError_t getDevice(int *device);
  cudaError_t getDeviceProperties(cudaDeviceProp_t *prop, int device);
  cudaError_t deviceSynchronize();

  // ── Memory Management ──────────────────────────────────────────────────
  cudaError_t malloc(void **devPtr, size_t size);
  cudaError_t mallocManaged(void **devPtr, size_t size, unsigned int flags = 0);
  cudaError_t free(void *devPtr);
  cudaError_t memcpy(void *dst, const void *src, size_t count,
                     cudaMemcpyKind_t kind);
  cudaError_t memcpyAsync(void *dst, const void *src, size_t count,
                          cudaMemcpyKind_t kind, cudaStream_t stream);
  cudaError_t memset(void *devPtr, int value, size_t count);
  cudaError_t memsetAsync(void *devPtr, int value, size_t count,
                          cudaStream_t stream);

  // ── Stream Management ──────────────────────────────────────────────────
  cudaError_t streamCreate(cudaStream_t *stream);
  cudaError_t streamDestroy(cudaStream_t stream);
  cudaError_t streamSynchronize(cudaStream_t stream);

  // ── Event Management ───────────────────────────────────────────────────
  cudaError_t eventCreate(cudaEvent_t *event);
  cudaError_t eventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
  cudaError_t eventRecord(cudaEvent_t event, cudaStream_t stream);
  cudaError_t eventSynchronize(cudaEvent_t event);
  cudaError_t eventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
  cudaError_t eventDestroy(cudaEvent_t event);

  // ── Kernel Launch ──────────────────────────────────────────────────────
  cudaError_t launchKernel(const std::string &name, const std::string &source,
                           dim3 gridDim, dim3 blockDim, void **args,
                           size_t sharedMem = 0, cudaStream_t stream = 0);

  // ── Module Management (Driver API style) ───────────────────────────────
  cudaError_t moduleLoad(CUmodule *module, const char *fname);
  cudaError_t moduleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                const char *name);
  cudaError_t moduleUnload(CUmodule hmod);

  // ── Error Handling ─────────────────────────────────────────────────────
  const char *getErrorString(cudaError_t error);
  cudaError_t getLastError();

  // ── CUDA Graphs API ────────────────────────────────────────────────────────
  cudaError_t streamBeginCapture(cudaStream_t stream);
  cudaError_t streamEndCapture(cudaStream_t stream, cudaGraph_t *graph);
  cudaError_t graphInstantiate(cudaGraphExec_t *exec, cudaGraph_t graph);
  cudaError_t graphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
  cudaError_t graphDestroy(cudaGraph_t graph);
  cudaError_t graphExecDestroy(cudaGraphExec_t exec);

  // Singleton
  static CUDAInterceptor &instance();

private:
  cudaError_t convertResult(VGREResult r);

  bool initialized_ = false;
  cudaError_t lastError_ = cudaSuccess;
};

} // namespace api
} // namespace vgre

#endif // VGRE_API_CUDA_INTERCEPTOR_H
