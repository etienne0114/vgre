#ifndef VGRE_API_CUDA_INTERCEPTOR_H
#define VGRE_API_CUDA_INTERCEPTOR_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/event.h"
#include "vgre/core/texture_manager.h"
#include <cstddef>
#include <string>

namespace vgre {
namespace api {

// ── CUDA-compatible error type ─────────────────────────────────────────────
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidDevice = 10;
constexpr cudaError_t cudaErrorNotInitialized = 3;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorLaunchFailure = 4;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr cudaError_t cudaErrorInvalidDeviceFunction = 8;
constexpr cudaError_t cudaErrorFileNotFound    = 301;
constexpr cudaError_t cudaErrorNotSupported    = 801;
constexpr cudaError_t cudaErrorInvalidPtx      = 218;
constexpr cudaError_t cudaErrorStubLibrary     = 34;
constexpr cudaError_t cudaErrorDevicesUnavailable = 46;

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
  cudaError_t deviceReset();
  cudaError_t setDeviceFlags(unsigned int flags);
  cudaError_t getDeviceFlags(unsigned int *flags);
  cudaError_t deviceGetPCIBusId(char *pciBusId, int len, int device);
  cudaError_t deviceGetByPCIBusId(int *device, const char *pciBusId);
  cudaError_t deviceCanAccessPeer(int *canAccessPeer, int device,
                                  int peerDevice);
  cudaError_t deviceEnablePeerAccess(int peerDevice, unsigned int flags);
  cudaError_t deviceDisablePeerAccess(int peerDevice);

  // ── Memory Management ──────────────────────────────────────────────────
  cudaError_t malloc(void **devPtr, size_t size);
  cudaError_t mallocManaged(void **devPtr, size_t size, unsigned int flags = 0);
  cudaError_t free(void *devPtr);
  cudaError_t memcpy(void *dst, const void *src, size_t count,
                     cudaMemcpyKind_t kind);
  cudaError_t memcpyAsync(void *dst, const void *src, size_t count,
                          cudaMemcpyKind_t kind, cudaStream_t stream);
  cudaError_t memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, cudaMemcpyKind_t kind);
  cudaError_t memcpy2DAsync(void *dst, size_t dpitch, const void *src,
                            size_t spitch, size_t width, size_t height,
                            cudaMemcpyKind_t kind, cudaStream_t stream);
  cudaError_t memcpyBatchAsync(void **dstPtr, const void **srcPtr,
                               size_t *size, size_t count,
                               cudaMemcpyKind_t kind, cudaStream_t stream);
  cudaError_t memcpyPeer(void *dst, int dstDevice, const void *src,
                         int srcDevice, size_t count);
  cudaError_t memcpyPeerAsync(void *dst, int dstDevice, const void *src,
                              int srcDevice, size_t count,
                              cudaStream_t stream);
  cudaError_t memset(void *devPtr, int value, size_t count);
  cudaError_t memsetAsync(void *devPtr, int value, size_t count,
                          cudaStream_t stream);
  cudaError_t mallocPitch(void **devPtr, size_t *pitch, size_t width,
                          size_t height);
  cudaError_t hostAlloc(void **pHost, size_t size, unsigned int flags);
  cudaError_t freeHost(void *pHost);
  cudaError_t hostRegister(void *pHost, size_t size, unsigned int flags);
  cudaError_t hostUnregister(void *pHost);

  // ── UVM Migration Hints ────────────────────────────────────────────────
  cudaError_t memAdvise(const void *devPtr, size_t count, int advice, int device);
  cudaError_t memPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);

  // ── Stream Management ──────────────────────────────────────────────────
  cudaError_t streamCreate(cudaStream_t *stream);
  cudaError_t streamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
  cudaError_t streamCreateWithPriority(cudaStream_t *stream, unsigned int flags,
                                       int priority);
  cudaError_t streamDestroy(cudaStream_t stream);
  cudaError_t streamSynchronize(cudaStream_t stream);
  cudaError_t streamQuery(cudaStream_t stream);
  cudaError_t streamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                              unsigned int flags);
  cudaError_t deviceGetStreamPriorityRange(int *leastPriority,
                                           int *greatestPriority);

  // ── Event Management ───────────────────────────────────────────────────
  cudaError_t eventCreate(cudaEvent_t *event);
  cudaError_t eventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
  cudaError_t eventRecord(cudaEvent_t event, cudaStream_t stream);
  cudaError_t eventQuery(cudaEvent_t event);
  cudaError_t eventSynchronize(cudaEvent_t event);
  cudaError_t eventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
  cudaError_t eventDestroy(cudaEvent_t event);

  // ── Kernel Launch ──────────────────────────────────────────────────────
  cudaError_t launchKernel(const std::string &name, const std::string &source,
                           dim3 gridDim, dim3 blockDim, void **args,
                           size_t sharedMem = 0, cudaStream_t stream = 0);
  cudaError_t launchCooperativeKernel(const std::string &name,
                                      const std::string &source,
                                      dim3 gridDim, dim3 blockDim,
                                      void **args, size_t sharedMem = 0,
                                      cudaStream_t stream = 0);

  // ── Module Management (Driver API style) ───────────────────────────────
  cudaError_t moduleLoad(CUmodule *module, const char *fname);
  cudaError_t moduleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                const char *name);
  cudaError_t moduleUnload(CUmodule hmod);

  // ── Device Attributes / Version / Memory Info ─────────────────────────
  cudaError_t deviceGetAttribute(int *value, int attr, int device);
  cudaError_t driverGetVersion(int *version);
  cudaError_t runtimeGetVersion(int *version);
  cudaError_t memGetInfo(size_t *freeBytes, size_t *totalBytes);

  // ── Error Handling ─────────────────────────────────────────────────────
  const char *getErrorString(cudaError_t error);
  cudaError_t getLastError();        // Returns and clears last error
  cudaError_t peekLastError() const; // Returns last error without clearing

  // ── Texture/Surface Memory API ──────────────────────────────────────────
  using cudaTextureObject_t = uint64_t;
  using cudaSurfaceObject_t = uint64_t;

  struct cudaChannelFormatDesc {
    int x, y, z, w;
    int f; // 0=float, 1=signed, 2=unsigned
  };

  enum CUresourceViewFormat {
    CU_RES_VIEW_FORMAT_NONE = 0x00,
    CU_RES_VIEW_FORMAT_UINT_8X1 = 0x01,
    CU_RES_VIEW_FORMAT_UINT_8X2 = 0x02,
    CU_RES_VIEW_FORMAT_UINT_8X4 = 0x03,
    CU_RES_VIEW_FORMAT_SINT_8X1 = 0x04,
    CU_RES_VIEW_FORMAT_SINT_8X2 = 0x05,
    CU_RES_VIEW_FORMAT_SINT_8X4 = 0x06,
    CU_RES_VIEW_FORMAT_UINT_16X1 = 0x07,
    CU_RES_VIEW_FORMAT_UINT_16X2 = 0x08,
    CU_RES_VIEW_FORMAT_UINT_16X4 = 0x09,
    CU_RES_VIEW_FORMAT_SINT_16X1 = 0x0a,
    CU_RES_VIEW_FORMAT_SINT_16X2 = 0x0b,
    CU_RES_VIEW_FORMAT_SINT_16X4 = 0x0c,
    CU_RES_VIEW_FORMAT_UINT_32X1 = 0x0d,
    CU_RES_VIEW_FORMAT_UINT_32X2 = 0x0e,
    CU_RES_VIEW_FORMAT_UINT_32X4 = 0x0f,
    CU_RES_VIEW_FORMAT_SINT_32X1 = 0x10,
    CU_RES_VIEW_FORMAT_SINT_32X2 = 0x11,
    CU_RES_VIEW_FORMAT_SINT_32X4 = 0x12,
    CU_RES_VIEW_FORMAT_FLOAT_16X1 = 0x13,
    CU_RES_VIEW_FORMAT_FLOAT_16X2 = 0x14,
    CU_RES_VIEW_FORMAT_FLOAT_16X4 = 0x15,
    CU_RES_VIEW_FORMAT_FLOAT_32X1 = 0x16,
    CU_RES_VIEW_FORMAT_FLOAT_32X2 = 0x17,
    CU_RES_VIEW_FORMAT_FLOAT_32X4 = 0x18,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c,
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e,
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20,
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22
  };

  struct CUDA_RESOURCE_VIEW_DESC {
    CUresourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
    unsigned int reserved[16];
  };

  struct cudaResourceDesc {
    int resType;
    struct {
      void *devPtr;
      cudaChannelFormatDesc desc;
      size_t sizeInBytes;
      size_t width;
      size_t height;
      size_t depth;
      size_t pitchInBytes;
      unsigned int layers;
    } res;
  };

  struct cudaTextureDesc {
    int addressMode[3];
    int filterMode;
    int readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
  };

  cudaError_t createTextureObject(cudaTextureObject_t *pTexObject,
                                  const cudaResourceDesc *pResDesc,
                                  const cudaTextureDesc *pTexDesc,
                                  const void *pResViewDesc);
  cudaError_t destroyTextureObject(cudaTextureObject_t texObject);

  cudaError_t createSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                  const cudaResourceDesc *pResDesc);
  cudaError_t destroySurfaceObject(cudaSurfaceObject_t surfObject);

  // ── CUDA Graphs API ────────────────────────────────────────────────────────
  using cudaGraphNode_t = uint64_t;

  enum cudaGraphExecUpdateResult {
      cudaGraphExecUpdateSuccess = 0,
      cudaGraphExecUpdateError = 1,
      cudaGraphExecUpdateErrorTopologyChanged = 2,
      cudaGraphExecUpdateErrorNodeTypeChanged = 3,
      cudaGraphExecUpdateErrorFunctionChanged = 4,
      cudaGraphExecUpdateErrorParametersChanged = 5,
      cudaGraphExecUpdateErrorNotSupported = 6,
      cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
      cudaGraphExecUpdateErrorAttributesChanged = 8
  };

  struct cudaPitchedPtr {
      void   *ptr;
      size_t  pitch;
      size_t  xsize;
      size_t  ysize;
  };
  struct cudaPos { size_t x, y, z; };
  struct cudaExtent { size_t width, height, depth; };
  // cudaArray handle backed by TextureManager::TextureId
  using cudaArray_t = uint64_t;

  struct cudaMemcpy3DParms {
      cudaArray_t            srcArray;  // TextureManager-backed array handle
      struct cudaPos         srcPos;
      struct cudaPitchedPtr  srcPtr;
      cudaArray_t            dstArray;  // TextureManager-backed array handle
      struct cudaPos         dstPos;
      struct cudaPitchedPtr  dstPtr;
      struct cudaExtent      extent;
      cudaMemcpyKind_t       kind;
  };

  // ── cudaArray Lifecycle (backed by TextureManager) ──────────────────────
  cudaError_t mallocArray(cudaArray_t *array, const cudaChannelFormatDesc *desc,
                          size_t width, size_t height, unsigned int flags);
  cudaError_t freeArray(cudaArray_t array);
  cudaError_t memcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                            const void *src, size_t count, cudaMemcpyKind_t kind);
  cudaError_t memcpyFromArray(void *dst, cudaArray_t src, size_t wOffset,
                              size_t hOffset, size_t count, cudaMemcpyKind_t kind);

  cudaError_t graphCreate(cudaGraph_t *graph, unsigned int flags);
  cudaError_t graphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
  cudaError_t streamBeginCapture(cudaStream_t stream);
  cudaError_t streamBeginCaptureToGraph(cudaStream_t stream,
                                         cudaGraph_t graph,
                                         const std::vector<uint64_t> &dependencies,
                                         unsigned int mode);
  cudaError_t streamEndCapture(cudaStream_t stream, cudaGraph_t *graph);
  // Stream attribute queries
  int          getStreamPriority(cudaStream_t stream) const;
  unsigned int getStreamFlags(cudaStream_t stream) const;
  cudaError_t graphInstantiate(cudaGraphExec_t *exec, cudaGraph_t graph);
  cudaError_t graphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
  cudaError_t graphDestroy(cudaGraph_t graph);
  cudaError_t graphExecDestroy(cudaGraphExec_t exec);

  cudaError_t graphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                 const cudaMemcpy3DParms *pCopyParams);
  
  cudaError_t graphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   void *dst, const void *src, size_t count, cudaMemcpyKind_t kind);

  cudaError_t graphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                              cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out);

  // Singleton
  static CUDAInterceptor &instance();

private:
  cudaError_t convertResult(VGREResult r);
  int convertMemcpyKind(cudaMemcpyKind_t kind);

  vgre::core::TextureElementType mapViewFormat(CUresourceViewFormat format);
  vgre::core::TextureElementType mapChannelDesc(const cudaChannelFormatDesc &cd);

  bool initialized_ = false;
  unsigned int deviceFlags_ = 0;
};

} // namespace api
} // namespace vgre

#endif // VGRE_API_CUDA_INTERCEPTOR_H
