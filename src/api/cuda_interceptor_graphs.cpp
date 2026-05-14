#include "vgre/api/cuda_interceptor.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/platform.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/texture_manager.h"
#include "vgre/core/virtual_gpu_device.h"

#include <cstring>
#include <vector>

namespace vgre {
namespace api {

// ── Module Management (Driver API style) ───────────────────────────────────
cudaError_t CUDAInterceptor::moduleLoad(CUmodule *module, const char *fname) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!module || !fname)
    return cudaErrorInvalidValue;

  auto r = core::RuntimeEngine::instance().loadModule(fname, *module);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::moduleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                               const char *name) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!hfunc || !hmod || !name)
    return cudaErrorInvalidValue;

  auto r =
      core::RuntimeEngine::instance().getKernelFromModule(hmod, name, *hfunc);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::moduleUnload(CUmodule hmod) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().unloadModule(hmod);
  cudaError_t err = convertResult(r);
  return err;
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::graphCreate(cudaGraph_t *graph, unsigned int flags) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  (void)flags;
  auto r = core::RuntimeEngine::instance().graphCreate(*graph);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphClone(cudaGraph_t *pGraphClone,
                                        cudaGraph_t originalGraph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!pGraphClone)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().graphClone(originalGraph, *pGraphClone);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::streamBeginCapture(cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  auto r = core::RuntimeEngine::instance().streamBeginCapture(stream);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::streamBeginCaptureToGraph(cudaStream_t stream,
                                                        cudaGraph_t graph,
                                                        const std::vector<uint64_t> &dependencies,
                                                        unsigned int /*mode*/) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  // Capture directly into the caller-provided graph and seed the dependency
  // frontier so the first captured node observes the requested incoming edges.
  auto r = core::RuntimeEngine::instance().streamBeginCaptureToGraph(
      stream, graph, dependencies);
  return convertResult(r);
}

int CUDAInterceptor::getStreamPriority(cudaStream_t stream) const {
  int priority = 0;
  core::RuntimeEngine::instance().getDevice().getStreamPriority(stream, priority);
  return priority;
}

unsigned int CUDAInterceptor::getStreamFlags(cudaStream_t stream) const {
  unsigned int flags = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamFlags(stream, flags);
  return flags;
}

cudaError_t CUDAInterceptor::streamEndCapture(cudaStream_t stream,
                                              cudaGraph_t *graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!graph)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().streamEndCapture(stream, *graph);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::graphInstantiate(cudaGraphExec_t *exec,
                                              cudaGraph_t graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!exec)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().graphInstantiate(graph, *exec);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::graphLaunch(cudaGraphExec_t exec,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().graphLaunch(exec, stream);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::graphDestroy(cudaGraph_t graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().graphDestroy(graph);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::graphExecDestroy(cudaGraphExec_t exec) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  auto r = core::RuntimeEngine::instance().graphExecDestroy(exec);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                const cudaMemcpy3DParms *pCopyParams) {
  if (!pGraphNode || !pCopyParams) return cudaErrorInvalidValue;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }

  // Resolve destination: prefer pitched pointer, fall back to cudaArray handle
  void *dst = pCopyParams->dstPtr.ptr;
  if (!dst && pCopyParams->dstArray != 0) {
    dst = core::TextureManager::instance().getCudaArrayData(pCopyParams->dstArray);
  }
  // Resolve source: prefer pitched pointer, fall back to cudaArray handle
  const void *src = pCopyParams->srcPtr.ptr;
  if (!src && pCopyParams->srcArray != 0) {
    src = core::TextureManager::instance().getCudaArrayData(pCopyParams->srcArray);
  }
  size_t count = pCopyParams->extent.width * (pCopyParams->extent.height > 0 ? pCopyParams->extent.height : 1) * (pCopyParams->extent.depth > 0 ? pCopyParams->extent.depth : 1);

  uint64_t outNodeId = 0;
  int internalKind = convertMemcpyKind(pCopyParams->kind);
  auto r = core::RuntimeEngine::instance().graphAddMemcpyNode(graph, dst, const_cast<void*>(src), count, internalKind, deps, outNodeId);
  if (r == VGREResult::SUCCESS) {
    *pGraphNode = outNodeId;
  }
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                  const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                  void *dst, const void *src, size_t count, cudaMemcpyKind_t kind) {
  if (!pGraphNode || !dst || !src) return cudaErrorInvalidValue;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }

  uint64_t outNodeId = 0;
  int internalKind = convertMemcpyKind(kind);
  auto r = core::RuntimeEngine::instance().graphAddMemcpyNode(graph, dst, const_cast<void*>(src), count, internalKind, deps, outNodeId);
  if (r == VGREResult::SUCCESS) {
    *pGraphNode = outNodeId;
  }
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                             cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  auto r = core::RuntimeEngine::instance().graphUpdateExec(hGraphExec, hGraph);
  if (r == VGREResult::SUCCESS && updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateSuccess;
  } else if (updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateError;
      if (hErrorNode_out) *hErrorNode_out = 0;
  }
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphExecUpdateV2(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                              const cudaGraphNode_t *updateNodeList, size_t updateNodeListSize,
                                              cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  std::vector<uint64_t> nodeIds;
  if (updateNodeList && updateNodeListSize > 0) {
      nodeIds.reserve(updateNodeListSize);
      for (size_t i = 0; i < updateNodeListSize; ++i)
          nodeIds.push_back(updateNodeList[i]);
  }
  auto r = core::RuntimeEngine::instance().graphUpdateExecV2(hGraphExec, hGraph, nodeIds);
  if (r == VGREResult::SUCCESS && updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateSuccess;
  } else if (updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateError;
      if (hErrorNode_out) *hErrorNode_out = 0;
  }
  return convertResult(r);
}

} // namespace api
} // namespace vgre

#ifndef VGRE_CUDA_INTERCEPTOR_NO_EXPORTS
extern "C" {

VGRE_PUBLIC_API
int cuInit(unsigned int flags) {
  (void)flags;
  return vgre::api::CUDAInterceptor::instance().init();
}

VGRE_PUBLIC_API
int cuMemAlloc(void** dptr, size_t bytesize) {
  return vgre::api::CUDAInterceptor::instance().malloc(dptr, bytesize);
}

VGRE_PUBLIC_API
int cuMemFree(void* dptr) {
  return vgre::api::CUDAInterceptor::instance().free(dptr);
}

VGRE_PUBLIC_API
int cuTexObjectCreate(uint64_t* pTexObject,
                     const vgre::api::CUDAInterceptor::cudaResourceDesc* pResDesc,
                     const vgre::api::CUDAInterceptor::cudaTextureDesc* pTexDesc,
                     const void *pResViewDesc) {
  return vgre::api::CUDAInterceptor::instance().createTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

VGRE_PUBLIC_API
int cuTexObjectDestroy(uint64_t texObject) {
  return vgre::api::CUDAInterceptor::instance().destroyTextureObject(texObject);
}

} // extern "C"
#endif
