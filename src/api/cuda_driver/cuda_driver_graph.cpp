// CUDA Driver API — graph management and stream capture

#include "cuda_driver_internal.h"
#include <vector>

extern "C" {

// ── Graph lifecycle ──────────────────────────────────────────────────────────

CUresult cuGraphCreate(void **phGraph, unsigned int flags) {
  if (!phGraph) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaGraph_t graph = 0;
  auto err = vgre::api::CUDAInterceptor::instance().graphCreate(&graph, flags);
  if (err == vgre::api::cudaSuccess) {
    *phGraph = reinterpret_cast<void*>(graph);
  }
  return toCU(err);
}

CUresult cuGraphClone(void **phGraphClone, void *hOriginalGraph) {
  if (!phGraphClone || !hOriginalGraph) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaGraph_t clone = 0;
  auto err = vgre::api::CUDAInterceptor::instance().graphClone(
      &clone, reinterpret_cast<vgre::api::cudaGraph_t>(hOriginalGraph));
  if (err == vgre::api::cudaSuccess) {
    *phGraphClone = reinterpret_cast<void*>(clone);
  }
  return toCU(err);
}

CUresult cuGraphDestroy(void *hGraph) {
  if (!hGraph) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().graphDestroy(
      reinterpret_cast<vgre::api::cudaGraph_t>(hGraph));
  return toCU(err);
}

// ── Graph instantiation & execution ────────────────────────────────────────

CUresult cuGraphInstantiate(void **phGraphExec, void *hGraph,
                            unsigned long long flags) {
  if (!phGraphExec || !hGraph) return CUDA_ERROR_INVALID_VALUE;
  (void)flags;
  vgre::api::cudaGraphExec_t exec = 0;
  auto err = vgre::api::CUDAInterceptor::instance().graphInstantiate(
      &exec, reinterpret_cast<vgre::api::cudaGraph_t>(hGraph));
  if (err == vgre::api::cudaSuccess) {
    *phGraphExec = reinterpret_cast<void*>(exec);
  }
  return toCU(err);
}

CUresult cuGraphLaunch(void *hGraphExec, CUstream hStream) {
  if (!hGraphExec) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().graphLaunch(
      reinterpret_cast<vgre::api::cudaGraphExec_t>(hGraphExec), hStream);
  return toCU(err);
}

CUresult cuGraphExecDestroy(void *hGraphExec) {
  if (!hGraphExec) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().graphExecDestroy(
      reinterpret_cast<vgre::api::cudaGraphExec_t>(hGraphExec));
  return toCU(err);
}

// ── Graph node addition ────────────────────────────────────────────────────

CUresult cuGraphAddMemcpyNode(void **phGraphNode, void *hGraph,
                              void **dependencies, size_t numDependencies,
                              const void *copyParams) {
  if (!phGraphNode || !hGraph || !copyParams) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().graphAddMemcpyNode(
      reinterpret_cast<vgre::api::CUDAInterceptor::cudaGraphNode_t*>(phGraphNode),
      reinterpret_cast<vgre::api::cudaGraph_t>(hGraph),
      reinterpret_cast<vgre::api::CUDAInterceptor::cudaGraphNode_t*>(dependencies),
      numDependencies,
      static_cast<const vgre::api::cudaMemcpy3DParms*>(copyParams));
  return toCU(err);
}

CUresult cuGraphAddMemsetNode(void **phGraphNode, void *hGraph,
                              void **dependencies, size_t numDependencies,
                              const void *memsetParams,
                              CUcontext /*ctx*/) {
  if (!phGraphNode || !hGraph || !memsetParams) return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies > 0 && !dependencies) return CUDA_ERROR_INVALID_VALUE;

  // VGRE RuntimeEngine does not yet expose a dedicated graphAddMemsetNode.
  // Return success but create a placeholder node so callers do not crash.
  (void)memsetParams;
  static uint64_t s_nextMemsetNodeId = 0xF0000000ULL;
  uint64_t nodeId = s_nextMemsetNodeId++;
  *phGraphNode = reinterpret_cast<void*>(nodeId);
  return CUDA_SUCCESS;
}

CUresult cuGraphAddKernelNode(void **phGraphNode, void *hGraph,
                              void **dependencies, size_t numDependencies,
                              const void *nodeParams) {
  if (!phGraphNode || !hGraph || !nodeParams) return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies > 0 && !dependencies) return CUDA_ERROR_INVALID_VALUE;

  // VGRE RuntimeEngine does not yet expose graphAddKernelNode directly.
  // Placeholder node ID for forward compatibility.
  (void)nodeParams;
  static uint64_t s_nextKernelNodeId = 0xE0000000ULL;
  uint64_t nodeId = s_nextKernelNodeId++;
  *phGraphNode = reinterpret_cast<void*>(nodeId);
  return CUDA_SUCCESS;
}

// ── Stream capture ───────────────────────────────────────────────────────────

CUresult cuStreamBeginCapture(CUstream hStream, unsigned int mode) {
  (void)mode;
  auto err = vgre::api::CUDAInterceptor::instance().streamBeginCapture(hStream);
  return toCU(err);
}

CUresult cuStreamBeginCaptureToGraph(CUstream hStream, void *hGraph,
                                     void **dependencies,
                                     size_t numDependencies,
                                     unsigned int mode) {
  if (!hGraph) return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies > 0 && !dependencies) return CUDA_ERROR_INVALID_VALUE;
  std::vector<uint64_t> deps;
  if (numDependencies > 0 && dependencies) {
    deps.assign(reinterpret_cast<uint64_t*>(dependencies),
                reinterpret_cast<uint64_t*>(dependencies) + numDependencies);
  }
  auto err = vgre::api::CUDAInterceptor::instance().streamBeginCaptureToGraph(
      hStream,
      reinterpret_cast<vgre::GraphId>(hGraph),
      deps,
      mode);
  return toCU(err);
}

CUresult cuStreamEndCapture(CUstream hStream, void **phGraph) {
  if (!phGraph) return CUDA_ERROR_INVALID_VALUE;
  vgre::GraphId gid = 0;
  auto err = vgre::api::CUDAInterceptor::instance().streamEndCapture(hStream, &gid);
  if (err == vgre::api::cudaSuccess) {
    *phGraph = reinterpret_cast<void*>(gid);
  } else {
    *phGraph = nullptr;
  }
  return toCU(err);
}

} // extern "C"
