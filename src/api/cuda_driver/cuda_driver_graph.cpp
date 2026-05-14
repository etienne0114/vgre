// CUDA Driver API — graph management and stream capture

#include "cuda_driver_internal.h"
#include "vgre/common/logger.h"
#include <vector>

// Forward declarations from cudart_shim.cpp (same shared library)
extern "C++" std::string vgre_lookup_kernel_name(const void *hostFn);
extern "C++" std::string vgre_lookup_kernel_source(const void *hostFn);

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

  auto *p = static_cast<const CUDA_MEMSET_NODE_PARAMS*>(memsetParams);
  if (!p->dst) return CUDA_ERROR_INVALID_VALUE;

  std::vector<uint64_t> deps;
  if (numDependencies > 0 && dependencies)
    deps.assign(reinterpret_cast<uint64_t*>(dependencies),
                reinterpret_cast<uint64_t*>(dependencies) + numDependencies);

  uint64_t outNodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddMemsetNode(
      reinterpret_cast<vgre::GraphId>(hGraph), p->dst, static_cast<int>(p->value),
      p->pitch, p->width, p->height, 1 /*depth*/, deps, outNodeId);
  if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_UNKNOWN;

  *phGraphNode = reinterpret_cast<void*>(outNodeId);
  return CUDA_SUCCESS;
}

CUresult cuGraphAddKernelNode(void **phGraphNode, void *hGraph,
                              void **dependencies, size_t numDependencies,
                              const void *nodeParams) {
  if (!phGraphNode || !hGraph || !nodeParams) return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies > 0 && !dependencies) return CUDA_ERROR_INVALID_VALUE;

  auto *p = static_cast<const CUDA_KERNEL_NODE_PARAMS*>(nodeParams);
  if (p->func == 0) return CUDA_ERROR_INVALID_VALUE;

  std::string kname = vgre_lookup_kernel_name(reinterpret_cast<void*>(p->func));
  std::string ksrc  = vgre_lookup_kernel_source(reinterpret_cast<void*>(p->func));
  if (kname.empty() || ksrc.empty()) {
    VGRE_LOG_ERROR("CUDA_DRIVER", "cuGraphAddKernelNode: unknown host function");
    return CUDA_ERROR_INVALID_VALUE;
  }

  vgre::KernelId kid = 0;
  if (vgre::core::RuntimeEngine::instance().registerKernel(kname, ksrc, kid)
      != vgre::VGREResult::SUCCESS || kid == 0)
    return CUDA_ERROR_UNKNOWN;

  std::vector<vgre::ArgType> argTypes;
  vgre::core::RuntimeEngine::instance().getKernelArgTypes(kid, argTypes);

  vgre::dim3 grid(p->gridDimX, p->gridDimY, p->gridDimZ);
  vgre::dim3 block(p->blockDimX, p->blockDimY, p->blockDimZ);
  if (grid.x == 0)  grid.x  = 1;
  if (block.x == 0) block.x = 1;

  std::vector<uint64_t> deps;
  if (numDependencies > 0 && dependencies)
    deps.assign(reinterpret_cast<uint64_t*>(dependencies),
                reinterpret_cast<uint64_t*>(dependencies) + numDependencies);

  uint64_t outNodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddKernelNode(
      reinterpret_cast<vgre::GraphId>(hGraph), kid, kname,
      grid, block, p->kernelParams, argTypes, deps, outNodeId);
  if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_UNKNOWN;

  *phGraphNode = reinterpret_cast<void*>(outNodeId);
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
