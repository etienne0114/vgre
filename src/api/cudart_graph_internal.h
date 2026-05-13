/**
 * VGRE — Internal CUDART Graph Node types (shared by the graph shim files).
 *
 * This header is private to src/api/ and must NOT be installed.  It defines
 * the CUDA-compatible parameter structs and type aliases that all four graph
 * shim translation units need, preventing redefinition across files.
 */

#pragma once

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace vgre::api;

// ── CUDA-compatible type aliases ──────────────────────────────────────────────

using cudaGraphNode_t = CUDAInterceptor::cudaGraphNode_t;   // uint64_t
using cudaGraph_t     = vgre::GraphId;                       // uint64_t
using cudaGraphExec_t = vgre::GraphExecId;                   // uint64_t

// ── cudaKernelNodeParams — matches CUDA 12.x layout ──────────────────────────

struct cudaKernelNodeParams {
    void *func;
    struct { unsigned int x, y, z; } gridDim;
    struct { unsigned int x, y, z; } blockDim;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};

// ── cudaMemsetParams ──────────────────────────────────────────────────────────

struct cudaMemsetParams {
    void        *dst;
    size_t       pitch;
    unsigned int value;    // low 8 bits = fill byte
    size_t       width;
    size_t       height;
};

// ── cudaHostNodeParams ────────────────────────────────────────────────────────

struct cudaHostNodeParams {
    void (*fn)(void *);
    void *userData;
};

// ── cudaMemAllocNodeParams ────────────────────────────────────────────────────

struct cudaMemAllocNodeParams {
    void       *poolProps;       // reserved — unused (default pool)
    const void *accessDescs;     // reserved — unused
    size_t      accessDescCount;
    size_t      bytesize;
    void       *dptr;            // OUTPUT: allocated pointer written here
};

// ── cudaMemFreeNodeParams ─────────────────────────────────────────────────────

struct cudaMemFreeNodeParams {
    void *dptr;
};

// ── cudaGraphNodeType enum ────────────────────────────────────────────────────

enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel       = 0,
    cudaGraphNodeTypeMemcpy       = 1,
    cudaGraphNodeTypeMemset       = 2,
    cudaGraphNodeTypeHost         = 3,
    cudaGraphNodeTypeGraph        = 4,  // child graph
    cudaGraphNodeTypeEmpty        = 5,
    cudaGraphNodeTypeWaitEvent    = 6,
    cudaGraphNodeTypeEventRecord  = 7,
    cudaGraphNodeTypeExtSemSignal = 8,
    cudaGraphNodeTypeExtSemWait   = 9,
    cudaGraphNodeTypeMemAlloc     = 10,
    cudaGraphNodeTypeMemFree      = 11,
};

// ── Kernel name/source lookup — implemented in cudart_shim.cpp ───────────────
// Plain C++ symbols exported from the same shared library (vgre_cudart.so).

std::string vgre_lookup_kernel_name(const void *hostFn);
std::string vgre_lookup_kernel_source(const void *hostFn);

// ── RuntimeEngine singleton shorthand ────────────────────────────────────────

inline vgre::core::RuntimeEngine &RE() {
    return vgre::core::RuntimeEngine::instance();
}

// ── GraphNodeType → cudaGraphNodeType converter ───────────────────────────────

inline cudaGraphNodeType toPublicNodeType(vgre::core::GraphNodeType t) {
    switch (t) {
    case vgre::core::GraphNodeType::KERNEL:       return cudaGraphNodeTypeKernel;
    case vgre::core::GraphNodeType::MEMCPY:       return cudaGraphNodeTypeMemcpy;
    case vgre::core::GraphNodeType::MEMSET:       return cudaGraphNodeTypeMemset;
    case vgre::core::GraphNodeType::HOST:         return cudaGraphNodeTypeHost;
    case vgre::core::GraphNodeType::CHILD:        return cudaGraphNodeTypeGraph;
    case vgre::core::GraphNodeType::EMPTY:        return cudaGraphNodeTypeEmpty;
    case vgre::core::GraphNodeType::EVENT_RECORD: return cudaGraphNodeTypeEventRecord;
    case vgre::core::GraphNodeType::EVENT_WAIT:   return cudaGraphNodeTypeWaitEvent;
    case vgre::core::GraphNodeType::MEMALLOC:     return cudaGraphNodeTypeMemAlloc;
    case vgre::core::GraphNodeType::MEMFREE:      return cudaGraphNodeTypeMemFree;
    default:                                       return cudaGraphNodeTypeEmpty;
    }
}
