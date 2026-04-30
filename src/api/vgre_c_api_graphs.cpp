/**
 * VGRE C API — Graph API Implementation
 */

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/core/runtime_engine.h"

#include <cstdlib>
#include <cstring>
#include <vector>

// Local helpers for this translation unit:
namespace {

static int to_status_local(vgre::VGREResult r) {
  switch (r) {
  case vgre::VGREResult::SUCCESS:            return VGRE_SUCCESS;
  case vgre::VGREResult::ERR_OUT_OF_MEMORY:  return VGRE_ERROR_OUT_OF_MEMORY;
  case vgre::VGREResult::ERR_INVALID_VALUE:  return VGRE_ERROR_INVALID_VALUE;
  case vgre::VGREResult::ERR_INVALID_KERNEL: return VGRE_ERROR_INVALID_KERNEL;
  case vgre::VGREResult::ERR_LAUNCH_FAILURE: return VGRE_ERROR_LAUNCH_FAILURE;
  case vgre::VGREResult::ERR_IO:             return VGRE_ERROR_IO;
  case vgre::VGREResult::ERR_NOT_INITIALIZED:return VGRE_ERROR_NOT_INIT;
  case vgre::VGREResult::ERR_AUTH_FAILED:    return VGRE_ERROR_AUTH_FAILED;
  case vgre::VGREResult::ERR_CRYPTO:         return VGRE_ERROR_CRYPTO;
  default:                                   return VGRE_ERROR_GENERIC;
  }
}

static int require_initialized_local() {
  if (!vgre::core::RuntimeEngine::instance().isInitialized())
    return VGRE_ERROR_NOT_INIT;
  return VGRE_SUCCESS;
}

static bool to_arg_types_local(const uint8_t *types, int count,
                                std::vector<vgre::ArgType> &out) {
  out.clear();
  if (count < 0) return false;
  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    uint8_t t = types ? types[i] : VGRE_ARG_POINTER;
    switch (t) {
    case VGRE_ARG_POINTER:  out.push_back(vgre::ArgType::POINTER);  break;
    case VGRE_ARG_INT32:    out.push_back(vgre::ArgType::INT32);    break;
    case VGRE_ARG_INT64:    out.push_back(vgre::ArgType::INT64);    break;
    case VGRE_ARG_FLOAT32:  out.push_back(vgre::ArgType::FLOAT32);  break;
    case VGRE_ARG_FLOAT64:  out.push_back(vgre::ArgType::FLOAT64);  break;
    case VGRE_ARG_UINT32:   out.push_back(vgre::ArgType::UINT32);   break;
    case VGRE_ARG_UINT64:   out.push_back(vgre::ArgType::UINT64);   break;
    case VGRE_ARG_STRUCT:   out.push_back(vgre::ArgType::STRUCT);   break;
    default: return false;
    }
  }
  return true;
}

} // anonymous namespace

/* ── CUDA Graphs (DAG) ──────────────────────────────────────────────────────
 */
int vgre_graphCreate(uint64_t *out_graph) {
  if (!out_graph)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  vgre::GraphId gid = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphCreate(gid);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_graph = gid;
  return VGRE_SUCCESS;
}

int vgre_graphClone(uint64_t src_graph, uint64_t *out_clone) {
  if (!out_clone)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  vgre::GraphId clone = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphClone(
      static_cast<vgre::GraphId>(src_graph), clone);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_clone = clone;
  return VGRE_SUCCESS;
}

int vgre_graphDestroy(uint64_t graph) {
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphDestroy(
      static_cast<vgre::GraphId>(graph));
  return to_status_local(r);
}

int vgre_graphInstantiate(uint64_t graph, uint64_t *out_exec) {
  if (!out_exec)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  vgre::GraphExecId exec = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphInstantiate(
      static_cast<vgre::GraphId>(graph), exec);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_exec = exec;
  return VGRE_SUCCESS;
}

int vgre_graphExecDestroy(uint64_t exec) {
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphExecDestroy(
      static_cast<vgre::GraphExecId>(exec));
  return to_status_local(r);
}

int vgre_graphLaunch(uint64_t exec, uint64_t stream) {
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphLaunch(
      static_cast<vgre::GraphExecId>(exec),
      static_cast<vgre::StreamId>(stream));
  return to_status_local(r);
}

int vgre_graphAddKernelNodeEx(uint64_t graph, uint64_t kernel_id,
                              const char *name, const uint32_t grid_dim[3],
                              const uint32_t block_dim[3], void **args,
                              const uint8_t *arg_types, int num_args,
                              const uint64_t *deps, int num_deps,
                              uint64_t *out_node_id) {
  if (!name || !grid_dim || !block_dim || num_args < 0 || num_deps < 0 ||
      !out_node_id) {
    return VGRE_ERROR_INVALID_VALUE;
  }
  if (num_args > 0 && !args)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_deps > 0 && !deps)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;

  std::vector<vgre::ArgType> argTypes;
  if (!to_arg_types_local(arg_types, num_args, argTypes))
    return VGRE_ERROR_INVALID_VALUE;
  std::vector<uint64_t> depList;
  depList.assign(deps, deps + num_deps);

  vgre::dim3 gd(grid_dim[0], grid_dim[1], grid_dim[2]);
  vgre::dim3 bd(block_dim[0], block_dim[1], block_dim[2]);
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddKernelNode(
      static_cast<vgre::GraphId>(graph),
      static_cast<vgre::KernelId>(kernel_id), std::string(name), gd, bd, args,
      argTypes, depList, nodeId);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_node_id = nodeId;
  return VGRE_SUCCESS;
}

int vgre_graphAddMemcpyNodeEx(uint64_t graph, void *dst, void *src,
                              size_t count, int kind, const uint64_t *deps,
                              int num_deps, uint64_t *out_node_id) {
  if (!dst || !src || count == 0 || num_deps < 0 || !out_node_id)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_deps > 0 && !deps)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  std::vector<uint64_t> depList;
  depList.assign(deps, deps + num_deps);
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddMemcpyNode(
      static_cast<vgre::GraphId>(graph), dst, src, count, kind, depList,
      nodeId);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_node_id = nodeId;
  return VGRE_SUCCESS;
}

// Add a conditional node (IF or WHILE) to a graph.
// cond_type: 0 = IF (body executes once if condFn != 0),
//            1 = WHILE (body loops while condFn != 0, up to max_iterations).
// body_graph: GraphId of the subgraph to execute as the body.
// max_iterations: safety cap for WHILE loops (0 → default 65536).
int vgre_graphAddConditionalNodeEx(uint64_t graph, int (*cond_fn)(void *),
                                    void *cond_ctx, uint64_t body_graph,
                                    int cond_type, unsigned int max_iterations,
                                    const uint64_t *deps, int num_deps,
                                    uint64_t *out_node_id) {
  if (!cond_fn || !out_node_id || num_deps < 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_deps > 0 && !deps)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  std::vector<uint64_t> depList(deps, deps + num_deps);
  vgre::core::GraphCondType ct =
      (cond_type == 1) ? vgre::core::GraphCondType::WHILE
                       : vgre::core::GraphCondType::IF;
  unsigned int iters = (max_iterations == 0) ? 65536 : max_iterations;
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddConditionalNode(
      static_cast<vgre::GraphId>(graph), cond_fn, cond_ctx,
      static_cast<vgre::GraphId>(body_graph), ct, iters, depList, nodeId);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status_local(r);
  *out_node_id = nodeId;
  return VGRE_SUCCESS;
}

int vgre_graphAddDependency(uint64_t graph, uint64_t node_id,
                            uint64_t depends_on) {
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphAddDependency(
      static_cast<vgre::GraphId>(graph), node_id, depends_on);
  return to_status_local(r);
}

int vgre_graphUpdateKernelNode(uint64_t graph, uint64_t node_id, void **args,
                               const uint8_t *arg_types, int num_args) {
  if (num_args < 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  std::vector<vgre::ArgType> argTypes;
  if (!to_arg_types_local(arg_types, num_args, argTypes))
    return VGRE_ERROR_INVALID_VALUE;
  auto r = vgre::core::RuntimeEngine::instance().graphUpdateKernelNode(
      static_cast<vgre::GraphId>(graph), node_id, args, argTypes);
  return to_status_local(r);
}

int vgre_graphUpdateMemcpyNode(uint64_t graph, uint64_t node_id, void *dst,
                               void *src, size_t count, int kind) {
  if (!dst || !src || count == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_local(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphUpdateMemcpyNode(
      static_cast<vgre::GraphId>(graph), node_id, dst, src, count, kind);
  return to_status_local(r);
}
