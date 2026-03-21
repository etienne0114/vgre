/**
 * VGRE C API — Implementation
 *
 * Delegates all calls to the C++ RuntimeEngine, MemoryManager, and Scheduler
 * singletons. This file is compiled into libvgre.so and provides the flat
 * extern "C" interface that Python (ctypes) and other FFI consumers use.
 */

#include "vgre/api/vgre_c_api.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

// ── Helper: convert VGREResult to C status code ────────────────────────────
static int to_status(vgre::VGREResult r) {
  switch (r) {
  case vgre::VGREResult::SUCCESS:
    return VGRE_SUCCESS;
  case vgre::VGREResult::ERROR_OUT_OF_MEMORY:
    return VGRE_ERROR_OUT_OF_MEMORY;
  case vgre::VGREResult::ERROR_INVALID_VALUE:
    return VGRE_ERROR_INVALID_VALUE;
  case vgre::VGREResult::ERROR_INVALID_KERNEL:
    return VGRE_ERROR_INVALID_KERNEL;
  case vgre::VGREResult::ERROR_LAUNCH_FAILURE:
    return VGRE_ERROR_LAUNCH_FAILURE;
  case vgre::VGREResult::ERROR_IO:
    return VGRE_ERROR_IO;
  case vgre::VGREResult::ERROR_NOT_INITIALIZED:
    return VGRE_ERROR_NOT_INIT;
  case vgre::VGREResult::ERROR_AUTH_FAILED:
    return VGRE_ERROR_AUTH_FAILED;
  case vgre::VGREResult::ERROR_CRYPTO:
    return VGRE_ERROR_CRYPTO;
  case vgre::VGREResult::ERROR_INVALID_DEVICE:
    return VGRE_ERROR_INVALID_VALUE; // Map to invalid value for now, or add specific C code
  default:
    return VGRE_ERROR_GENERIC;
  }
}

static uint64_t telemetry_now_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

static std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

static int require_initialized() {
  if (!vgre::core::RuntimeEngine::instance().isInitialized()) {
    return VGRE_ERROR_NOT_INIT;
  }
  return VGRE_SUCCESS;
}

static bool to_arg_types(const uint8_t *types, int count,
                         std::vector<vgre::ArgType> &out) {
  out.clear();
  if (count < 0)
    return false;
  out.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    uint8_t t = types ? types[i] : VGRE_ARG_POINTER;
    switch (t) {
    case VGRE_ARG_POINTER:
      out.push_back(vgre::ArgType::POINTER);
      break;
    case VGRE_ARG_INT32:
      out.push_back(vgre::ArgType::INT32);
      break;
    case VGRE_ARG_INT64:
      out.push_back(vgre::ArgType::INT64);
      break;
    case VGRE_ARG_FLOAT32:
      out.push_back(vgre::ArgType::FLOAT32);
      break;
    case VGRE_ARG_FLOAT64:
      out.push_back(vgre::ArgType::FLOAT64);
      break;
    case VGRE_ARG_UINT32:
      out.push_back(vgre::ArgType::UINT32);
      break;
    case VGRE_ARG_UINT64:
      out.push_back(vgre::ArgType::UINT64);
      break;
    case VGRE_ARG_STRUCT:
      out.push_back(vgre::ArgType::STRUCT);
      break;
    default:
      return false;
    }
  }
  return true;
}

// ── Initialization ─────────────────────────────────────────────────────────

int vgre_init(void) {
  auto result = vgre::core::RuntimeEngine::instance().initialize();
  return to_status(result);
}

int vgre_shutdown(void) {
  auto r = vgre::core::RuntimeEngine::instance().shutdown();
  return to_status(r);
}

// ── Device Management ──────────────────────────────────────────────────────

int vgre_get_device_count(int *count) {
  if (!count)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  *count = vgre::core::RuntimeEngine::instance().getDeviceCount();
  return VGRE_SUCCESS;
}

int vgre_set_device(int device_id) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().setDevice(device_id);
  return to_status(r);
}

int vgre_get_device(int *device_id) {
  if (!device_id)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  *device_id = vgre::core::RuntimeEngine::instance().getDeviceId();
  return VGRE_SUCCESS;
}

int vgre_get_device_properties(int device_id, vgre_device_properties_t *props) {
  if (!props)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::DeviceProperties dp;
  auto r =
      vgre::core::RuntimeEngine::instance().getDeviceProperties(device_id, dp);
  if (r != vgre::VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("C_API", "getDeviceProperties failed for ID " + std::to_string(device_id) + " with code " + std::to_string(static_cast<int>(r)));
    return to_status(r);
  }

  std::memset(props, 0, sizeof(vgre_device_properties_t));
  std::snprintf(props->name, sizeof(props->name), "%s", dp.name);
  props->total_global_mem = dp.totalGlobalMem;
  props->shared_mem_per_block = dp.sharedMemPerBlock;
  props->max_threads_per_block = dp.maxThreadsPerBlock;
  std::memcpy(props->max_threads_dim, dp.maxThreadsDim,
              sizeof(dp.maxThreadsDim));
  std::memcpy(props->max_grid_size, dp.maxGridSize, sizeof(dp.maxGridSize));
  props->warp_size = dp.warpSize;
  props->multi_processor_count = dp.multiProcessorCount;
  props->major = dp.major;
  props->minor = dp.minor;
  props->clock_rate = dp.clockRate;
  props->total_const_mem = dp.totalConstMem;

  return VGRE_SUCCESS;
}

int vgre_synchronize(void) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().synchronize();
  return to_status(r);
}

// ── Memory Management ──────────────────────────────────────────────────────

int vgre_malloc(void **ptr, size_t size) {
  if (!ptr || size == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::MemoryHandle handle;
  auto r = vgre::core::RuntimeEngine::instance().malloc(size, handle);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *ptr = handle;
  return VGRE_SUCCESS;
}

int vgre_malloc_managed(void **ptr, size_t size) {
  if (!ptr || size == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::MemoryHandle handle;
  auto r = vgre::core::RuntimeEngine::instance().mallocManaged(size, handle);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *ptr = handle;
  return VGRE_SUCCESS;
}

int vgre_free(void *ptr) {
  if (!ptr)
    return VGRE_SUCCESS; // freeing NULL is valid
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().getMemoryManager().free(ptr);
  return to_status(r);
}

int vgre_memcpy(void *dst, const void *src, size_t count, int direction) {
  if (!dst || !src || count == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  vgre::VGREResult r;

  switch (direction) {
  case VGRE_MEMCPY_HOST_TO_DEVICE:
    r = mm.copyHostToDevice(dst, src, count);
    break;
  case VGRE_MEMCPY_DEVICE_TO_HOST:
    r = mm.copyDeviceToHost(dst, const_cast<void *>(src), count);
    break;
  case VGRE_MEMCPY_DEVICE_TO_DEVICE:
    r = mm.copyDeviceToDevice(dst, const_cast<void *>(src), count);
    break;
  default:
    return VGRE_ERROR_INVALID_VALUE;
  }

  return to_status(r);
}

int vgre_memset(void *ptr, int value, size_t count) {
  if (!ptr || count == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  if (!mm.isValidHandle(ptr))
    return VGRE_ERROR_INVALID_VALUE;
  size_t allocSize = mm.getAllocationSize(ptr);
  if (count > allocSize)
    return VGRE_ERROR_INVALID_VALUE;
  void *raw = mm.getPointer(ptr);
  if (!raw)
    return VGRE_ERROR_INVALID_VALUE;
  std::memset(raw, value, count);
  return VGRE_SUCCESS;
}

int vgre_device_can_access_peer(int *can_access, int device, int peer_device) {
  if (!can_access)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().deviceCanAccessPeer(
      device, peer_device, can_access);
  return to_status(r);
}

int vgre_device_enable_peer_access(int peer_device) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r =
      vgre::core::RuntimeEngine::instance().deviceEnablePeerAccess(peer_device);
  return to_status(r);
}

int vgre_device_disable_peer_access(int peer_device) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().deviceDisablePeerAccess(
      peer_device);
  return to_status(r);
}

// ── Kernel Registration & Launch ───────────────────────────────────────────

int vgre_register_kernel(const char *name, const char *source,
                         uint64_t *out_kernel_id) {
  if (!name || !source || !out_kernel_id)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::KernelId kid;
  auto r = vgre::core::RuntimeEngine::instance().registerKernel(
      std::string(name), std::string(source), kid);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  // Broadcast to cluster if enabled
  vgre::advanced::TCPClusterManager::instance().broadcastKernelRegistration(kid, std::string(name), std::string(source));

  *out_kernel_id = kid;
  return VGRE_SUCCESS;
}

int vgre_module_load(void **module, const char *path) {
  if (!module || !path)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto r = vgre::core::RuntimeEngine::instance().loadModule(path, *module);
  return to_status(r);
}

int vgre_module_get_function(uint64_t *kernel, void *module, const char *name) {
  if (!kernel || !module || !name)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::KernelId kid;
  auto r = vgre::core::RuntimeEngine::instance().getKernelFromModule(
      module, name, kid);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *kernel = kid;
  return VGRE_SUCCESS;
}

int vgre_module_get_global(void *module, const char *name, void **dptr,
                           size_t *bytes) {
  if (!module || !name || !dptr || !bytes)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  void *addr = nullptr;
  size_t size = 0;
  auto r = vgre::core::RuntimeEngine::instance().getModuleGlobal(module, name,
                                                                 addr, size);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *dptr = addr;
  *bytes = size;
  return VGRE_SUCCESS;
}

int vgre_module_unload(void *module) {
  if (!module)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto r = vgre::core::RuntimeEngine::instance().unloadModule(module);
  return to_status(r);
}

int vgre_launch_kernel(uint64_t kernel_id, const uint32_t grid_dim[3],
                       const uint32_t block_dim[3], void **args, int num_args,
                       size_t shared_mem, uint64_t stream_id) {
  if (!grid_dim || !block_dim)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  if (num_args < 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGRE_ERROR_INVALID_VALUE;
  if (grid_dim[0] == 0 || block_dim[0] == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if ((grid_dim[1] == 0 && grid_dim[2] != 0) ||
      (block_dim[1] == 0 && block_dim[2] != 0)) {
    return VGRE_ERROR_INVALID_VALUE;
  }

  vgre::dim3 gd(grid_dim[0], grid_dim[1], grid_dim[2]);
  vgre::dim3 bd(block_dim[0], block_dim[1], block_dim[2]);

  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      static_cast<vgre::KernelId>(kernel_id), gd, bd, args, shared_mem,
      static_cast<vgre::StreamId>(stream_id));
  return to_status(r);
}

// ── Stream Management ──────────────────────────────────────────────────────

int vgre_stream_create(uint64_t *out_stream_id) {
  if (!out_stream_id)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::StreamId id;
  auto r = vgre::core::RuntimeEngine::instance().getDevice().createStream(id);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *out_stream_id = id;
  return VGRE_SUCCESS;
}

int vgre_stream_create_with_priority(uint64_t *out_stream_id, int priority) {
  if (!out_stream_id)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::StreamId id;
  auto r = vgre::core::RuntimeEngine::instance().getDevice().createStream(
      id, priority);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *out_stream_id = id;
  return VGRE_SUCCESS;
}

int vgre_stream_synchronize(uint64_t stream_id) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().streamSynchronize(
      static_cast<vgre::StreamId>(stream_id));
  return to_status(r);
}

int vgre_stream_destroy(uint64_t stream_id) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().getDevice().destroyStream(
      static_cast<vgre::StreamId>(stream_id));
  return to_status(r);
}

/* ── CUDA Graphs (DAG) ──────────────────────────────────────────────────────
 */
int vgre_graphCreate(uint64_t *out_graph) {
  if (!out_graph)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  vgre::GraphId gid = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphCreate(gid);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);
  *out_graph = gid;
  return VGRE_SUCCESS;
}

int vgre_graphDestroy(uint64_t graph) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphDestroy(
      static_cast<vgre::GraphId>(graph));
  return to_status(r);
}

int vgre_graphInstantiate(uint64_t graph, uint64_t *out_exec) {
  if (!out_exec)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  vgre::GraphExecId exec = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphInstantiate(
      static_cast<vgre::GraphId>(graph), exec);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);
  *out_exec = exec;
  return VGRE_SUCCESS;
}

int vgre_graphExecDestroy(uint64_t exec) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphExecDestroy(
      static_cast<vgre::GraphExecId>(exec));
  return to_status(r);
}

int vgre_graphLaunch(uint64_t exec, uint64_t stream) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphLaunch(
      static_cast<vgre::GraphExecId>(exec),
      static_cast<vgre::StreamId>(stream));
  return to_status(r);
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
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  std::vector<vgre::ArgType> argTypes;
  if (!to_arg_types(arg_types, num_args, argTypes))
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
    return to_status(r);
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
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  std::vector<uint64_t> depList;
  depList.assign(deps, deps + num_deps);
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddMemcpyNode(
      static_cast<vgre::GraphId>(graph), dst, src, count, kind, depList,
      nodeId);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);
  *out_node_id = nodeId;
  return VGRE_SUCCESS;
}

int vgre_graphAddDependency(uint64_t graph, uint64_t node_id,
                            uint64_t depends_on) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphAddDependency(
      static_cast<vgre::GraphId>(graph), node_id, depends_on);
  return to_status(r);
}

int vgre_graphUpdateKernelNode(uint64_t graph, uint64_t node_id, void **args,
                               const uint8_t *arg_types, int num_args) {
  if (num_args < 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  std::vector<vgre::ArgType> argTypes;
  if (!to_arg_types(arg_types, num_args, argTypes))
    return VGRE_ERROR_INVALID_VALUE;
  auto r = vgre::core::RuntimeEngine::instance().graphUpdateKernelNode(
      static_cast<vgre::GraphId>(graph), node_id, args, argTypes);
  return to_status(r);
}

int vgre_graphUpdateMemcpyNode(uint64_t graph, uint64_t node_id, void *dst,
                               void *src, size_t count, int kind) {
  if (!dst || !src || count == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  auto r = vgre::core::RuntimeEngine::instance().graphUpdateMemcpyNode(
      static_cast<vgre::GraphId>(graph), node_id, dst, src, count, kind);
  return to_status(r);
}

/* ── Telemetry ──────────────────────────────────────────────────────────────
 */

int vgre_get_telemetry(vgre_telemetry_t *telemetry) {
  if (!telemetry)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  std::memset(telemetry, 0, sizeof(*telemetry));

  auto &ae = vgre::advanced::AdaptiveExecutionEngine::instance();
  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  auto &profiler = vgre::advanced::RuntimeProfiler::instance();

  telemetry->timestamp = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  telemetry->version_major = 2;
  telemetry->version_minor = 1;

  // GFLOPS
  telemetry->gflops = ae.getTotalGFLOPS();
  telemetry->max_gflops = ae.getMaxGFLOPS();
  telemetry->compute_utilization =
      (telemetry->max_gflops > 0.0)
          ? (telemetry->gflops / telemetry->max_gflops) * 100.0
          : 0.0;
  if (telemetry->compute_utilization < 0.0)
    telemetry->compute_utilization = 0.0;
  if (telemetry->compute_utilization > 100.0)
    telemetry->compute_utilization = 100.0;

  // Memory
  telemetry->memory_bandwidth_gbps = ae.getMemoryBandwidth();
  telemetry->max_memory_bandwidth_gbps = ae.getMaxMemoryBandwidth();
  telemetry->memory_bus_utilization =
      (telemetry->max_memory_bandwidth_gbps > 0.0)
          ? (telemetry->memory_bandwidth_gbps /
             telemetry->max_memory_bandwidth_gbps) *
                100.0
          : 0.0;
  if (telemetry->memory_bus_utilization < 0.0)
    telemetry->memory_bus_utilization = 0.0;
  if (telemetry->memory_bus_utilization > 100.0)
    telemetry->memory_bus_utilization = 100.0;

  telemetry->memory_used_bytes = mm.getUsedMemory();
  telemetry->memory_total_bytes = mm.getTotalMemory();

  // UVM Stats
  telemetry->total_pages = mm.getTotalMemory() / 4096;
  mm.getPageResidency(telemetry->uvm_map);

  // Normalize resident pages for the 1024-cell UI grid
  int residentCells = mm.getResidentPageCount();
  if (residentCells < 0)
    residentCells = 0;
  if (residentCells > 1024)
    residentCells = 1024;
  telemetry->resident_pages =
      (telemetry->total_pages * static_cast<uint64_t>(residentCells)) / 1024;
  if (telemetry->resident_pages > telemetry->total_pages) {
    telemetry->resident_pages = telemetry->total_pages;
  }
  telemetry->evicted_pages = telemetry->total_pages - telemetry->resident_pages;
  telemetry->page_faults_per_sec = static_cast<double>(mm.getPageFaultRate());

  // Device Stats
  auto &sched = vgre::core::Scheduler::instance();
  auto pendingTasks = sched.getPendingTasks();
  telemetry->active_kernels =
      pendingTasks > 0 ? static_cast<int64_t>(pendingTasks)
                       : static_cast<int64_t>(ae.getActiveKernelCount());
  auto threadCount = sched.getThreadCount();
  telemetry->active_threads =
      pendingTasks > 0
          ? static_cast<int64_t>(
                std::min<uint64_t>(pendingTasks,
                                   static_cast<uint64_t>(threadCount)))
          : 0;

  // Real device properties
  auto &dev = vgre::core::RuntimeEngine::instance().getDevice();
  auto props = dev.getProperties();
  telemetry->device_clock_mhz = static_cast<double>(props.clockRate) / 1000.0;

  // Real ECC reporting: DISABLED for Intel Integrated Graphics
  bool is_intel = std::string(props.name).find("Intel") != std::string::npos;
  telemetry->ecc_enabled = is_intel ? 0 : (props.major >= 7 ? 1 : 0);

  telemetry->avg_kernel_latency_ms = ae.getAvgLatencyMs();

  // If runtime profiler is enabled, prefer measured averages.
  if (profiler.isEnabled()) {
    auto stats = profiler.getAllStats();
    if (!stats.empty()) {
      double totalInv = 0.0;
      double gflops = 0.0;
      double bw = 0.0;
      double avgMs = 0.0;
      for (const auto &s : stats) {
        totalInv += static_cast<double>(s.invocations);
        gflops += s.avgGflops * static_cast<double>(s.invocations);
        bw += s.avgThroughputGBps * static_cast<double>(s.invocations);
        avgMs += s.avgTimeMs * static_cast<double>(s.invocations);
      }
      if (totalInv > 0.0) {
        telemetry->gflops = gflops / totalInv;
        telemetry->memory_bandwidth_gbps = bw / totalInv;
        telemetry->avg_kernel_latency_ms = avgMs / totalInv;
        telemetry->compute_utilization =
            (telemetry->max_gflops > 0.0)
                ? (telemetry->gflops / telemetry->max_gflops) * 100.0
                : 0.0;
        if (telemetry->compute_utilization < 0.0)
          telemetry->compute_utilization = 0.0;
        if (telemetry->compute_utilization > 100.0)
          telemetry->compute_utilization = 100.0;

        telemetry->memory_bus_utilization =
            (telemetry->max_memory_bandwidth_gbps > 0.0)
                ? (telemetry->memory_bandwidth_gbps /
                   telemetry->max_memory_bandwidth_gbps) *
                      100.0
                : 0.0;
        if (telemetry->memory_bus_utilization < 0.0)
          telemetry->memory_bus_utilization = 0.0;
        if (telemetry->memory_bus_utilization > 100.0)
          telemetry->memory_bus_utilization = 100.0;
      }
    }
  }

  // Smooth jitter for dashboard consumption (EMA).
  // Keeps UI stable without hiding trend direction.
  {
    static std::mutex s_telemetry_mutex;
    static bool s_has_prev = false;
    static double s_gflops = 0.0;
    static double s_bw = 0.0;
    static double s_latency = 0.0;
    static uint64_t s_last_ts = 0;

    std::lock_guard<std::mutex> lock(s_telemetry_mutex);
    const uint64_t ts = telemetry->timestamp;
    const double alpha = 0.35;

    if (!s_has_prev) {
      s_gflops = telemetry->gflops;
      s_bw = telemetry->memory_bandwidth_gbps;
      s_latency = telemetry->avg_kernel_latency_ms;
      s_last_ts = ts;
      s_has_prev = true;
    } else if (ts >= s_last_ts) {
      s_gflops = s_gflops * (1.0 - alpha) + telemetry->gflops * alpha;
      s_bw = s_bw * (1.0 - alpha) + telemetry->memory_bandwidth_gbps * alpha;
      s_latency =
          s_latency * (1.0 - alpha) + telemetry->avg_kernel_latency_ms * alpha;
      s_last_ts = ts;
    }

    telemetry->gflops = s_gflops;
    telemetry->memory_bandwidth_gbps = s_bw;
    telemetry->avg_kernel_latency_ms = s_latency;

    telemetry->compute_utilization =
        (telemetry->max_gflops > 0.0)
            ? (telemetry->gflops / telemetry->max_gflops) * 100.0
            : 0.0;
    if (telemetry->compute_utilization < 0.0)
      telemetry->compute_utilization = 0.0;
    if (telemetry->compute_utilization > 100.0)
      telemetry->compute_utilization = 100.0;

    telemetry->memory_bus_utilization =
        (telemetry->max_memory_bandwidth_gbps > 0.0)
            ? (telemetry->memory_bandwidth_gbps /
               telemetry->max_memory_bandwidth_gbps) *
                  100.0
            : 0.0;
    if (telemetry->memory_bus_utilization < 0.0)
      telemetry->memory_bus_utilization = 0.0;
    if (telemetry->memory_bus_utilization > 100.0)
      telemetry->memory_bus_utilization = 100.0;
  }
  telemetry->device_temperature =
      static_cast<double>(ae.getDeviceTemperature());
  telemetry->background_compute_active = static_cast<int64_t>(
      vgre::advanced::WorkloadEngine::instance().isEnabled() ? 1 : 0);

  // Add IPC aggregation if we are the master (Dashboard)
  auto &ipc = vgre::advanced::IPCManager::instance();
  if (ipc.isEnabled()) {
    // 1. Update our slot in shared memory with current local stats
    ipc.updateLocalTelemetry(*telemetry);
    // 2. Aggregate global stats (local + other processes + remote cluster)
    ipc.getGlobalTelemetry(*telemetry);
  }

  // Add TCP Cluster aggregation
  auto &tcp = vgre::advanced::TCPClusterManager::instance();
  if (tcp.isEnabled()) {
    if (tcp.isMaster()) {
      tcp.aggregateRemoteTelemetry(*telemetry);
    } else {
      tcp.broadcastLocalTelemetry(*telemetry);
    }
  }

  return VGRE_SUCCESS;
}

int vgre_get_memory_info_json(char **out_json) {
  if (!out_json)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto &mm = vgre::core::MemoryManager::instance();
  
  std::stringstream ss;
  ss << "{\"allocations\":[";
  
  bool first = true;
  for (const auto& [handle, alloc] : mm.getAllocations()) {
    if (!first) ss << ",";
    ss << "{\"ptr\":\"" << handle << "\","
       << "\"size\":" << alloc.size << ","
       << "\"managed\":" << (alloc.isManaged ? "true" : "false") << ","
       << "\"resident\":" << (alloc.isResidentOnHost ? "true" : "false") << ","
       << "\"device\":" << alloc.deviceId << "}";
    first = false;
  }
  
  ss << "],\"pools\":[";
  first = true;
  for (const auto& [id, pool] : mm.getPools()) {
    if (!first) ss << ",";
    ss << "{\"id\":" << id << ","
       << "\"blockSize\":" << pool.blockSize << ","
       << "\"total\":" << pool.totalAllocated << ","
       << "\"peak\":" << pool.peakAllocated << ","
       << "\"active\":" << pool.activeList.size() << ","
       << "\"free\":" << pool.freeList.size() << "}";
    first = false;
  }
  ss << "]}";
  
  std::string s = ss.str();
  *out_json = (char *)std::malloc(s.size() + 1);
  if (!*out_json) return VGRE_ERROR_OUT_OF_MEMORY;
  std::strcpy(*out_json, s.c_str());
  return VGRE_SUCCESS;
}

int vgre_get_logs(char ***buffer, int *count) {
  if (!buffer || !count)
    return VGRE_ERROR_INVALID_VALUE;

  auto logs = vgre::Logger::instance().getRecentLogs();
  if (logs.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return VGRE_ERROR_OUT_OF_MEMORY;
  }
  *count = static_cast<int>(logs.size());

  if (*count == 0) {
    *buffer = nullptr;
    return VGRE_SUCCESS;
  }

  char **lines = (char **)malloc(sizeof(char *) * (*count));
  if (!lines) {
    *buffer = nullptr;
    *count = 0;
    return VGRE_ERROR_OUT_OF_MEMORY;
  }
  for (int i = 0; i < *count; ++i) {
    const auto &line = logs[static_cast<size_t>(i)];
    size_t len = line.size();
    lines[i] = static_cast<char *>(malloc(len + 1));
    if (lines[i]) {
      std::memcpy(lines[i], line.c_str(), len + 1);
    }
    if (!lines[i]) {
      for (int j = 0; j < i; ++j) {
        free(lines[j]);
      }
      free(lines);
      *buffer = nullptr;
      *count = 0;
      return VGRE_ERROR_OUT_OF_MEMORY;
    }
  }

  *buffer = lines;
  return VGRE_SUCCESS;
}

void vgre_free_logs(char **buffer, int count) {
  if (!buffer)
    return;
  for (int i = 0; i < count; ++i) {
    free(buffer[i]);
  }
  free(buffer);
}

int vgre_get_profiler_json(char **out_json, int top_n) {
  if (!out_json)
    return VGRE_ERROR_INVALID_VALUE;

  auto &profiler = vgre::advanced::RuntimeProfiler::instance();
  if (!profiler.isEnabled()) {
    *out_json = nullptr;
    return VGRE_SUCCESS;
  }

  auto stats = profiler.getAllStats();
  if (top_n > 0 && static_cast<size_t>(top_n) < stats.size()) {
    stats.resize(static_cast<size_t>(top_n));
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "{\n";
  oss << "  \"timestamp_ms\": " << telemetry_now_ms() << ",\n";
  oss << "  \"total_kernels\": " << stats.size() << ",\n";
  oss << "  \"top_kernels\": [\n";

  for (size_t i = 0; i < stats.size(); ++i) {
    const auto &s = stats[i];
    oss << "    {\n";
    oss << "      \"name\": \"" << json_escape(s.kernelName) << "\",\n";
    oss << "      \"invocations\": " << s.invocations << ",\n";
    oss << "      \"total_time_ms\": " << s.totalTimeMs << ",\n";
    oss << "      \"avg_time_ms\": " << s.avgTimeMs << ",\n";
    oss << "      \"min_time_ms\": " << s.minTimeMs << ",\n";
    oss << "      \"max_time_ms\": " << s.maxTimeMs << ",\n";
      oss << "      \"avg_throughput_gbps\": " << s.avgThroughputGBps << ",\n";
      oss << "      \"avg_gflops\": " << s.avgGflops << ",\n";
      oss << "      \"source_code\": \"" << json_escape(s.sourceCode) << "\",\n";
      oss << "      \"ir_code\": \"" << json_escape(s.irCode) << "\"\n";
      oss << "    }" << (i + 1 < stats.size() ? "," : "") << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";

  const std::string json = oss.str();
  char *buf = static_cast<char *>(malloc(json.size() + 1));
  if (!buf)
    return VGRE_ERROR_OUT_OF_MEMORY;
  std::memcpy(buf, json.c_str(), json.size() + 1);
  *out_json = buf;
  return VGRE_SUCCESS;
}

int vgre_get_kernel_history_json(const char *kernel_name, char **out_json) {
  if (!kernel_name || !out_json)
    return VGRE_ERROR_INVALID_VALUE;
  
  auto &profiler = vgre::advanced::RuntimeProfiler::instance();
  auto events = profiler.getEventsByKernel(kernel_name);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "[\n";
  for (size_t i = 0; i < events.size(); ++i) {
    const auto &ev = events[i];
    oss << "  {\n";
    oss << "    \"timestamp_ms\": " << ev.timestamp_ms << ",\n";
    oss << "    \"duration_ms\": " << ev.durationMs << ",\n";
    oss << "    \"throughput_gbps\": " << ev.throughputGBps << ",\n";
    oss << "    \"gflops\": " << ev.gflops << ",\n";
    oss << "    \"threads_used\": " << ev.threadsUsed << "\n";
    oss << "  }" << (i + 1 < events.size() ? "," : "") << "\n";
  }
  oss << "]\n";

  const std::string json = oss.str();
  char *buf = static_cast<char *>(malloc(json.size() + 1));
  if (!buf) return VGRE_ERROR_OUT_OF_MEMORY;
  std::memcpy(buf, json.c_str(), json.size() + 1);
  *out_json = buf;
  return VGRE_SUCCESS;
}

void vgre_free_string(char *str) {
  if (str)
    free(str);
}

int vgre_set_profiler_enabled(int enabled) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  vgre::advanced::RuntimeProfiler::instance().setEnabled(enabled != 0);
  return VGRE_SUCCESS;
}

int vgre_set_background_compute(int enabled) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  vgre::advanced::WorkloadEngine::instance().setEnabled(enabled != 0);
  return VGRE_SUCCESS;
}

int vgre_set_service_mode(int is_master) {
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;
  if (!vgre::advanced::IPCManager::instance().initialize(is_master != 0)) {
    return VGRE_ERROR_IO;
  }
  return VGRE_SUCCESS;
}

int vgre_set_block_threads(int enabled) {
  const char *value = enabled ? "1" : "0";
#if defined(_WIN32)
  if (_putenv_s("VGRE_BLOCK_THREADS", value) != 0) {
    return VGRE_ERROR_IO;
  }
#else
  if (setenv("VGRE_BLOCK_THREADS", value, 1) != 0) {
    return VGRE_ERROR_IO;
  }
#endif
  VGRE_LOG_INFO("VGRE", std::string("VGRE_BLOCK_THREADS set to ") + value);
  return VGRE_SUCCESS;
}

// ── Phase 5: Global Compute Network ──────────────────────────────────────

int vgre_cluster_set_security(int enabled) {
  return to_status(vgre::advanced::TCPClusterManager::instance().enableSecurity(enabled != 0));
}

int vgre_cluster_get_security_info(vgre_security_info_t *info) {
  if (!info) return VGRE_ERROR_INVALID_VALUE;
  
  auto sinfo = vgre::advanced::TCPClusterManager::instance().getSecurityInfo();
  std::strncpy(info->cipher_name, sinfo.cipher_name, sizeof(info->cipher_name) - 1);
  info->cipher_name[sizeof(info->cipher_name) - 1] = '\0'; // Ensure null termination
  std::strncpy(info->key_fingerprint, sinfo.key_fingerprint, sizeof(info->key_fingerprint) - 1);
  info->key_fingerprint[sizeof(info->key_fingerprint) - 1] = '\0'; // Ensure null termination
  info->session_seconds = sinfo.session_seconds;
  info->is_encrypted = sinfo.is_encrypted ? 1 : 0;
  info->packets_sent = sinfo.packets_sent;
  info->packets_received = sinfo.packets_received;
  info->bytes_sent = sinfo.bytes_sent;
  info->bytes_received = sinfo.bytes_received;
  
  return VGRE_SUCCESS;
}

int vgre_credits_get_balance(const char *address, vgre_credit_info_t *info) {
  if (!address || !info) return VGRE_ERROR_INVALID_VALUE;
  
  vgre::advanced::NodeBalance bal;
  vgre::VGREResult r = vgre::advanced::ResourceLedger::instance().getBalance(address, bal);
  if (r != vgre::VGREResult::SUCCESS) return to_status(r);
  
  std::strncpy(info->address, bal.address.c_str(), sizeof(info->address) - 1);
  info->address[sizeof(info->address) - 1] = '\0'; // Ensure null termination
  info->total_credits = bal.total_credits;
  info->total_debits = bal.total_debits;
  info->balance = bal.balance;
  info->last_activity = bal.last_activity;
  info->transaction_count = bal.transaction_count;
  
  return VGRE_SUCCESS;
}

int vgre_credits_get_all(vgre_credit_info_t *nodes, int *count) {
  if (!count) return VGRE_ERROR_INVALID_VALUE;
  
  auto balances = vgre::advanced::ResourceLedger::instance().getAllBalances();
  int total = static_cast<int>(balances.size());
  
  if (!nodes) {
    *count = total;
    return VGRE_SUCCESS;
  }
  
  int to_fill = std::min(*count, total);
  for (int i = 0; i < to_fill; ++i) {
    std::strncpy(nodes[i].address, balances[i].address.c_str(), sizeof(nodes[i].address) - 1);
    nodes[i].address[sizeof(nodes[i].address) - 1] = '\0'; // Ensure null termination
    nodes[i].total_credits = balances[i].total_credits;
    nodes[i].total_debits = balances[i].total_debits;
    nodes[i].balance = balances[i].balance;
    nodes[i].last_activity = balances[i].last_activity;
    nodes[i].transaction_count = balances[i].transaction_count;
  }
  
  *count = to_fill;
  return VGRE_SUCCESS;
}

int vgre_credits_reset(void) {
  vgre::advanced::ResourceLedger::instance().reset();
  return VGRE_SUCCESS;
}

// ── JIT Telemetry Reporting ───────────────────────────────────────────────

extern "C" void vgre_jit_report_flops(uint64_t flops) {
  vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(flops);
  // Phase 5: Report compute time to master for billing if we are a worker
  auto &cluster = vgre::advanced::TCPClusterManager::instance();
  if (cluster.isEnabled() && cluster.isWorker()) {
     // For now, we don't have a direct GFLOP to second mapping here, 
     // but we trigger the wall-clock reporting in the handler.
  }
}

extern "C" void vgre_jit_report_memory(uint64_t bytes) {
  vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytes);
}

// ── Memory Pool C-API ─────────────────────────────────────────────────────

int vgre_pool_create(uint64_t *out_pool, size_t block_size) {
  if (!out_pool) return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().createPool(*out_pool, block_size);
  return to_status(r);
}

int vgre_pool_alloc(uint64_t pool, size_t size, void **out_ptr) {
  if (!out_ptr || size == 0) return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS) return s;
  vgre::MemoryHandle handle;
  auto r = vgre::core::MemoryManager::instance().allocateFromPool(pool, size, handle);
  if (r != vgre::VGREResult::SUCCESS) return to_status(r);
  *out_ptr = handle;
  return VGRE_SUCCESS;
}

int vgre_pool_free(uint64_t pool, void *ptr) {
  if (!ptr) return VGRE_SUCCESS;
  if (int s = require_initialized(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().freeToPool(pool, ptr);
  return to_status(r);
}

int vgre_pool_destroy(uint64_t pool) {
  if (int s = require_initialized(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().destroyPool(pool);
  return to_status(r);
}

int vgre_get_cluster_nodes(vgre_cluster_node_t *nodes, int *count) {
  if (!count) return VGRE_ERROR_INVALID_VALUE;
  
  auto &tcp = vgre::advanced::TCPClusterManager::instance();
  if (!tcp.isEnabled() || !tcp.isMaster()) {
    *count = 0;
    return VGRE_SUCCESS;
  }

  std::vector<vgre::advanced::TCPClusterManager::ClusterNodeInfo> connections;
  tcp.getConnectedNodes(connections);

  int max_count = *count;
  int actual_count = static_cast<int>(connections.size());
  *count = actual_count;

  if (!nodes || max_count <= 0) return VGRE_SUCCESS;

  int copy_count = std::min(max_count, actual_count);
  for (int i = 0; i < copy_count; ++i) {
    const auto &conn = connections[i];
    std::strncpy(nodes[i].address, conn.ip_address.c_str(), sizeof(nodes[i].address) - 1);
    nodes[i].port = 7780; // Standardized Phase 3 Port
    nodes[i].cpu_cores = conn.cpu_cores;
    nodes[i].memory_bytes = conn.cpu_memory;
    nodes[i].latency_ms = conn.last_telemetry.avg_kernel_latency_ms;
    nodes[i].available = conn.active ? 1 : 0;
    std::strncpy(nodes[i].igpu_name, conn.igpu_name, sizeof(nodes[i].igpu_name) - 1);
  }

  return VGRE_SUCCESS;
}

// ── Version Info ───────────────────────────────────────────────────────────

const char *vgre_get_version(void) { return "0.1.0"; }
