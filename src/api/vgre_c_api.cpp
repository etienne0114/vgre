/**
 * VGRE C API — Implementation
 *
 * Delegates all calls to the C++ RuntimeEngine, MemoryManager, and Scheduler
 * singletons. This file is compiled into libvgre.so and provides the flat
 * extern "C" interface that Python (ctypes) and other FFI consumers use.
 */

#include "vgre/api/vgre_c_api.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/runtime/gpu_cache.h"
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
#include <thread>
#include <vector>
#if defined(__linux__)
#include <malloc.h>
#endif

// ── Helper: convert VGREResult to C status code ────────────────────────────
static int to_status(vgre::VGREResult r) {
  switch (r) {
  case vgre::VGREResult::SUCCESS:
    return VGRE_SUCCESS;
  case vgre::VGREResult::ERR_OUT_OF_MEMORY:
    return VGRE_ERROR_OUT_OF_MEMORY;
  case vgre::VGREResult::ERR_INVALID_VALUE:
    return VGRE_ERROR_INVALID_VALUE;
  case vgre::VGREResult::ERR_INVALID_KERNEL:
    return VGRE_ERROR_INVALID_KERNEL;
  case vgre::VGREResult::ERR_LAUNCH_FAILURE:
    return VGRE_ERROR_LAUNCH_FAILURE;
  case vgre::VGREResult::ERR_IO:
    return VGRE_ERROR_IO;
  case vgre::VGREResult::ERR_NOT_INITIALIZED:
    return VGRE_ERROR_NOT_INIT;
  case vgre::VGREResult::ERR_AUTH_FAILED:
    return VGRE_ERROR_AUTH_FAILED;
  case vgre::VGREResult::ERR_CRYPTO:
    return VGRE_ERROR_CRYPTO;
  case vgre::VGREResult::ERR_INVALID_DEVICE:
    return VGRE_ERROR_INVALID_VALUE; // Map to invalid value for now, or add specific C code
  default:
    return VGRE_ERROR_GENERIC;
  }
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

static int require_initialized() {
  if (!vgre::core::RuntimeEngine::instance().isInitialized()) {
    return VGRE_ERROR_NOT_INIT;
  }
  return VGRE_SUCCESS;
}

// ── Initialization ─────────────────────────────────────────────────────────

static std::mutex g_init_mutex;
static bool g_initialized = false;

int vgre_init(void) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    return VGRE_SUCCESS;
  }

  auto result = vgre::core::RuntimeEngine::instance().initialize();
  if (result == vgre::VGREResult::SUCCESS) {
    g_initialized = true;
  }
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
  // flags=2 (cudaMemAttachHost semantic) maps memory R/W immediately so
  // application code can write without triggering the SIGSEGV handler.
  auto r = vgre::core::RuntimeEngine::instance().mallocManaged(size, handle, 2);
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

  vgre::KernelId kid = 0;
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

// ── Async Memory Pool C-API ───────────────────────────────────────────────

int vgre_malloc_async(void **ptr, size_t size, uint64_t pool, uint64_t stream) {
  (void)stream; // Pool alloc is synchronous, submitted to stream context
  if (!ptr || size == 0)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  vgre::MemoryHandle handle;
  auto r =
      vgre::core::RuntimeEngine::instance().getMemoryManager().allocateFromPool(
          static_cast<vgre::core::PoolHandle>(pool), size, handle);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *ptr = handle;
  return VGRE_SUCCESS;
}

int vgre_free_async(void *ptr, uint64_t pool, uint64_t stream) {
  (void)stream;
  if (!ptr)
    return VGRE_SUCCESS; // freeing NULL is valid
  if (int s = require_initialized(); s != VGRE_SUCCESS)
    return s;

  auto r =
      vgre::core::RuntimeEngine::instance().getMemoryManager().freeToPool(
          static_cast<vgre::core::PoolHandle>(pool), ptr);
  return to_status(r);
}

// ── Version Info ───────────────────────────────────────────────────────────

const char *vgre_get_version(void) { return "0.1.2"; }
