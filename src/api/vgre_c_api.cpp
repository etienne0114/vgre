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
#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstring>

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
  default:
    return VGRE_ERROR_GENERIC;
  }
}

// ── Initialization ─────────────────────────────────────────────────────────

int vgre_init(void) {
  auto result = vgre::core::RuntimeEngine::instance().initialize();
  if (result == vgre::VGREResult::SUCCESS) {
    // Try to attach to global session as client by default
    vgre::advanced::IPCManager::instance().initialize(false);
  }
  return static_cast<int>(result);
}

int vgre_shutdown(void) {
  auto r = vgre::core::RuntimeEngine::instance().shutdown();
  return to_status(r);
}

// ── Device Management ──────────────────────────────────────────────────────

int vgre_get_device_count(int *count) {
  if (!count)
    return VGRE_ERROR_INVALID_VALUE;
  *count = vgre::core::RuntimeEngine::instance().getDeviceCount();
  return VGRE_SUCCESS;
}

int vgre_set_device(int device_id) {
  auto r = vgre::core::RuntimeEngine::instance().setDevice(device_id);
  return to_status(r);
}

int vgre_get_device(int *device_id) {
  if (!device_id)
    return VGRE_ERROR_INVALID_VALUE;
  *device_id = vgre::core::RuntimeEngine::instance().getDeviceId();
  return VGRE_SUCCESS;
}

int vgre_get_device_properties(int device_id, vgre_device_properties_t *props) {
  if (!props)
    return VGRE_ERROR_INVALID_VALUE;

  vgre::DeviceProperties dp;
  auto r =
      vgre::core::RuntimeEngine::instance().getDeviceProperties(device_id, dp);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

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
  auto r = vgre::core::RuntimeEngine::instance().synchronize();
  return to_status(r);
}

// ── Memory Management ──────────────────────────────────────────────────────

int vgre_malloc(void **ptr, size_t size) {
  if (!ptr || size == 0)
    return VGRE_ERROR_INVALID_VALUE;

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
  auto r = vgre::core::RuntimeEngine::instance().getMemoryManager().free(ptr);
  return to_status(r);
}

int vgre_memcpy(void *dst, const void *src, size_t count, int direction) {
  if (!dst || !src || count == 0)
    return VGRE_ERROR_INVALID_VALUE;

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
  std::memset(ptr, value, count);
  return VGRE_SUCCESS;
}

int vgre_device_can_access_peer(int *can_access, int device, int peer_device) {
  auto r = vgre::core::RuntimeEngine::instance().deviceCanAccessPeer(
      device, peer_device, can_access);
  return to_status(r);
}

int vgre_device_enable_peer_access(int peer_device) {
  auto r =
      vgre::core::RuntimeEngine::instance().deviceEnablePeerAccess(peer_device);
  return to_status(r);
}

int vgre_device_disable_peer_access(int peer_device) {
  auto r = vgre::core::RuntimeEngine::instance().deviceDisablePeerAccess(
      peer_device);
  return to_status(r);
}

// ── Kernel Registration & Launch ───────────────────────────────────────────

int vgre_register_kernel(const char *name, const char *source,
                         uint64_t *out_kernel_id) {
  if (!name || !source || !out_kernel_id)
    return VGRE_ERROR_INVALID_VALUE;

  vgre::KernelId kid;
  auto r = vgre::core::RuntimeEngine::instance().registerKernel(
      std::string(name), std::string(source), kid);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *out_kernel_id = kid;
  return VGRE_SUCCESS;
}

int vgre_launch_kernel(uint64_t kernel_id, const uint32_t grid_dim[3],
                       const uint32_t block_dim[3], void **args, int num_args,
                       size_t shared_mem, uint64_t stream_id) {
  (void)num_args; // args count is for callers' bookkeeping

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

  vgre::StreamId id;
  auto r = vgre::core::RuntimeEngine::instance().getDevice().createStream(
      id, priority);
  if (r != vgre::VGREResult::SUCCESS)
    return to_status(r);

  *out_stream_id = id;
  return VGRE_SUCCESS;
}

int vgre_stream_synchronize(uint64_t stream_id) {
  auto r = vgre::core::RuntimeEngine::instance().streamSynchronize(
      static_cast<vgre::StreamId>(stream_id));
  return to_status(r);
}

int vgre_stream_destroy(uint64_t stream_id) {
  auto r = vgre::core::RuntimeEngine::instance().getDevice().destroyStream(
      static_cast<vgre::StreamId>(stream_id));
  return to_status(r);
}

/* ── Telemetry ──────────────────────────────────────────────────────────────
 */

int vgre_get_telemetry(vgre_telemetry_t *telemetry) {
  if (!telemetry)
    return VGRE_ERROR_INVALID_VALUE;

  auto &ae = vgre::advanced::AdaptiveExecutionEngine::instance();
  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();

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

  // Memory
  telemetry->memory_bandwidth_gbps = ae.getMemoryBandwidth();
  telemetry->max_memory_bandwidth_gbps = ae.getMaxMemoryBandwidth();
  telemetry->memory_bus_utilization =
      (telemetry->max_memory_bandwidth_gbps > 0.0)
          ? (telemetry->memory_bandwidth_gbps /
             telemetry->max_memory_bandwidth_gbps) *
                100.0
          : 0.0;

  telemetry->memory_used_bytes = mm.getUsedMemory();
  telemetry->memory_total_bytes = mm.getTotalMemory();

  // UVM Stats
  telemetry->total_pages = mm.getTotalMemory() / 4096;
  mm.getPageResidency(telemetry->uvm_map);

  // Normalize resident pages for the 1024-cell UI grid
  int residentCells = mm.getResidentPageCount(); // 0-1024
  telemetry->resident_pages = (telemetry->total_pages * residentCells) / 1024;
  telemetry->evicted_pages = telemetry->total_pages - telemetry->resident_pages;
  telemetry->page_faults_per_sec = static_cast<double>(mm.getPageFaultRate());

  // Device Stats
  telemetry->active_kernels = static_cast<int64_t>(ae.getActiveKernelCount());
  telemetry->active_threads =
      static_cast<int64_t>(vgre::core::Scheduler::instance().getThreadCount());

  // Real device properties
  auto &dev = vgre::core::RuntimeEngine::instance().getDevice();
  auto props = dev.getProperties();
  telemetry->device_clock_mhz = static_cast<double>(props.clockRate) / 1000.0;

  // Real ECC reporting: DISABLED for Intel Integrated Graphics
  bool is_intel = std::string(props.name).find("Intel") != std::string::npos;
  telemetry->ecc_enabled = is_intel ? 0 : (props.major >= 7 ? 1 : 0);

  telemetry->avg_kernel_latency_ms = ae.getAvgLatencyMs();
  telemetry->device_temperature =
      static_cast<double>(ae.getDeviceTemperature());
  telemetry->background_compute_active = static_cast<int64_t>(
      vgre::advanced::WorkloadEngine::instance().isEnabled() ? 1 : 0);

  // Add IPC aggregation if we are the master (Dashboard)
  auto &ipc = vgre::advanced::IPCManager::instance();
  if (ipc.isEnabled()) {
    ipc.updateLocalTelemetry(*telemetry);
    ipc.getGlobalTelemetry(*telemetry);
  }

  return VGRE_SUCCESS;
}

int vgre_get_logs(char ***buffer, int *count) {
  if (!buffer || !count)
    return VGRE_ERROR_INVALID_VALUE;

  auto logs = vgre::Logger::instance().getRecentLogs();
  *count = static_cast<int>(logs.size());

  if (*count == 0) {
    *buffer = nullptr;
    return VGRE_SUCCESS;
  }

  char **lines = (char **)malloc(sizeof(char *) * (*count));
  for (int i = 0; i < *count; ++i) {
    lines[i] = strdup(logs[i].c_str());
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

int vgre_set_background_compute(int enabled) {
  vgre::advanced::WorkloadEngine::instance().setEnabled(enabled != 0);
  return VGRE_SUCCESS;
}

int vgre_set_service_mode(int is_master) {
  vgre::advanced::IPCManager::instance().initialize(is_master != 0);
  return VGRE_SUCCESS;
}

// ── Version Info ───────────────────────────────────────────────────────────

const char *vgre_get_version(void) { return "0.1.0"; }
