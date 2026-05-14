// CUDA Driver API — cuda driver device context

#include "cuda_driver_internal.h"
#include <unordered_map>
#include <mutex>

extern "C" {

CUresult cuInit(unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().init();
  return toCU(err);
}

CUresult cuDeviceGetCount(int *count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
  return toCU(err);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(&count);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  std::snprintf(name, static_cast<size_t>(len), "%s", prop.name);
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *bytes = prop.totalGlobalMem;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().deviceGetAttribute(pi, attrib, dev);
  return toCU(err);
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().setDevice(dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(dev + 1));
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (g_current_ctx == ctx) g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  if (ctx) {
    int dev = static_cast<int>(reinterpret_cast<uintptr_t>(ctx)) - 1;
    vgre::api::CUDAInterceptor::instance().setDevice(dev);
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  auto err = vgre::api::CUDAInterceptor::instance().deviceSynchronize();
  return toCU(err);
}

// ── Context queries ─────────────────────────────────────────────────────────

CUresult cuCtxGetDevice(CUdevice *device) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  if (!g_current_ctx) return CUDA_ERROR_NOT_INITIALIZED;
  *device = static_cast<int>(reinterpret_cast<uintptr_t>(g_current_ctx)) - 1;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetFlags(unsigned int *flags) {
  if (!flags) return CUDA_ERROR_INVALID_VALUE;
  *flags = 0; // VGRE uses default flags (CU_CTX_SCHED_AUTO)
  return CUDA_SUCCESS;
}

// Static device limit storage (per-context limits tracked by device ordinal)
static std::unordered_map<int, std::unordered_map<int, size_t>> g_ctxLimits;
static std::mutex g_ctxLimitMu;

static size_t getDefaultLimit(int limit) {
  switch (limit) {
    case 0x00: return 1024 * 1024;          // CU_LIMIT_STACK_SIZE
    case 0x01: return 64 * 1024 * 1024;     // CU_LIMIT_PRINTF_FIFO_SIZE
    case 0x02: return 128 * 1024 * 1024;   // CU_LIMIT_MALLOC_HEAP_SIZE
    case 0x03: return 64 * 1024;           // CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
    case 0x04: return 2048;                // CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
    default:   return 0;
  }
}

CUresult cuCtxGetLimit(size_t *pvalue, int limit) {
  if (!pvalue) return CUDA_ERROR_INVALID_VALUE;
  if (!g_current_ctx) return CUDA_ERROR_NOT_INITIALIZED;
  int dev = static_cast<int>(reinterpret_cast<uintptr_t>(g_current_ctx)) - 1;
  std::lock_guard<std::mutex> lk(g_ctxLimitMu);
  auto &devMap = g_ctxLimits[dev];
  auto it = devMap.find(limit);
  if (it != devMap.end()) {
    *pvalue = it->second;
  } else {
    *pvalue = getDefaultLimit(limit);
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxSetLimit(int limit, size_t value) {
  if (!g_current_ctx) return CUDA_ERROR_NOT_INITIALIZED;
  int dev = static_cast<int>(reinterpret_cast<uintptr_t>(g_current_ctx)) - 1;
  std::lock_guard<std::mutex> lk(g_ctxLimitMu);
  g_ctxLimits[dev][limit] = value;
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCacheConfig(int *pconfig) {
  if (!pconfig) return CUDA_ERROR_INVALID_VALUE;
  *pconfig = 0; // CU_FUNC_CACHE_PREFER_NONE
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCacheConfig(int config) {
  (void)config;
  return CUDA_SUCCESS; // No-op in CPU emulation
}

CUresult cuCtxGetSharedMemConfig(int *pConfig) {
  if (!pConfig) return CUDA_ERROR_INVALID_VALUE;
  *pConfig = 0; // CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE (4 bytes)
  return CUDA_SUCCESS;
}

CUresult cuCtxSetSharedMemConfig(int config) {
  (void)config;
  return CUDA_SUCCESS; // No-op in CPU emulation
}

CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  auto err = vgre::api::CUDAInterceptor::instance().deviceGetStreamPriorityRange(
      leastPriority, greatestPriority);
  return toCU(err);
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId) {
  if (!ctxId) return CUDA_ERROR_INVALID_VALUE;
  // Context ID = device ordinal + 1 encoded into the pointer value
  *ctxId = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ctx));
  return CUDA_SUCCESS;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  (void)ctx;
  if (!version) return CUDA_ERROR_INVALID_VALUE;
  *version = 11080; // Report CUDA 11.8 driver API version
  return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxDetach(CUcontext ctx) {
  (void)ctx;
  return CUDA_SUCCESS;
}

// ── Device queries ─────────────────────────────────────────────────────────

CUresult cuDeviceGetUuid(unsigned char *uuid, CUdevice dev) {
  if (!uuid) return CUDA_ERROR_INVALID_VALUE;
  // Generate deterministic UUID from device ordinal
  // Format: GPU-VGRE-XXXX-0000-0000-0000-000000000XXX
  static const unsigned char kVgreUuidPrefix[12] = {
      0x47, 0x50, 0x55, 0x2d, 0x56, 0x47, 0x52, 0x45,
      0x2d, 0x30, 0x30, 0x30
  };
  std::memcpy(uuid, kVgreUuidPrefix, 12);
  uuid[12] = 0x00; uuid[13] = 0x00;
  uuid[14] = 0x00; uuid[15] = static_cast<unsigned char>(dev);
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidth,
                                            int fmt, int numChannels,
                                            CUdevice dev) {
  (void)dev;
  if (!maxWidth) return CUDA_ERROR_INVALID_VALUE;
  // Conservative limit for CPU emulation
  size_t bytesPerElement = 0;
  switch (fmt) {
    case 0x01: bytesPerElement = 1; break;  // 8-bit unsigned
    case 0x02: bytesPerElement = 2; break;  // 16-bit unsigned
    case 0x03: bytesPerElement = 4; break;  // 32-bit unsigned
    case 0x20: bytesPerElement = 4; break;  // float
    default:   bytesPerElement = 4; break;
  }
  *maxWidth = (1024ULL * 1024 * 1024) / (bytesPerElement * std::max(1, numChannels));
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetP2PAttribute(int *value, int attrib, int srcDevice, int dstDevice) {
  (void)srcDevice; (void)dstDevice;
  if (!value) return CUDA_ERROR_INVALID_VALUE;
  // In single-node CPU emulation, P2P is always possible between any devices
  if (attrib == 1) { // CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK
    *value = 0; // Best rank (same node)
  } else if (attrib == 2) { // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
    *value = 1; // Supported
  } else if (attrib == 3) { // CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
    *value = 0; // Not supported in CPU emulation
  } else {
    *value = 0;
  }
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetGraphMemAttribute(int device, int attr, void *value) {
  (void)device;
  if (!value) return CUDA_ERROR_INVALID_VALUE;
  // Return conservative defaults for graph memory pool attributes
  switch (attr) {
    case 0: *static_cast<size_t*>(value) = 256 * 1024 * 1024; break; // used
    case 1: *static_cast<size_t*>(value) = 512 * 1024 * 1024; break; // reserved
    default: *static_cast<size_t*>(value) = 0; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuDeviceSetGraphMemAttribute(int device, int attr, void *value) {
  (void)device; (void)attr; (void)value;
  return CUDA_SUCCESS; // No-op: CPU emulation doesn't pool graph memory
}

CUresult cuDeviceFlushGPUDirectRDMAWrites(int scope) {
  (void)scope;
  return CUDA_SUCCESS; // No-op: no RDMA in CPU emulation
}

} // extern "C"
