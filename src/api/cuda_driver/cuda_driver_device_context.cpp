// CUDA Driver API — cuda driver device context

#include "cuda_driver_internal.h"
#include "vgre/common/platform.h"
#include "vgre/mig/mig_manager.h"
#include <unordered_map>
#include <mutex>

extern "C" {

thread_local CUcontext g_current_ctx = nullptr;

VGRE_PUBLIC_API
CUresult cuInit(unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().init();
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuDeviceGetCount(int *count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(&count);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  std::snprintf(name, static_cast<size_t>(len), "%s", prop.name);
  return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  // When this process is pinned to a MIG instance (VGRE_MIG_DEVICE), report the
  // instance's memory budget rather than the whole device — real MIG behaviour.
  vgre::mig::MigManager::instance().configureDevice(prop.totalGlobalMem, 7, 8);
  *bytes = vgre::mig::MigManager::instance().effectiveDeviceMemory(prop.totalGlobalMem);
  return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().deviceGetAttribute(pi, attrib, dev);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().setDevice(dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(dev + 1));
  return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
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

VGRE_PUBLIC_API
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
  // Deterministic 16-byte UUID (RFC 4122 variant 2, version 4 layout):
  //   Bytes 0-3:  "VGRE" magic (0x56 0x47 0x52 0x45)
  //   Bytes 4-7:  device ordinal as big-endian uint32 — unique across >255 devices
  //   Bytes 8-11: fixed VGRE marker 0x00 0x00 0x00 0x00
  //   Bytes 12-15: reserved zeros
  memset(uuid, 0, 16);
  uuid[0] = 0x56; uuid[1] = 0x47; uuid[2] = 0x52; uuid[3] = 0x45;  // "VGRE"
  uint32_t devU = static_cast<uint32_t>(dev);
  uuid[4]  = static_cast<unsigned char>((devU >> 24) & 0xFF);
  uuid[5]  = static_cast<unsigned char>((devU >> 16) & 0xFF);
  uuid[6]  = static_cast<unsigned char>((devU >>  8) & 0xFF);
  uuid[7]  = static_cast<unsigned char>( devU        & 0xFF);
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

// ── cuFuncGetAttribute / cuFuncSetAttribute ────────────────────────────────────
// Driver-level function attribute query/set. Mirrors the runtime-level
// cudaFuncGetAttributes / cudaFuncSetAttribute; here CUfunction is the
// JIT-compiled function handle (CUfunction = vgre::api::CUfunction).

enum CUfunction_attribute {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK       = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES           = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES            = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES            = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS                    = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION                 = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION              = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA               = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET    = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH      = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT     = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH      = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
};

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction /*hfunc*/) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  switch (attrib) {
  case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:    *pi = 1024;  break;
  case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:         *pi = 0;     break;
  case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:          *pi = 0;     break;
  case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:          *pi = 0;     break;
  case CU_FUNC_ATTRIBUTE_NUM_REGS:                  *pi = 32;    break;
  case CU_FUNC_ATTRIBUTE_PTX_VERSION:               *pi = 86;    break;
  case CU_FUNC_ATTRIBUTE_BINARY_VERSION:            *pi = 86;    break;
  case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:             *pi = 0;     break;
  case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *pi = 49152; break;
  case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: *pi = 0; break;
  default: *pi = 0; break;
  }
  return CUDA_SUCCESS;
}

CUresult cuFuncSetAttribute(CUfunction /*hfunc*/, int attrib, int val) {
  (void)val;
  // Only MAX_DYNAMIC_SHARED_SIZE_BYTES has effect in VGRE; the runtime-level
  // cudaFuncSetAttribute stores it per host-function pointer.  The driver-level
  // CUfunction handle does not currently carry a separate map, but accepting
  // silently is correct (runtime path already handles the value).
  switch (attrib) {
  case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
  case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
  case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
  case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
  case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH:
  case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED:
  case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE:
    return CUDA_SUCCESS;
  default:
    return CUDA_ERROR_INVALID_VALUE;
  }
}

} // extern "C"
