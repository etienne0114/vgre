/**
 * VGRE CUDART Shim — Device & Function Attributes
 *
 * P1.21: cudaFuncGetAttributes, cudaFuncSetCacheConfig, cudaFuncSetSharedMemConfig,
 *         cudaDeviceGetLimit / SetLimit, cudaDeviceGetCacheConfig / SetCacheConfig,
 *         cudaDeviceGetSharedMemConfig / SetSharedMemConfig, cudaChooseDevice,
 *         cudaThreadExit / Synchronize / SetLimit / GetLimit,
 *         cudaDeviceGetP2PAttribute, cudaDeviceFlushGPUDirectRDMAWrites,
 *         cudaDeviceGetGraphMemAttribute / SetGraphMemAttribute,
 *         cudaLaunchKernelExC / cudaLaunchConfig.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

using namespace vgre::api;

// cudaErrorUnsupportedLimit = 42 (matches CUDA headers)
static constexpr cudaError_t cudaErrorUnsupportedLimit = 42;

// ── cudaFuncAttributes ────────────────────────────────────────────────────────

struct cudaFuncAttributes {
    size_t sharedSizeBytes;      // static shared memory per block
    size_t constSizeBytes;       // constant memory usage
    size_t localSizeBytes;       // local memory per thread
    int    maxThreadsPerBlock;   // max block size
    int    numRegs;              // registers per thread
    int    ptxVersion;           // PTX version (compute capability minor*10+major)
    int    binaryVersion;        // cubin version
    int    cacheModeCA;          // cache mode set for function
    int    maxDynamicSharedSizeBytes; // max allowed dynamic smem per block
    int    preferredShmemCarveout;    // shared memory / L1 split preference
};

// Forward declarations from cudart_shim.cpp (same library).
std::string vgre_lookup_kernel_name(const void *hostFn);

// ── Per-device limit tracking ─────────────────────────────────────────────────

namespace {

struct DeviceLimits {
    size_t stackSize         = 8192;       // cudaLimitStackSize
    size_t printfFifoSize    = 1 << 20;   // cudaLimitPrintfFifoSize (1 MB)
    size_t mallocHeapSize    = 8 << 20;   // cudaLimitMallocHeapSize (8 MB)
    size_t devRuntimeSyncDepth = 2;       // cudaLimitDevRuntimeSyncDepth
    size_t devRuntimePendingLaunchCount = 2048; // cudaLimitDevRuntimePendingLaunchCount
    size_t maxL2FetchGranularity = 128;   // cudaLimitMaxL2FetchGranularity
    size_t persistingL2CacheSize = 0;     // cudaLimitPersistingL2CacheSize
};

std::mutex g_limitMu;
DeviceLimits g_limits[4];  // indexed by device id

// cudaLimit enum values (matching CUDA headers)
constexpr int cudaLimitStackSize                        = 0x00;
constexpr int cudaLimitPrintfFifoSize                   = 0x01;
constexpr int cudaLimitMallocHeapSize                   = 0x02;
constexpr int cudaLimitDevRuntimeSyncDepth              = 0x03;
constexpr int cudaLimitDevRuntimePendingLaunchCount     = 0x04;
constexpr int cudaLimitMaxL2FetchGranularity            = 0x05;
constexpr int cudaLimitPersistingL2CacheSize            = 0x06;

// cudaCacheConfig enum
constexpr int cudaFuncCachePreferNone   = 0;
constexpr int cudaFuncCachePreferShared = 1;
constexpr int cudaFuncCachePreferL1     = 2;
constexpr int cudaFuncCachePreferEqual  = 3;

// cudaSharedMemConfig enum
constexpr int cudaSharedMemBankSizeDefault  = 0;
constexpr int cudaSharedMemBankSizeFourByte = 1;
constexpr int cudaSharedMemBankSizeEightByte = 2;

// Per-device cache and shared-mem config (not meaningfully enforced on CPU)
int g_cacheConfig[4]     = {cudaFuncCachePreferNone, cudaFuncCachePreferNone,
                             cudaFuncCachePreferNone, cudaFuncCachePreferNone};
int g_sharedMemConfig[4] = {cudaSharedMemBankSizeDefault, cudaSharedMemBankSizeDefault,
                             cudaSharedMemBankSizeDefault, cudaSharedMemBankSizeDefault};

// Per-device graph memory pool attribute: graph pool reserved memory (bytes).
std::unordered_map<int, size_t> g_graphMemReserved;
std::unordered_map<int, size_t> g_graphMemUsed;

size_t *getLimitField(int device, int limit) {
    if (device < 0 || device >= 4) return nullptr;
    auto &l = g_limits[device];
    switch (limit) {
    case cudaLimitStackSize:                    return &l.stackSize;
    case cudaLimitPrintfFifoSize:               return &l.printfFifoSize;
    case cudaLimitMallocHeapSize:               return &l.mallocHeapSize;
    case cudaLimitDevRuntimeSyncDepth:          return &l.devRuntimeSyncDepth;
    case cudaLimitDevRuntimePendingLaunchCount: return &l.devRuntimePendingLaunchCount;
    case cudaLimitMaxL2FetchGranularity:        return &l.maxL2FetchGranularity;
    case cudaLimitPersistingL2CacheSize:        return &l.persistingL2CacheSize;
    default: return nullptr;
    }
}

} // namespace

// ── cudaFuncGetAttributes ─────────────────────────────────────────────────────

extern "C" cudaError_t cudaFuncGetAttributes(
        cudaFuncAttributes *attr, const void *func) {
    if (!attr || !func) return cudaErrorInvalidValue;

    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;

    std::memset(attr, 0, sizeof(*attr));
    attr->maxThreadsPerBlock       = 1024;
    attr->ptxVersion               = 86;  // SM 8.6 (Ampere)
    attr->binaryVersion            = 86;

    // Attempt to query real kernel metadata from KernelIR.
    std::string kname = vgre_lookup_kernel_name(func);
    if (!kname.empty()) {
        vgre::KernelId kid = re.lookupKernelIdByName(kname.c_str());
        if (kid != 0) {
            const vgre::KernelIR *ir = re.getKernelIR(kid);
            if (ir) {
                attr->sharedSizeBytes  = ir->sharedMemSize;
                attr->numRegs          = (ir->registersPerThread > 0)
                                         ? ir->registersPerThread : 32;
                // maxThreadsPerBlock: not stored in KernelIR directly;
                // use the device's hardware maximum.
                attr->maxThreadsPerBlock = 1024;
            }
        }
    }
    attr->maxDynamicSharedSizeBytes = 49152;  // 48 KB default
    attr->preferredShmemCarveout    = 0;      // driver default
    return cudaSuccess;
}

// ── cudaFuncSetCacheConfig / cudaFuncSetSharedMemConfig ───────────────────────
// On CPU these are no-ops; we record the requested configuration so queries return it.

extern "C" cudaError_t cudaFuncSetCacheConfig(
        const void *func, int cacheConfig) {
    (void)func;
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    if (device >= 0 && device < 4) {
        std::lock_guard<std::mutex> lk(g_limitMu);
        g_cacheConfig[device] = cacheConfig;
    }
    return cudaSuccess;
}

extern "C" cudaError_t cudaFuncSetSharedMemConfig(
        const void *func, int config) {
    (void)func;
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    if (device >= 0 && device < 4) {
        std::lock_guard<std::mutex> lk(g_limitMu);
        g_sharedMemConfig[device] = config;
    }
    return cudaSuccess;
}

// ── cudaDeviceGetLimit / SetLimit ─────────────────────────────────────────────

extern "C" cudaError_t cudaDeviceGetLimit(size_t *pValue, int limit) {
    if (!pValue) return cudaErrorInvalidValue;
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    std::lock_guard<std::mutex> lk(g_limitMu);
    size_t *f = getLimitField(device, limit);
    if (!f) return cudaErrorUnsupportedLimit;
    *pValue = *f;
    return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceSetLimit(int limit, size_t value) {
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    std::lock_guard<std::mutex> lk(g_limitMu);
    size_t *f = getLimitField(device, limit);
    if (!f) return cudaErrorUnsupportedLimit;
    *f = value;
    return cudaSuccess;
}

// ── cudaThreadSetLimit / GetLimit (legacy 1.x wrappers) ───────────────────────

extern "C" cudaError_t cudaThreadSetLimit(int limit, size_t value) {
    return cudaDeviceSetLimit(limit, value);
}

extern "C" cudaError_t cudaThreadGetLimit(size_t *pValue, int limit) {
    return cudaDeviceGetLimit(pValue, limit);
}

// ── cudaDeviceGetCacheConfig / SetCacheConfig ─────────────────────────────────

extern "C" cudaError_t cudaDeviceGetCacheConfig(int *pCacheConfig) {
    if (!pCacheConfig) return cudaErrorInvalidValue;
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    std::lock_guard<std::mutex> lk(g_limitMu);
    *pCacheConfig = (device >= 0 && device < 4) ? g_cacheConfig[device]
                                                 : cudaFuncCachePreferNone;
    return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceSetCacheConfig(int cacheConfig) {
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    if (device >= 0 && device < 4) {
        std::lock_guard<std::mutex> lk(g_limitMu);
        g_cacheConfig[device] = cacheConfig;
    }
    return cudaSuccess;
}

// ── cudaDeviceGetSharedMemConfig / SetSharedMemConfig ────────────────────────

extern "C" cudaError_t cudaDeviceGetSharedMemConfig(int *pConfig) {
    if (!pConfig) return cudaErrorInvalidValue;
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    std::lock_guard<std::mutex> lk(g_limitMu);
    *pConfig = (device >= 0 && device < 4) ? g_sharedMemConfig[device]
                                            : cudaSharedMemBankSizeDefault;
    return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceSetSharedMemConfig(int config) {
    int device = 0;
    CUDAInterceptor::instance().getDevice(&device);
    if (device >= 0 && device < 4) {
        std::lock_guard<std::mutex> lk(g_limitMu);
        g_sharedMemConfig[device] = config;
    }
    return cudaSuccess;
}

// ── cudaChooseDevice ──────────────────────────────────────────────────────────
// Selects the device whose properties best match the requested properties.
// VGRE only has one virtual device (id=0), so we always return 0.

extern "C" cudaError_t cudaChooseDevice(int *device,
                                         const cudaDeviceProp_t *prop) {
    if (!device || !prop) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

// ── cudaThreadExit / Synchronize (legacy) ────────────────────────────────────

extern "C" cudaError_t cudaThreadExit(void) {
    // Equivalent to cudaDeviceReset() for the calling thread.
    return CUDAInterceptor::instance().deviceReset();
}

extern "C" cudaError_t cudaThreadSynchronize(void) {
    return CUDAInterceptor::instance().deviceSynchronize();
}

// ── cudaDeviceGetP2PAttribute ─────────────────────────────────────────────────

// cudaDevP2PAttr enum values
constexpr int cudaDevP2PAttrPerformanceRank              = 1;
constexpr int cudaDevP2PAttrAccessSupported              = 2;
constexpr int cudaDevP2PAttrNativeAtomicSupported        = 3;
constexpr int cudaDevP2PAttrCudaArrayAccessSupported     = 4;

extern "C" cudaError_t cudaDeviceGetP2PAttribute(
        int *value, int attr, int srcDevice, int dstDevice) {
    if (!value) return cudaErrorInvalidValue;
    int count = 0;
    CUDAInterceptor::instance().getDeviceCount(&count);
    if (srcDevice < 0 || srcDevice >= count ||
        dstDevice < 0 || dstDevice >= count)
        return cudaErrorInvalidDevice;

    switch (attr) {
    case cudaDevP2PAttrPerformanceRank:
        *value = 0;   // all devices have the same rank in CPU emulation
        break;
    case cudaDevP2PAttrAccessSupported: {
        // Check if peer access is enabled via MemoryManager.
        auto &re = vgre::core::RuntimeEngine::instance();
        int can = 0;
        re.deviceCanAccessPeer(srcDevice, dstDevice, &can);
        *value = can;
        break;
    }
    case cudaDevP2PAttrNativeAtomicSupported:
    case cudaDevP2PAttrCudaArrayAccessSupported:
        *value = 0;   // conservative: not supported in CPU model
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// ── cudaDeviceFlushGPUDirectRDMAWrites ────────────────────────────────────────
// RDMA is not applicable in CPU emulation; this is a no-op.

extern "C" cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(
        int target, int scope) {
    (void)target; (void)scope;
    if (!vgre::core::RuntimeEngine::instance().isInitialized())
        return cudaErrorNotInitialized;
    return cudaSuccess;
}

// ── cudaDeviceGetGraphMemAttribute / SetGraphMemAttribute ─────────────────────

// cudaGraphMemAttr enum values
constexpr int cudaGraphMemAttrUsedMemCurrent  = 0;
constexpr int cudaGraphMemAttrUsedMemHigh     = 1;
constexpr int cudaGraphMemAttrReservedMemCurrent = 2;
constexpr int cudaGraphMemAttrReservedMemHigh = 3;

extern "C" cudaError_t cudaDeviceGetGraphMemAttribute(
        int device, int attr, void *value) {
    if (!value) return cudaErrorInvalidValue;
    int count = 0;
    CUDAInterceptor::instance().getDeviceCount(&count);
    if (device < 0 || device >= count) return cudaErrorInvalidDevice;

    std::lock_guard<std::mutex> lk(g_limitMu);
    uint64_t v = 0;
    switch (attr) {
    case cudaGraphMemAttrUsedMemCurrent:
    case cudaGraphMemAttrUsedMemHigh: {
        auto it = g_graphMemUsed.find(device);
        v = (it != g_graphMemUsed.end()) ? it->second : 0;
        break;
    }
    case cudaGraphMemAttrReservedMemCurrent:
    case cudaGraphMemAttrReservedMemHigh: {
        auto it = g_graphMemReserved.find(device);
        v = (it != g_graphMemReserved.end()) ? it->second : 0;
        break;
    }
    default:
        return cudaErrorInvalidValue;
    }
    *static_cast<uint64_t *>(value) = v;
    return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceSetGraphMemAttribute(
        int device, int attr, void *value) {
    if (!value) return cudaErrorInvalidValue;
    int count = 0;
    CUDAInterceptor::instance().getDeviceCount(&count);
    if (device < 0 || device >= count) return cudaErrorInvalidDevice;

    uint64_t v = *static_cast<uint64_t *>(value);
    std::lock_guard<std::mutex> lk(g_limitMu);
    switch (attr) {
    case cudaGraphMemAttrReservedMemCurrent:
    case cudaGraphMemAttrReservedMemHigh:
        g_graphMemReserved[device] = v;
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

// ── cudaLaunchKernelExC / cudaLaunchConfig ─────────────────────────────────────
// Extended kernel launch with configuration struct.  Converts to the existing
// launch path.

struct cudaLaunchConfig_t {
    struct { unsigned int x, y, z; } gridDim;
    struct { unsigned int x, y, z; } blockDim;
    unsigned int  dynamicSmemBytes;
    cudaStream_t  stream;
    void         *attrs;     // cudaLaunchAttribute array (unused in VGRE)
    unsigned int  numAttrs;
};

std::string vgre_lookup_kernel_source(const void *hostFn);

extern "C" cudaError_t cudaLaunchKernelExC(
        const cudaLaunchConfig_t *config,
        const void *func,
        void **args) {
    if (!config || !func) return cudaErrorInvalidValue;

    std::string kname  = vgre_lookup_kernel_name(func);
    std::string ksrc   = vgre_lookup_kernel_source(func);
    if (kname.empty() || ksrc.empty())
        return cudaErrorInvalidDeviceFunction;

    vgre::dim3 grid(config->gridDim.x, config->gridDim.y, config->gridDim.z);
    vgre::dim3 block(config->blockDim.x, config->blockDim.y, config->blockDim.z);
    if (grid.x == 0)  grid.x  = 1;
    if (block.x == 0) block.x = 1;

    return CUDAInterceptor::instance().launchKernel(
        kname, ksrc, grid, block, args,
        static_cast<size_t>(config->dynamicSmemBytes),
        static_cast<vgre::StreamId>(config->stream));
}

