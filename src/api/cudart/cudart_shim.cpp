/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/compiler/kernel_parser.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// To avoid name conflicts, we define exactly the symbols frameworks need.

using namespace vgre::api;

struct uint3 {
  unsigned int x, y, z;
};

struct dim3 {
  unsigned int x, y, z;
};

// ── Global Kernel Registry ─────────────────────────────────────────────────
// Robust registry for tracking fatbinary modules and their associated kernels

using namespace vgre::common;

static std::string extractPTXFromImage(const void *image, size_t sizeHint = 0) {
  if (!image)
    return "";

  const uint32_t FATBIN_MAGIC = 0xba55ed01;
  const uint32_t ELF_MAGIC = 0x464c457f;
  const uint32_t *header = reinterpret_cast<const uint32_t *>(image);
  
  const char *data = reinterpret_cast<const char *>(image);
  size_t scanSize = sizeHint > 0 ? sizeHint : 2 * 1024 * 1024;

  if (header[0] == ELF_MAGIC) {
    VGRE_LOG_INFO("CUDART", "Parsing ELF/cubin container...");
    ELFReader reader(image, scanSize);
    size_t sSize = 0;
    // Try .nv_ptx first, then fall back to .nv_bitcode
    const char* ptx = reader.getSectionData(".nv_ptx", sSize);
    if (ptx) return std::string(ptx, sSize);
    
    const char* bc = reader.getSectionData(".nv_bitcode", sSize);
    if (bc) return std::string(bc, sSize);
  } else if (header[0] == FATBIN_MAGIC) {
    scanSize = std::min(scanSize, (size_t)1024 * 1024);
  }

  // Fallback: Scan for PTX signatures (.version, .target).
  for (size_t i = 0; i < scanSize - 64; ++i) {
    if (data[i] == '.' && std::strncmp(&data[i], ".version", 8) == 0) {
      bool foundTarget = false;
      for (size_t j = i + 8; j < i + 1024 && j < scanSize - 8; ++j) {
        if (data[j] == '.' && std::strncmp(&data[j], ".target", 7) == 0) {
          foundTarget = true;
          break;
        }
      }
      if (foundTarget) {
        return std::string(&data[i]);
      }
    }
  }
  return "";
}

class CUDAModuleRegistry {
public:
  static CUDAModuleRegistry &instance() {
    static CUDAModuleRegistry inst;
    return inst;
  }

  ~CUDAModuleRegistry() {
    // Do not acquire mutex in destructor to avoid deadlock during static destruction
    // At this point, no other threads should be accessing the registry
    // The test calls vgre_unregister_module_data explicitly to clean up resources
    // Any remaining device memory will be reclaimed by the OS
    modules_.clear();
    moduleVariables_.clear();
    moduleSources_.clear();
    hostToName_.clear();
    hostToSource_.clear();
    nameToHost_.clear();
    hostVarToDevicePtr_.clear();
    moduleNamedVariables_.clear();
    moduleTextureRefs_.clear();
    launchBoundsMap_.clear();
  }

  void **registerFatBinary(void *fatCubin) {
    if (!fatCubin) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto handle = new ModuleHandleWrapper{nextModuleId_++, fatCubin};

    // Actively scan the fatbinary for PTX source code.
    // This allows VGRE to JIT-compile kernels even when the original PTX
    // is embedded in a complex container.
    std::string extractedSource = extractPTXFromImage(fatCubin);

    modules_[handle] = {};
    moduleSources_[handle] = extractedSource; // Store extracted PTX
    return reinterpret_cast<void **>(handle);
  }

  void *registerModuleData(const void *data, size_t size) {
    if (!data || size == 0) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto handle = new ModuleHandleWrapper{nextModuleId_++, const_cast<void*>(data)};

    // Attempt to extract PTX text from raw data
    std::string extractedSource = extractPTXFromImage(data, size);

    modules_[handle] = {};
    moduleSources_[handle] = extractedSource;

    // Enhanced metadata extraction for binary images
    ELFReader reader(data, size);
    if (reader.isValid()) {
      auto globals = reader.getGlobalSymbols();
      for (const auto& sym : globals) {
        // Allocate device memory for this named global from the cubin
        void *devPtr = nullptr;
        auto err = vgre::api::CUDAInterceptor::instance().malloc(&devPtr, sym.size);
        if (err == cudaSuccess && devPtr) {
          moduleNamedVariables_[handle][sym.name] = {devPtr, sym.size};
          VGRE_LOG_INFO("CUDART", "Registered global symbol '" + sym.name + "' (" + 
                        std::to_string(sym.size) + " bytes) from cubin");
        }
      }
    } else if (!extractedSource.empty()) {
      // Fallback: Scan for global variables if it's raw PTX
      parseGlobalsFromPTX(handle, extractedSource);
    }

    return handle;
  }

  std::string lookupModuleSource(void *handlePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
    auto it = moduleSources_.find(handle);
    if (it != moduleSources_.end()) {
      return it->second;
    }
    return "";
  }

  bool unregisterModule(void *handlePtr) {
    if (!handlePtr) return false;
    std::vector<void *> varsToFree;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
      if (modules_.find(handle) == modules_.end()) {
        return false;
      }
      
      for (const auto &funcPtr : modules_[handle]) {
        auto nit = hostToName_.find(funcPtr);
        if (nit != hostToName_.end()) {
          nameToHost_.erase(nit->second);
          hostToName_.erase(nit);
        }
        hostToSource_.erase(funcPtr);
      }
      auto mvIt = moduleVariables_.find(handle);
      if (mvIt != moduleVariables_.end()) {
        for (const auto &hostVar : mvIt->second) {
          auto vit = hostVarToDevicePtr_.find(hostVar);
          if (vit != hostVarToDevicePtr_.end()) {
            varsToFree.push_back(vit->second);
            hostVarToDevicePtr_.erase(vit);
          }
        }
        moduleVariables_.erase(mvIt);
      }

      // Also free named variables
      auto nvIt = moduleNamedVariables_.find(handle);
      if (nvIt != moduleNamedVariables_.end()) {
        for (auto &pair : nvIt->second) {
          varsToFree.push_back(pair.second.ptr);
        }
        moduleNamedVariables_.erase(nvIt);
      }

      modules_.erase(handle);
      moduleSources_.erase(handle);
      delete handle;
    }
    for (void *ptr : varsToFree) {
      vgre::api::CUDAInterceptor::instance().free(ptr);
    }
    return true;
  }

  void unregisterFatBinary(void **handlePtr) {
    unregisterModule(handlePtr);
  }

  void registerFunction(void **handlePtr, const void *hostFun,
                        const char *deviceName) {
    if (!hostFun) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::string name =
        deviceName ? deviceName
                   : ("vgre_auto_kernel_" + std::to_string(nextFunctionId_++));
    hostToName_[hostFun] = name;
    nameToHost_[name] = hostFun;

    if (handlePtr) {
      auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
      if (modules_.find(handle) != modules_.end()) {
        modules_[handle].push_back(hostFun);
        // Link the host function to its module's extracted PTX source
        hostToSource_[hostFun] = moduleSources_[handle];
      }
    }
  }

  std::string lookupKernelName(const void *hostFun) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostToName_.find(hostFun);
    if (it != hostToName_.end())
      return it->second;
    return "";
  }

  std::string lookupKernelSource(const void *hostFun) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostToSource_.find(hostFun);
    if (it != hostToSource_.end() && !it->second.empty())
      return it->second;
    return "";
  }

  const void *lookupHostFunction(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nameToHost_.find(name);
    if (it != nameToHost_.end())
      return it->second;
    return nullptr;
  }

  void registerVariable(void **handlePtr, const void *hostVar, size_t size) {
    if (!hostVar || size == 0) {
      return;
    }

    ModuleHandleWrapper *owner = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (handlePtr) {
        owner = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
        if (modules_.find(owner) == modules_.end()) {
          owner = nullptr;
        }
      }
      if (owner) {
        moduleVariables_[owner].push_back(hostVar);
      }
      if (hostVarToDevicePtr_.find(hostVar) != hostVarToDevicePtr_.end()) {
        return;
      }
    }

    // Allocate outside registry lock to avoid lock inversion with runtime
    // allocation paths.
    void *devPtr = nullptr;
    auto err = vgre::api::CUDAInterceptor::instance().malloc(&devPtr, size);
    if (err != cudaSuccess || !devPtr) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = hostVarToDevicePtr_.emplace(hostVar, devPtr);
    if (!inserted) {
      vgre::api::CUDAInterceptor::instance().free(devPtr);
      (void)it;
    }
  }

  void *lookupVariable(const void *hostVar) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostVarToDevicePtr_.find(hostVar);
    if (it != hostVarToDevicePtr_.end())
      return it->second;
    return nullptr;
  }

  void *lookupVariableByName(void *handlePtr, const char *name, size_t *size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
    auto it = moduleNamedVariables_.find(handle);
    if (it != moduleNamedVariables_.end()) {
      auto vit = it->second.find(name);
      if (vit != it->second.end()) {
        if (size) *size = vit->second.size;
        return vit->second.ptr;
      }
    }
    return nullptr;
  }

  void registerTextureRef(void *handlePtr, const char *name, void *texRef) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
    moduleTextureRefs_[handle][name] = texRef;
    VGRE_LOG_INFO("CUDART", "Registered texture reference '" + std::string(name) + "' for module " + std::to_string(handle->id));
  }

  void *lookupTextureRef(void *handlePtr, const char *name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
    auto it = moduleTextureRefs_.find(handle);
    if (it != moduleTextureRefs_.end()) {
      auto tit = it->second.find(name);
      if (tit != it->second.end()) {
        return tit->second;
      }
    }
    return nullptr;
  }

private:
  struct ModuleHandleWrapper {
    uint64_t id;
    void *fatCubin;
  };

  struct VariableMetadata {
    void *ptr;
    size_t size;
  };

  void parseGlobalsFromPTX(ModuleHandleWrapper *handle, const std::string &ptx) {
    // Simple regex-based scanner for .global variables in PTX
    const std::regex global_pattern(R"(\.global\s+\.align\s+\d+\s+\.(b8|b16|b32|b64)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d*)\s*\]\s*;)");
    std::smatch match;
    std::string::const_iterator searchStart(ptx.begin());
    while (std::regex_search(searchStart, ptx.end(), match, global_pattern)) {
      std::string type = match[1];
      std::string name = match[2];
      std::string countStr = match[3];
      size_t count = countStr.empty() ? 1 : std::stoul(countStr);
      size_t elementSize = (type == "b8") ? 1 : (type == "b16") ? 2 : (type == "b32") ? 4 : 8;
      size_t totalSize = elementSize * count;

      // Allocate device memory for this named global
      void *devPtr = nullptr;
      auto err = vgre::api::CUDAInterceptor::instance().malloc(&devPtr, totalSize);
      if (err == cudaSuccess && devPtr) {
        moduleNamedVariables_[handle][name] = {devPtr, totalSize};
        VGRE_LOG_INFO("CUDART", "Registered global variable '" + name + "' (" + std::to_string(totalSize) + " bytes)");
      }
      searchStart = match.suffix().first;
    }
  }


  std::mutex mutex_;
  uint64_t nextModuleId_ = 1;
  uint64_t nextFunctionId_ = 1;

  std::unordered_map<ModuleHandleWrapper *, std::vector<const void *>> modules_;
  std::unordered_map<ModuleHandleWrapper *, std::vector<const void *>>
      moduleVariables_;
  std::unordered_map<ModuleHandleWrapper *, std::string> moduleSources_;
  std::unordered_map<const void *, std::string> hostToName_;
  std::unordered_map<const void *, std::string> hostToSource_;
  std::unordered_map<std::string, const void *> nameToHost_;
  std::unordered_map<const void *, void *> hostVarToDevicePtr_;
  std::unordered_map<ModuleHandleWrapper *, std::unordered_map<std::string, VariableMetadata>> moduleNamedVariables_;
  std::unordered_map<ModuleHandleWrapper *, std::unordered_map<std::string, void *>> moduleTextureRefs_;

  struct LaunchBoundsMeta {
    int maxThreadsPerBlock;
    vgre::dim3 blockDim;
    vgre::dim3 gridDim;
  };
  std::unordered_map<std::string, LaunchBoundsMeta> launchBoundsMap_;

public:
  void registerLaunchBounds(const char* deviceFunName, int maxThreads,
                            vgre::dim3 bDim, vgre::dim3 gDim) {
    if (!deviceFunName) return;
    std::lock_guard<std::mutex> lock(mutex_);
    launchBoundsMap_[deviceFunName] = {maxThreads, bDim, gDim};
  }
  bool getLaunchBounds(const std::string& name, int& maxThreads) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = launchBoundsMap_.find(name);
    if (it == launchBoundsMap_.end()) return false;
    maxThreads = it->second.maxThreadsPerBlock;
    return true;
  }
};

extern "C" const void *vgre_lookup_host_function_by_name(const char *name) {
  return CUDAModuleRegistry::instance().lookupHostFunction(name ? name : "");
}

extern "C" void *vgre_lookup_symbol(void *handle, const char *name, size_t *size) {
  return CUDAModuleRegistry::instance().lookupVariableByName(handle, name, size);
}

extern "C" bool vgre_unregister_module_data(void *handle) {
  if (!handle) return false;
  return CUDAModuleRegistry::instance().unregisterModule(handle);
}

extern "C" void *vgre_register_module_data(const void *data, size_t size) {
  return CUDAModuleRegistry::instance().registerModuleData(data, size);
}

extern "C" void vgre_register_texture_ref(void *handle, const char *name, void *texRef) {
  CUDAModuleRegistry::instance().registerTextureRef(handle, name, texRef);
}

extern "C" void *vgre_lookup_texture_ref(void *handle, const char *name) {
  return CUDAModuleRegistry::instance().lookupTextureRef(handle, name);
}

extern "C" const char *vgre_get_module_source(void *handle) {
  static thread_local std::string* s_source_ptr = nullptr;
  if (!s_source_ptr) s_source_ptr = new std::string();
  
  *s_source_ptr = CUDAModuleRegistry::instance().lookupModuleSource(handle);
  return s_source_ptr->empty() ? nullptr : s_source_ptr->c_str();
}


// ── Kernel Registration & Launch ───────────────────────────────────────────

extern "C" {

void **__cudaRegisterFatBinary(void *fatCubin) {
  return CUDAModuleRegistry::instance().registerFatBinary(fatCubin);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  CUDAModuleRegistry::instance().unregisterFatBinary(fatCubinHandle);
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  // Register kernel with execution configuration metadata.
  // thread_limit, bDim, gDim inform the executor's block/grid dimension caps;
  // they are passed through to CUDAModuleRegistry for optional advisory use.
  CUDAModuleRegistry::instance().registerFunction(fatCubinHandle, hostFun,
                                                  deviceName);
  // Store advisory launch bounds: if thread_limit > 0, record as the per-block
  // thread cap (mirrors __launch_bounds__).  Kernels that exceed this are still
  // executed — VGRE does not enforce a hard limit — but the profiler can use the
  // values to warn about sub-optimal occupancy.
  if (thread_limit > 0 && deviceFun && *deviceFun) {
    CUDAModuleRegistry::instance().registerLaunchBounds(
        deviceFun,
        thread_limit,
        bDim ? vgre::dim3(bDim->x, bDim->y, bDim->z) : vgre::dim3(0,0,0),
        gDim ? vgre::dim3(gDim->x, gDim->y, gDim->z) : vgre::dim3(0,0,0));
  }
  (void)tid; (void)bid; (void)wSize; // Thread/block IDs are per-launch, not per-register
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
  (void)fatCubinHandle;
  (void)deviceAddress;
  (void)deviceName;
  (void)ext;
  (void)constant;
  (void)global;

  // High-level business logic: Actively allocate backend VRAM for the detected
  // global variable so that ML frameworks can transparently copy data into/out
  // of `.cu` constant boundaries.
  if (size > 0 && hostVar) {
    CUDAModuleRegistry::instance().registerVariable(fatCubinHandle, hostVar,
                                                    size);
  }
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  if (!devPtr || !symbol) {
    return cudaErrorInvalidValue;
  }
  *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!*devPtr)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                               size_t offset, cudaMemcpyKind_t kind) {
  if (!symbol || !src || count == 0) {
    return cudaErrorInvalidValue;
  }
  void *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!devPtr)
    return cudaErrorInvalidSymbol;
  return vgre::api::CUDAInterceptor::instance().memcpy(
      static_cast<char *>(devPtr) + offset, src, count, kind);
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
                                    size_t count, size_t offset,
                                    cudaMemcpyKind_t kind,
                                    cudaStream_t stream) {
  if (!symbol || !src || count == 0) {
    return cudaErrorInvalidValue;
  }
  void *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!devPtr)
    return cudaErrorInvalidSymbol;
  return vgre::api::CUDAInterceptor::instance().memcpyAsync(
      static_cast<char *>(devPtr) + offset, src, count, kind, stream);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                  size_t offset, cudaMemcpyKind_t kind) {
  if (!dst || !symbol || count == 0) {
    return cudaErrorInvalidValue;
  }
  void *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!devPtr)
    return cudaErrorInvalidSymbol;
  return vgre::api::CUDAInterceptor::instance().memcpy(
      dst, static_cast<char *>(devPtr) + offset, count, kind);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                      size_t count, size_t offset,
                                      cudaMemcpyKind_t kind,
                                      cudaStream_t stream) {
  if (!dst || !symbol || count == 0) {
    return cudaErrorInvalidValue;
  }
  void *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!devPtr)
    return cudaErrorInvalidSymbol;
  return vgre::api::CUDAInterceptor::instance().memcpyAsync(
      dst, static_cast<char *>(devPtr) + offset, count, kind, stream);
}

cudaError_t cudaLaunchKernel(const void *hostFun, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  if (!hostFun || gridDim.x == 0 || blockDim.x == 0) {
    return cudaErrorInvalidValue;
  }
  std::string kernelName =
      CUDAModuleRegistry::instance().lookupKernelName(hostFun);
  std::string kernelSource =
      CUDAModuleRegistry::instance().lookupKernelSource(hostFun);
  if (kernelName.empty() || kernelSource.empty()) {
    return cudaErrorInvalidDeviceFunction;
  }

  vgre::dim3 vgreGrid(gridDim.x, gridDim.y, gridDim.z);
  vgre::dim3 vgreBlock(blockDim.x, blockDim.y, blockDim.z);

  // Launch via VGRE interceptor using extracted source code.
  return vgre::api::CUDAInterceptor::instance().launchKernel(
      kernelName, kernelSource, vgreGrid, vgreBlock, args, sharedMem, stream);
}

// ── Memory Pool APIs ──────────────────────────────────────────────────────

using cudaMemPool_t = uint64_t;

// Shared default pool handle for cudaMallocAsync / cudaFreeAsync
static uint64_t& getDefaultPoolHandle() {
  static uint64_t handle = 0;
  return handle;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // Pool alloc is synchronous, submitted to stream context
  if (!devPtr) return cudaErrorInvalidValue;

  uint64_t& defaultPool = getDefaultPoolHandle();
  if (defaultPool == 0) {
    auto r = vgre::core::MemoryManager::instance().createPool(defaultPool);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorMemoryAllocation;
  }

  void *handle = nullptr;
  auto r = vgre::core::MemoryManager::instance().allocateFromPool(
      defaultPool, size, handle);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorMemoryAllocation;
  *devPtr = handle;
  return cudaSuccess;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
  (void)stream;
  if (!devPtr) return cudaSuccess;

  uint64_t& defaultPool = getDefaultPoolHandle();
  if (defaultPool == 0) {
    // Pool was never created — fallback to regular free
    return vgre::api::CUDAInterceptor::instance().free(devPtr);
  }

  auto r = vgre::core::MemoryManager::instance().freeToPool(defaultPool, devPtr);
  if (r != vgre::VGREResult::SUCCESS) {
    // Fallback: regular free
    return vgre::api::CUDAInterceptor::instance().free(devPtr);
  }
  return cudaSuccess;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t *pool, unsigned int flags) {
  (void)flags;
  if (!pool) return cudaErrorInvalidValue;
  auto r = vgre::core::MemoryManager::instance().createPool(*pool);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t pool) {
  auto r = vgre::core::MemoryManager::instance().destroyPool(pool);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// ── Memory pool allocation and attribute APIs (CUDA 11.2+) ────────────────────
// cudaMallocFromPoolAsync: stream-ordered allocation from a named pool.
cudaError_t cudaMallocFromPoolAsync(void **devPtr, size_t size,
                                     cudaMemPool_t pool, cudaStream_t stream) {
  (void)stream; // stream ordering respected by MemoryManager task queue
  if (!devPtr || size == 0) return cudaErrorInvalidValue;
  auto r = vgre::core::MemoryManager::instance().allocateFromPool(pool, size, *devPtr);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
}

// Pool attribute enums (mirror CUDA headers — defined locally to avoid real CUDA headers)
enum cudaMemPoolAttr {
  cudaMemPoolAttrReleaseThreshold        = 0x1,
  cudaMemPoolAttrReservedMemCurrent      = 0x2,
  cudaMemPoolAttrReservedMemHigh         = 0x3,
  cudaMemPoolAttrUsedMemCurrent          = 0x4,
  cudaMemPoolAttrUsedMemHigh             = 0x5,
  cudaMemPoolReuseFollowEventDependencies= 0x6,
  cudaMemPoolReuseAllowOpportunistic     = 0x7,
  cudaMemPoolReuseAllowInternalDependencies = 0x8,
};

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void *value) {
  if (!value) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  switch (attr) {
    case cudaMemPoolAttrReleaseThreshold: {
      auto threshold = *static_cast<uint64_t *>(value);
      auto r = mm.setPoolReleaseThreshold(pool, threshold);
      return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
    }
    case cudaMemPoolReuseFollowEventDependencies:
    case cudaMemPoolReuseAllowOpportunistic:
    case cudaMemPoolReuseAllowInternalDependencies: {
      bool follow = true, opp = true, internal = true;
      (void)mm.getPoolReuseFlags(pool, follow, opp, internal);
      int v = *static_cast<int *>(value);
      if (attr == cudaMemPoolReuseFollowEventDependencies) follow = (v != 0);
      if (attr == cudaMemPoolReuseAllowOpportunistic) opp = (v != 0);
      if (attr == cudaMemPoolReuseAllowInternalDependencies) internal = (v != 0);
      auto r = mm.setPoolReuseFlags(pool, follow, opp, internal);
      return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
    }
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void *value) {
  if (!value) return cudaErrorInvalidValue;
  auto& mm = vgre::core::MemoryManager::instance();
  switch (attr) {
    case cudaMemPoolAttrUsedMemCurrent:
    case cudaMemPoolAttrReservedMemCurrent:
      *static_cast<uint64_t*>(value) = mm.getPoolUsedBytes(pool);
      break;
    case cudaMemPoolAttrUsedMemHigh:
    case cudaMemPoolAttrReservedMemHigh:
      *static_cast<uint64_t*>(value) = mm.getPoolPeakBytes(pool);
      break;
    case cudaMemPoolAttrReleaseThreshold: {
      uint64_t threshold = ~uint64_t{0};
      auto r = mm.getPoolReleaseThreshold(pool, threshold);
      if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
      *static_cast<uint64_t*>(value) = threshold;
      break;
    }
    case cudaMemPoolReuseFollowEventDependencies:
    case cudaMemPoolReuseAllowOpportunistic:
    case cudaMemPoolReuseAllowInternalDependencies: {
      bool follow = true, opp = true, internal = true;
      auto r = mm.getPoolReuseFlags(pool, follow, opp, internal);
      if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
      int out = 0;
      if (attr == cudaMemPoolReuseFollowEventDependencies) out = follow ? 1 : 0;
      if (attr == cudaMemPoolReuseAllowOpportunistic) out = opp ? 1 : 0;
      if (attr == cudaMemPoolReuseAllowInternalDependencies) out = internal ? 1 : 0;
      *static_cast<int*>(value) = out;
      break;
    }
    default:
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep) {
  // Release unused slab blocks back to OS while keeping at least minBytesToKeep.
  auto &mm = vgre::core::MemoryManager::instance();
  uint64_t threshold = 0;
  if (mm.getPoolReleaseThreshold(pool, threshold) == vgre::VGREResult::SUCCESS) {
    minBytesToKeep = std::max<size_t>(minBytesToKeep, static_cast<size_t>(threshold));
  }
  auto r = mm.trimPool(pool, minBytesToKeep);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

struct cudaMemLocation {
  int type;
  int id;
};

struct cudaMemAccessDesc {
  cudaMemLocation location;
  unsigned int flags;
};

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t pool,
                                  const void* descList, size_t count) {
  if (!descList && count > 0) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  auto *descs = static_cast<const cudaMemAccessDesc *>(descList);
  for (size_t i = 0; i < count; ++i) {
    auto r = mm.setPoolAccess(pool, descs[i].location.id, descs[i].flags);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemPoolGetAccess(unsigned int *flags,
                                  cudaMemPool_t pool, const void* location) {
  if (!flags || !location) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  const auto *loc = static_cast<const cudaMemLocation *>(location);
  unsigned int out = 0;
  auto r = mm.getPoolAccess(pool, loc->id, out);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
  *flags = out;
  return cudaSuccess;
}

struct vgreMemPoolShareableHandle {
  uint64_t magic;
  uint64_t pool;
};

struct vgreMemPoolPointerExportData {
  uint64_t magic;
  uint64_t pool;
  uintptr_t ptr;
};

static constexpr uint64_t kPoolShareMagic = 0x56475245504F4F4Cull; // "VGREPOOL"
static constexpr uint64_t kPoolPtrMagic   = 0x5647524550505452ull; // "VGREP PTR"

cudaError_t cudaMemPoolExportToShareableHandle(void* handle, cudaMemPool_t pool,
                                                unsigned int /*handleType*/, unsigned int /*flags*/) {
  if (!handle) return cudaErrorInvalidValue;
  // Validate pool exists.
  unsigned int tmp = 0;
  auto &mm = vgre::core::MemoryManager::instance();
  if (mm.getPoolAccess(pool, 0, tmp) != vgre::VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }
  auto *h = static_cast<vgreMemPoolShareableHandle *>(handle);
  h->magic = kPoolShareMagic;
  h->pool = pool;
  return cudaSuccess;
}
cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* pool, void* handle,
                                                  unsigned int /*handleType*/, unsigned int /*flags*/) {
  if (!pool || !handle) return cudaErrorInvalidValue;
  auto *h = static_cast<vgreMemPoolShareableHandle *>(handle);
  if (h->magic != kPoolShareMagic) return cudaErrorInvalidValue;
  unsigned int tmp = 0;
  auto &mm = vgre::core::MemoryManager::instance();
  if (mm.getPoolAccess(h->pool, 0, tmp) != vgre::VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }
  *pool = h->pool;
  return cudaSuccess;
}
cudaError_t cudaMemPoolExportPointer(void* exportData, void* ptr) {
  if (!exportData || !ptr) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  if (!mm.isValidHandle(ptr)) return cudaErrorInvalidValue;

  cudaMemPool_t ownerPool = 0;
  const auto &pools = mm.getPools();
  for (const auto &entry : pools) {
    const auto &pool = entry.second;
    if (pool.liveSlabAllocs.find(ptr) != pool.liveSlabAllocs.end() ||
        pool.oversizedAllocs.find(ptr) != pool.oversizedAllocs.end()) {
      ownerPool = entry.first;
      break;
    }
  }
  if (ownerPool == 0) return cudaErrorInvalidValue;

  auto *out = static_cast<vgreMemPoolPointerExportData *>(exportData);
  out->magic = kPoolPtrMagic;
  out->pool = ownerPool;
  out->ptr = reinterpret_cast<uintptr_t>(ptr);
  return cudaSuccess;
}
cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t pool,
                                      void* exportData) {
  if (!ptr || !exportData) return cudaErrorInvalidValue;
  auto *in = static_cast<vgreMemPoolPointerExportData *>(exportData);
  if (in->magic != kPoolPtrMagic || in->pool != pool) return cudaErrorInvalidValue;

  auto *candidate = reinterpret_cast<void *>(in->ptr);
  auto &mm = vgre::core::MemoryManager::instance();
  if (!mm.isValidHandle(candidate)) return cudaErrorInvalidValue;
  *ptr = candidate;
  return cudaSuccess;
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────

using cudaGraph_t = uint64_t;
using cudaGraphExec_t = uint64_t;
using cudaGraphNode_t = uint64_t;

using cudaGraphExecUpdateResult = vgre::api::CUDAInterceptor::cudaGraphExecUpdateResult;
using cudaMemcpy3DParms = vgre::api::cudaMemcpy3DParms;

// External semaphore APIs (implemented in cuda_external_semaphore.cpp)
using cudaExternalSemaphore_t = uint64_t;
struct cudaExternalSemaphoreSignalParams {
  struct { uint64_t value; } fence;
  unsigned int reserved[16];
  unsigned int flags;
};
struct cudaExternalSemaphoreWaitParams {
  struct { uint64_t value; } fence;
  unsigned int reserved[16];
  unsigned int flags;
};
extern "C" int cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSems,
    const cudaExternalSemaphoreSignalParams *params,
    unsigned int count, void *stream);
extern "C" int cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSems,
    const cudaExternalSemaphoreWaitParams *params,
    unsigned int count, void *stream);

struct cudaExternalSemaphoreSignalNodeParams {
  const cudaExternalSemaphore_t *extSemArray;
  const cudaExternalSemaphoreSignalParams *paramsArray;
  unsigned int numExtSems;
};
struct cudaExternalSemaphoreWaitNodeParams {
  const cudaExternalSemaphore_t *extSemArray;
  const cudaExternalSemaphoreWaitParams *paramsArray;
  unsigned int numExtSems;
};

namespace {
struct GraphExtSemNodeState {
  bool isWait = false;
  std::vector<cudaExternalSemaphore_t> sems;
  std::vector<cudaExternalSemaphoreSignalParams> signalParams;
  std::vector<cudaExternalSemaphoreWaitParams> waitParams;
};

// Use function-local static initialization to prevent blocking during library load
// This ensures the mutex and map are only initialized when first used
std::mutex& getGraphExtSemMutex() {
  static std::mutex mu;
  return mu;
}

std::unordered_map<cudaGraphNode_t, std::shared_ptr<GraphExtSemNodeState>>&
getGraphExtSemNodes() {
  static std::unordered_map<cudaGraphNode_t, std::shared_ptr<GraphExtSemNodeState>> nodes;
  return nodes;
}

static int vgre_ext_sem_graph_callback(void *ctx) {
  auto *state = static_cast<GraphExtSemNodeState *>(ctx);
  if (!state || state->sems.empty()) return 0;
  if (state->isWait) {
    (void)cudaWaitExternalSemaphoresAsync(state->sems.data(),
                                          state->waitParams.data(),
                                          static_cast<unsigned int>(state->sems.size()),
                                          nullptr);
  } else {
    (void)cudaSignalExternalSemaphoresAsync(state->sems.data(),
                                            state->signalParams.data(),
                                            static_cast<unsigned int>(state->sems.size()),
                                            nullptr);
  }
  return 0;
}

} // namespace

// ── Exposed helpers for cudart_shim_external_memory.cpp ─────────────────────
extern "C" cudaError_t vgreGraphExtSemGetSignalNodeParams(
        cudaGraphNode_t node,
        cudaExternalSemaphoreSignalNodeParams *params_out) {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto *state = it->second.get();
    if (state->isWait) return cudaErrorInvalidValue;
    if (!params_out) return cudaErrorInvalidValue;
    params_out->extSemArray = state->sems.data();
    params_out->paramsArray = state->signalParams.data();
    params_out->numExtSems = static_cast<unsigned int>(state->sems.size());
    return cudaSuccess;
}

extern "C" cudaError_t vgreGraphExtSemGetWaitNodeParams(
        cudaGraphNode_t node,
        cudaExternalSemaphoreWaitNodeParams *params_out) {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto *state = it->second.get();
    if (!state->isWait) return cudaErrorInvalidValue;
    if (!params_out) return cudaErrorInvalidValue;
    params_out->extSemArray = state->sems.data();
    params_out->paramsArray = state->waitParams.data();
    params_out->numExtSems = static_cast<unsigned int>(state->sems.size());
    return cudaSuccess;
}

cudaError_t cudaGraphCreate(cudaGraph_t *graph, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().graphCreate(graph, flags);
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, unsigned int mode) {
  (void)mode;
  return vgre::api::CUDAInterceptor::instance().streamBeginCapture(stream);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
  return vgre::api::CUDAInterceptor::instance().streamEndCapture(stream, pGraph);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                 cudaGraphNode_t *pErrorNode, char *pLogBuffer,
                                 size_t bufferSize) {
  cudaError_t r = vgre::api::CUDAInterceptor::instance().graphInstantiate(pGraphExec, graph);
  if (r != cudaSuccess) {
    // Report the failed node back to the caller and populate the log buffer.
    if (pErrorNode) *pErrorNode = 0; // VGRE DAG validates all nodes; no specific failed node
    if (pLogBuffer && bufferSize > 0) {
      const char* msg = (r == cudaErrorInvalidValue)
          ? "Graph instantiation failed: invalid node or dependency cycle"
          : "Graph instantiation failed: internal error";
      std::strncpy(pLogBuffer, msg, bufferSize - 1);
      pLogBuffer[bufferSize - 1] = '\0';
    }
  }
  return r;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().graphLaunch(graphExec, stream);
}

cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
  return vgre::api::CUDAInterceptor::instance().graphClone(pGraphClone, originalGraph);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  return vgre::api::CUDAInterceptor::instance().graphDestroy(graph);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  return vgre::api::CUDAInterceptor::instance().graphExecDestroy(graphExec);
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                   const cudaMemcpy3DParms *pCopyParams) {
  return vgre::api::CUDAInterceptor::instance().graphAddMemcpyNode(
      pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                     const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                     void *dst, const void *src, size_t count, cudaMemcpyKind_t kind) {
  return vgre::api::CUDAInterceptor::instance().graphAddMemcpyNode1D(
      pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) {
  return vgre::api::CUDAInterceptor::instance().graphExecUpdate(
      hGraphExec, hGraph, hErrorNode_out, reinterpret_cast<vgre::api::CUDAInterceptor::cudaGraphExecUpdateResult*>(updateResult_out));
}

// Phase 10: cudaGraphExecUpdate_v2 — targeted update of specific nodes only.
cudaError_t cudaGraphExecUpdate_v2(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                   const cudaGraphNode_t *updateNodeList, size_t updateNodeListSize,
                                   cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) {
  return vgre::api::CUDAInterceptor::instance().graphExecUpdateV2(
      hGraphExec, hGraph, updateNodeList, updateNodeListSize, hErrorNode_out,
      reinterpret_cast<vgre::api::CUDAInterceptor::cudaGraphExecUpdateResult*>(updateResult_out));
}

// VGRE extension: conditional graph node (IF/WHILE semantics).
// condFn(condCtx) returns non-zero to execute the body.
// flags: 0 = IF (body executes once), 1 = WHILE (body loops while condFn != 0).
// maxIterations: safety cap for WHILE loops (default 65536 when 0 is passed).
cudaError_t cudaGraphAddConditionalNode(cudaGraphNode_t *pGraphNode,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t *pDependencies,
                                        size_t numDependencies,
                                        int (*condFn)(void *),
                                        void *condCtx,
                                        cudaGraph_t bodyGraph,
                                        unsigned int flags,
                                        unsigned int maxIterations) {
  if (!pGraphNode) return cudaError_t(1); // cudaErrorInvalidValue
  auto &engine = vgre::core::RuntimeEngine::instance();
  if (!engine.isInitialized()) return cudaError_t(3); // cudaErrorNotInitialized

  std::vector<uint64_t> deps(pDependencies, pDependencies + numDependencies);
  vgre::core::GraphCondType condType = vgre::core::GraphCondType::IF;
  if (flags == 1) condType = vgre::core::GraphCondType::WHILE;
  if (flags == 2) condType = vgre::core::GraphCondType::SWITCH;
  unsigned int iters = (maxIterations == 0) ? 65536 : maxIterations;
  uint64_t outNodeId = 0;
  auto r = engine.graphAddConditionalNode(
      static_cast<vgre::GraphId>(graph), condFn, condCtx,
      static_cast<vgre::GraphId>(bodyGraph), condType, iters, deps, outNodeId);
  if (r != vgre::VGREResult::SUCCESS) return cudaError_t(1);
  *pGraphNode = outNodeId;
  return cudaError_t(0); // cudaSuccess
}

cudaError_t cudaGraphAddExternalSemaphoreSignalNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const cudaExternalSemaphoreSignalNodeParams *nodeParams) {
  if (!pGraphNode || !nodeParams || nodeParams->numExtSems == 0 ||
      !nodeParams->extSemArray || !nodeParams->paramsArray) {
    return cudaErrorInvalidValue;
  }
  auto state = std::make_shared<GraphExtSemNodeState>();
  state->isWait = false;
  state->sems.assign(nodeParams->extSemArray,
                     nodeParams->extSemArray + nodeParams->numExtSems);
  state->signalParams.assign(nodeParams->paramsArray,
                             nodeParams->paramsArray + nodeParams->numExtSems);

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddConditionalNode(
      static_cast<vgre::GraphId>(graph), vgre_ext_sem_graph_callback,
      static_cast<void *>(state.get()), 0,
      vgre::core::GraphCondType::IF, 1, deps, nodeId);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

  {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    getGraphExtSemNodes()[nodeId] = std::move(state);
  }
  *pGraphNode = nodeId;
  return cudaSuccess;
}

cudaError_t cudaGraphAddExternalSemaphoreWaitNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const cudaExternalSemaphoreWaitNodeParams *nodeParams) {
  if (!pGraphNode || !nodeParams || nodeParams->numExtSems == 0 ||
      !nodeParams->extSemArray || !nodeParams->paramsArray) {
    return cudaErrorInvalidValue;
  }
  auto state = std::make_shared<GraphExtSemNodeState>();
  state->isWait = true;
  state->sems.assign(nodeParams->extSemArray,
                     nodeParams->extSemArray + nodeParams->numExtSems);
  state->waitParams.assign(nodeParams->paramsArray,
                           nodeParams->paramsArray + nodeParams->numExtSems);

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }
  uint64_t nodeId = 0;
  auto r = vgre::core::RuntimeEngine::instance().graphAddConditionalNode(
      static_cast<vgre::GraphId>(graph), vgre_ext_sem_graph_callback,
      static_cast<void *>(state.get()), 0,
      vgre::core::GraphCondType::IF, 1, deps, nodeId);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

  {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    getGraphExtSemNodes()[nodeId] = std::move(state);
  }
  *pGraphNode = nodeId;
  return cudaSuccess;
}

cudaError_t cudaGraphExecExternalSemaphoreSignalNodeSetParams(
    cudaGraphExec_t /*graphExec*/, cudaGraphNode_t node,
    const cudaExternalSemaphoreSignalNodeParams *nodeParams) {
  if (!nodeParams || nodeParams->numExtSems == 0 || !nodeParams->extSemArray ||
      !nodeParams->paramsArray) {
    return cudaErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
  auto it = getGraphExtSemNodes().find(node);
  if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
  auto &state = *it->second;
  state.isWait = false;
  state.sems.assign(nodeParams->extSemArray,
                    nodeParams->extSemArray + nodeParams->numExtSems);
  state.signalParams.assign(nodeParams->paramsArray,
                            nodeParams->paramsArray + nodeParams->numExtSems);
  state.waitParams.clear();
  return cudaSuccess;
}

cudaError_t cudaGraphExecExternalSemaphoreWaitNodeSetParams(
    cudaGraphExec_t /*graphExec*/, cudaGraphNode_t node,
    const cudaExternalSemaphoreWaitNodeParams *nodeParams) {
  if (!nodeParams || nodeParams->numExtSems == 0 || !nodeParams->extSemArray ||
      !nodeParams->paramsArray) {
    return cudaErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
  auto it = getGraphExtSemNodes().find(node);
  if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
  auto &state = *it->second;
  state.isWait = true;
  state.sems.assign(nodeParams->extSemArray,
                    nodeParams->extSemArray + nodeParams->numExtSems);
  state.waitParams.assign(nodeParams->paramsArray,
                          nodeParams->paramsArray + nodeParams->numExtSems);
  state.signalParams.clear();
  return cudaSuccess;
}

// ── Cooperative Launch ────────────────────────────────────────────────────

cudaError_t cudaLaunchCooperativeKernel(const void *hostFun, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
  if (!hostFun || gridDim.x == 0 || blockDim.x == 0)
    return cudaErrorInvalidValue;

  std::string kernelName =
      CUDAModuleRegistry::instance().lookupKernelName(hostFun);
  std::string kernelSource =
      CUDAModuleRegistry::instance().lookupKernelSource(hostFun);
  if (kernelName.empty() || kernelSource.empty())
    return cudaErrorInvalidDeviceFunction;

  vgre::dim3 vgreGrid(gridDim.x, gridDim.y, gridDim.z);
  vgre::dim3 vgreBlock(blockDim.x, blockDim.y, blockDim.z);

  // Route through the cooperative execution path so that
  // this_grid().sync() / vgre_jit_syncgrid() works correctly.
  return vgre::api::CUDAInterceptor::instance().launchCooperativeKernel(
      kernelName, kernelSource, vgreGrid, vgreBlock, args, sharedMem, stream);
}

struct cudaLaunchParams {
  const void *func;
  dim3 gridDim;
  dim3 blockDim;
  void **args;
  size_t sharedMem;
  cudaStream_t stream;
};

cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    cudaLaunchParams *launchParamsList, unsigned int numDevices,
    unsigned int flags) {
  // flags interpretation (matches CUDA docs):
  //   cudaCooperativeLaunchMultiDeviceNoPreSync  (0x01) — we always start fresh,
  //   cudaCooperativeLaunchMultiDeviceNoPostSync (0x02) — caller syncs manually.
  // Both are no-ops in VGRE's CPU model; the ready-gate in Phase 2 of
  // launchCooperativeKernelMultiDevice already ensures simultaneous start.
  (void)flags;

  if (!launchParamsList || numDevices == 0) return cudaErrorInvalidValue;

  // Phase 1 — resolve kernel names from the module registry before handing
  // off to RuntimeEngine (registry lookup is not thread-safe, so done here
  // on the calling thread before device threads are spawned).
  std::vector<vgre::core::RuntimeEngine::CoopMultiLaunchParams> params;
  params.reserve(numDevices);

  for (unsigned i = 0; i < numDevices; ++i) {
    const auto &p = launchParamsList[i];
    if (!p.func || p.gridDim.x == 0 || p.blockDim.x == 0)
      return cudaErrorInvalidValue;

    std::string name = CUDAModuleRegistry::instance().lookupKernelName(p.func);
    std::string src  = CUDAModuleRegistry::instance().lookupKernelSource(p.func);
    if (name.empty() || src.empty()) return cudaErrorInvalidDeviceFunction;

    params.push_back({
        name, src,
        vgre::dim3(p.gridDim.x, p.gridDim.y, p.gridDim.z),
        vgre::dim3(p.blockDim.x, p.blockDim.y, p.blockDim.z),
        p.args,
        p.sharedMem,
        // cudaStream_t is opaque; cast to uint64_t for VGRE's StreamId.
        static_cast<vgre::StreamId>(static_cast<uintptr_t>(p.stream))
    });
  }

  auto r = vgre::core::RuntimeEngine::instance()
               .launchCooperativeKernelMultiDevice(params);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize) {
  if (!numBlocks || blockSize <= 0) return cudaErrorInvalidValue;

  // ── Query current device SM limits (architecture-aware) ─────────────────────
  vgre::DeviceProperties dp{};
  vgre::core::RuntimeEngine::instance().getDeviceProperties(
      vgre::core::RuntimeEngine::instance().getDeviceId(), dp);

  const int kMaxWarpsPerSM     = dp.maxWarpsPerSM;
  const int kMaxBlocksPerSM    = dp.maxBlocksPerSM;
  const int kMaxThreadsPerSM   = dp.maxThreadsPerSM;
  const int kMaxRegsPerSM      = dp.maxRegsPerSM;
  const int kMaxSharedMemPerSM = dp.maxSharedMemPerSM;

  int warpsPerBlock = (blockSize + dp.warpSize - 1) / dp.warpSize;

  // Limit 1: warp capacity
  int limitWarps = kMaxWarpsPerSM / std::max(1, warpsPerBlock);

  // Limit 2: thread capacity
  int limitThreads = kMaxThreadsPerSM / blockSize;

  // ── Look up kernel metadata via RuntimeEngine ──────────────────────────────
  int registersPerThread = 32; // conservative default
  size_t staticSMem      = 0;

  if (func) {
    // Path: host stub pointer → device name → KernelId → KernelIR
    std::string kName = CUDAModuleRegistry::instance().lookupKernelName(func);
    if (!kName.empty()) {
      vgre::KernelId kid =
          vgre::core::RuntimeEngine::instance().lookupKernelIdByName(kName.c_str());
      if (kid != 0) {
        const vgre::KernelIR* ir =
            vgre::core::RuntimeEngine::instance().getKernelIR(kid);
        if (ir) {
          staticSMem = ir->sharedMemSize;
          if (ir->registersPerThread > 0 && ir->registersPerThread != 32) {
            registersPerThread = ir->registersPerThread;
          } else {
            // Try to parse registers from PTX on demand
            std::string ptx = CUDAModuleRegistry::instance().lookupKernelSource(func);
            int parsedRegs = vgre::compiler::parsePTXRegisterCount(ptx, kName);
            if (parsedRegs > 0) {
              registersPerThread = parsedRegs;
              // Cache the parsed value back into KernelIR for future calls
              const_cast<vgre::KernelIR*>(ir)->registersPerThread = parsedRegs;
            }
          }
        }
      }
    }
  }

  // Limit 3: register pressure
  int regsPerBlock = warpsPerBlock * dp.warpSize * registersPerThread;
  int limitRegs = (regsPerBlock > 0) ? (kMaxRegsPerSM / regsPerBlock) : kMaxBlocksPerSM;

  // Limit 4: shared memory
  size_t totalSMem = staticSMem + dynamicSMemSize;
  int limitSMem = (totalSMem > 0)
      ? static_cast<int>(kMaxSharedMemPerSM / totalSMem)
      : kMaxBlocksPerSM;

  // Limit 5: hard SM block cap
  int active = std::min({limitWarps, limitThreads, limitRegs, limitSMem, kMaxBlocksPerSM});
  *numBlocks = std::max(1, active);

  VGRE_LOG_DEBUG("Occupancy",
      "blockSize=" + std::to_string(blockSize) +
      " warps=" + std::to_string(warpsPerBlock) +
      " regs/thread=" + std::to_string(registersPerThread) +
      " smem=" + std::to_string(totalSMem) +
      " → activeBlocks=" + std::to_string(*numBlocks) +
      " (limitW=" + std::to_string(limitWarps) +
      " limitT=" + std::to_string(limitThreads) +
      " limitR=" + std::to_string(limitRegs) +
      " limitS=" + std::to_string(limitSMem) +
      " arch=" + std::to_string(dp.major) + "." + std::to_string(dp.minor) + ")");
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize,
    size_t dynamicSMemSize, unsigned int /*flags*/) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, func, blockSize, dynamicSMemSize);
}

// ── Mipmapped Array API ────────────────────────────────────────────────────
// cudaMipmappedArray_t is an opaque handle; we use TextureId cast to void*.

#include "vgre/core/texture_manager.h"

cudaError_t cudaMallocMipmappedArray(void **mipmappedArrayPtr,
                                     const cudaChannelFormatDesc *desc,
                                     size_t width, size_t height,
                                     unsigned int numLevels,
                                     unsigned int flags) {
  (void)flags;
  if (!mipmappedArrayPtr || !desc || width == 0)
    return cudaErrorInvalidValue;

  // Determine element size from channel descriptor (x+y+z+w bits → bytes)
  size_t bits = static_cast<size_t>(desc->x + desc->y + desc->z + desc->w);
  size_t elementSize = (bits == 0) ? 4 : (bits + 7) / 8;

  vgre::core::TextureDescriptor td;
  td.filterMode = vgre::core::TextureFilterMode::LINEAR;
  td.addressMode = vgre::core::TextureAddressMode::CLAMP;

  vgre::core::TextureId id = 0;
  auto r = vgre::core::TextureManager::instance().createMipmappedArray(
      id, width, height, elementSize, numLevels, td);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorMemoryAllocation;

  *mipmappedArrayPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(id));
  return cudaSuccess;
}

cudaError_t cudaFreeMipmappedArray(void *mipmappedArray) {
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  vgre::core::TextureManager::instance().destroyCudaArray(id);
  return cudaSuccess;
}

// Returns a pointer to a specific mip level (as a cudaArray_t = void*).
cudaError_t cudaGetMipmappedArrayLevel(void **levelArrayPtr,
                                       void *mipmappedArray,
                                       unsigned int level) {
  if (!levelArrayPtr || !mipmappedArray) return cudaErrorInvalidValue;
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  void *ptr = vgre::core::TextureManager::instance().getMipmapLevelData(id, level);
  if (!ptr) return cudaErrorInvalidValue;
  *levelArrayPtr = ptr;
  return cudaSuccess;
}

// Generate mip levels from the base (level 0) data using box filtering.
cudaError_t cudaGenerateMipmaps(void *mipmappedArray) {
  auto id = static_cast<vgre::core::TextureId>(
      reinterpret_cast<uintptr_t>(mipmappedArray));
  auto r = vgre::core::TextureManager::instance().generateMipmaps(id);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

} // extern "C"

// ── Cross-file kernel registry helpers ────────────────────────────────────────
// Exported as plain C++ symbols (not extern "C") so other .cpp files in the
// same shared library can call them without name-mangling issues.

std::string vgre_lookup_kernel_name(const void *hostFn) {
  return CUDAModuleRegistry::instance().lookupKernelName(hostFn);
}

std::string vgre_lookup_kernel_source(const void *hostFn) {
  return CUDAModuleRegistry::instance().lookupKernelSource(hostFn);
}
