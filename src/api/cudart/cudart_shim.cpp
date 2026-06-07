/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/api/fatbinary_utils.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/compiler/kernel_parser.h"
#include "../../compiler/sass/sass_decoder.h"
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

using namespace vgre::fatbin;

static std::string extractPTXFromImage(const void *image, size_t sizeHint = 0) {
  if (!image)
    return "";

  const uint32_t ELF_MAGIC = 0x464c457f;
  const uint32_t *header = reinterpret_cast<const uint32_t *>(image);
  const uint8_t  *bytes  = reinterpret_cast<const uint8_t *>(image);
  const char     *data   = reinterpret_cast<const char *>(image);
  size_t scanSize = sizeHint > 0 ? sizeHint : 2 * 1024 * 1024;

  // ── 1. ELF container (.cubin or thin ELF with PTX sections) ─────────────
  if (header[0] == ELF_MAGIC) {
    VGRE_LOG_INFO("CUDART", "Parsing ELF/cubin container...");
    ELFReader reader(image, scanSize);
    size_t sSize = 0;
    const char* ptx = reader.getSectionData(".nv_ptx", sSize);
    if (ptx) return std::string(ptx, sSize);
    const char* bc = reader.getSectionData(".nv_bitcode", sSize);
    if (bc) return std::string(bc, sSize);
    // ELF is SASS-only if no PTX section — attempt SASS decode before giving up
    VGRE_LOG_WARN("CUDART", "ELF/cubin has no .nv_ptx section — attempting SASS decode");
    {
        std::string decoded = vgre::sass::decodeSassToPtx(
            reinterpret_cast<const uint8_t*>(image), scanSize);
        if (!decoded.empty()) {
            VGRE_LOG_INFO("CUDART", "SASS decode produced " +
                          std::to_string(decoded.size()) + " bytes of PTX");
            return decoded;
        }
    }
    VGRE_LOG_WARN("CUDART", "SASS decode failed — SASS-only binary unsupported");
    return kSassOnlyMarker;
  }

  // ── 2. __fatBinC_Wrapper_t: wrapper struct whose [2] field is a data ptr ─
  if (header[0] == kWrapper) {
    // Layout: magic(4), version(4), data*(8), filename*(8)
    const uint64_t* wrapperFields = reinterpret_cast<const uint64_t*>(image);
    // data pointer is in the second 8-byte slot (bytes 8-15)
    uint64_t dataPtr = wrapperFields[1];
    if (dataPtr != 0) {
      const uint8_t* fatbinData = reinterpret_cast<const uint8_t*>(
          static_cast<uintptr_t>(dataPtr));
      std::string parsed = parseSections(fatbinData, 64 * 1024 * 1024u);
      if (!parsed.empty()) return parsed;
    }
  }

  // ── 3. Raw fatbin container ──────────────────────────────────────────────
  if (header[0] == kMagic || header[0] == kMagicOld) {
    std::string parsed = parseSections(bytes, scanSize);
    if (!parsed.empty()) return parsed;
    // parseSections returned "" → compressed PTX; fall through to linear scan
  }

  // ── 4. Fallback: linear scan for PTX signature (.version + .target) ──────
  for (size_t i = 0; i + 64 < scanSize; ++i) {
    if (data[i] == '.' && std::strncmp(&data[i], ".version", 8) == 0) {
      bool foundTarget = false;
      for (size_t j = i + 8; j < i + 1024 && j + 8 < scanSize; ++j) {
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

VGRE_PUBLIC_API
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

VGRE_PUBLIC_API
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

// ── CUDA Graphs API moved to cudart_shim_graph_basic.cpp ─────────────────────
// (graph create/destroy/capture/launch/clone/update + external-semaphore nodes)

using cudaExternalSemaphore_t = uint64_t;

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
