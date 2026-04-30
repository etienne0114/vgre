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
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
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
    static CUDAModuleRegistry* inst = new CUDAModuleRegistry();
    return *inst;
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
        hostToName_.erase(funcPtr);
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
  std::unordered_map<const void *, void *> hostVarToDevicePtr_;
  std::unordered_map<ModuleHandleWrapper *, std::unordered_map<std::string, VariableMetadata>> moduleNamedVariables_;
  std::unordered_map<ModuleHandleWrapper *, std::unordered_map<std::string, void *>> moduleTextureRefs_;
};

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

  (void)deviceFun;
  (void)thread_limit;
  (void)tid;
  (void)bid;
  (void)bDim;
  (void)gDim;
  (void)wSize;
  CUDAModuleRegistry::instance().registerFunction(fatCubinHandle, hostFun,
                                                  deviceName);
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

// ── CUDA Graphs API ────────────────────────────────────────────────────────

using cudaGraph_t = uint64_t;
using cudaGraphExec_t = uint64_t;
using cudaGraphNode_t = uint64_t;

using cudaGraphExecUpdateResult = vgre::api::CUDAInterceptor::cudaGraphExecUpdateResult;
using cudaMemcpy3DParms = vgre::api::CUDAInterceptor::cudaMemcpy3DParms;

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
  (void)pErrorNode;
  (void)pLogBuffer;
  (void)bufferSize;
  return vgre::api::CUDAInterceptor::instance().graphInstantiate(pGraphExec, graph);
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
  vgre::core::GraphCondType condType =
      (flags == 1) ? vgre::core::GraphCondType::WHILE
                   : vgre::core::GraphCondType::IF;
  unsigned int iters = (maxIterations == 0) ? 65536 : maxIterations;
  uint64_t outNodeId = 0;
  auto r = engine.graphAddConditionalNode(
      static_cast<vgre::GraphId>(graph), condFn, condCtx,
      static_cast<vgre::GraphId>(bodyGraph), condType, iters, deps, outNodeId);
  if (r != vgre::VGREResult::SUCCESS) return cudaError_t(1);
  *pGraphNode = outNodeId;
  return cudaError_t(0); // cudaSuccess
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
  (void)func;
  (void)dynamicSMemSize;
  if (!numBlocks || blockSize <= 0) return cudaErrorInvalidValue;

  // For CPU execution: max concurrent blocks = hardware threads / blockSize
  int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
  if (hwThreads == 0) hwThreads = 4;
  *numBlocks = std::max(1, hwThreads / blockSize);
  return cudaSuccess;
}

// ── Mipmapped Array API ────────────────────────────────────────────────────
// cudaMipmappedArray_t is an opaque handle; we use TextureId cast to void*.

#include "vgre/core/texture_manager.h"

struct cudaChannelFormatDesc {
  int x, y, z, w;
  int f; // kind: 0=signed int, 1=unsigned int, 2=float
};

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
