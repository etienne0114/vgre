/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// To avoid name conflicts, we define exactly the symbols frameworks need.
extern "C" {

using namespace vgre::api;

struct uint3 {
  unsigned int x, y, z;
};

struct dim3 {
  unsigned int x, y, z;
};

// ── Global Kernel Registry ─────────────────────────────────────────────────
// Robust registry for tracking fatbinary modules and their associated kernels

class ELFReader {
public:
  struct Header {
    uint8_t  magic[4]; // 0x7f, 'E', 'L', 'F'
    uint8_t  cls;
    uint8_t  data_encoding;
    uint8_t  version;
    uint8_t  osabi;
    uint8_t  abiversion;
    uint8_t  pad[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint64_t entry;
    uint64_t phoff;
    uint64_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
  };

  struct SectionHeader {
    uint32_t name;
    uint32_t type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t addralign;
    uint64_t entsize;
  };

  struct Symbol {
    uint32_t name;
    uint8_t  info;
    uint8_t  other;
    uint16_t shndx;
    uint64_t value;
    uint64_t size;
  };

  ELFReader(const void* data, size_t size) : data_(reinterpret_cast<const uint8_t*>(data)), size_(size) {}

  bool isValid() const {
    if (size_ < sizeof(Header)) return false;
    const Header* h = reinterpret_cast<const Header*>(data_);
    return h->magic[0] == 0x7f && h->magic[1] == 'E' && h->magic[2] == 'L' && h->magic[3] == 'F';
  }

  const char* getSectionData(const char* targetName, size_t& outSize) {
    if (!isValid()) return nullptr;
    const Header* h = reinterpret_cast<const Header*>(data_);
    if (h->shoff == 0 || h->shnum == 0) return nullptr;

    const SectionHeader* shstr = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + h->shstrndx * h->shentsize);
    const char* strtab = reinterpret_cast<const char*>(data_ + shstr->offset);

    for (int i = 0; i < h->shnum; ++i) {
      const SectionHeader* sh = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + i * h->shentsize);
      const char* name = strtab + sh->name;
      if (std::strcmp(name, targetName) == 0) {
        outSize = sh->size;
        return reinterpret_cast<const char*>(data_ + sh->offset);
      }
    }
    return nullptr;
  }

  struct SymbolInfo {
    std::string name;
    uint64_t value;
    uint64_t size;
  };

  std::vector<SymbolInfo> getGlobalSymbols() {
    std::vector<SymbolInfo> results;
    if (!isValid()) return results;
    const Header* h = reinterpret_cast<const Header*>(data_);
    
    const SectionHeader* symtab_sh = nullptr;
    const SectionHeader* strtab_sh = nullptr;

    const SectionHeader* shstr = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + h->shstrndx * h->shentsize);
    const char* shstrtab = reinterpret_cast<const char*>(data_ + shstr->offset);

    for (int i = 0; i < h->shnum; ++i) {
      const SectionHeader* sh = reinterpret_cast<const SectionHeader*>(data_ + h->shoff + i * h->shentsize);
      const char* name = shstrtab + sh->name;
      if (sh->type == 2) { // SHT_SYMTAB
        symtab_sh = sh;
      } else if (std::strcmp(name, ".strtab") == 0) {
        strtab_sh = sh;
      }
    }

    if (symtab_sh && strtab_sh) {
      const Symbol* syms = reinterpret_cast<const Symbol*>(data_ + symtab_sh->offset);
      const char* strings = reinterpret_cast<const char*>(data_ + strtab_sh->offset);
      size_t numSyms = symtab_sh->size / sizeof(Symbol);

      for (size_t i = 0; i < numSyms; ++i) {
        const Symbol& s = syms[i];
        uint8_t bind = s.info >> 4;
        uint8_t type = s.info & 0xf;
        // STB_GLOBAL and (STT_OBJECT or STT_FUNC)
        if (bind == 1 && (type == 1 || type == 2) && s.name != 0) {
          results.push_back({strings + s.name, s.value, s.size});
        }
      }
    }
    return results;
  }

private:
  const uint8_t* data_;
  size_t size_;
};

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
    static CUDAModuleRegistry registry;
    return registry;
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

extern "C" const char *vgre_get_module_source(void *handle) {
  static thread_local std::string s_source;
  s_source = CUDAModuleRegistry::instance().lookupModuleSource(handle);
  return s_source.empty() ? nullptr : s_source.c_str();
}

// ── Initialization & Error Handling ────────────────────────────────────────

cudaError_t cudaGetLastError(void) {
  return vgre::api::CUDAInterceptor::instance().getLastError();
}

cudaError_t cudaPeekAtLastError(void) {
  // Peek returns the last error WITHOUT clearing it (unlike cudaGetLastError)
  return vgre::api::CUDAInterceptor::instance().peekLastError();
}

const char *cudaGetErrorString(cudaError_t error) {
  return vgre::api::CUDAInterceptor::instance().getErrorString(error);
}

const char *cudaGetErrorName(cudaError_t error) {
  return vgre::api::CUDAInterceptor::instance().getErrorString(error);
}

// ── Device Management ──────────────────────────────────────────────────────

cudaError_t cudaGetDeviceCount(int *count) {
  return vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
}

cudaError_t cudaSetDevice(int device) {
  return vgre::api::CUDAInterceptor::instance().setDevice(device);
}

cudaError_t cudaGetDevice(int *device) {
  return vgre::api::CUDAInterceptor::instance().getDevice(device);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t *prop, int device) {
  return vgre::api::CUDAInterceptor::instance().getDeviceProperties(prop,
                                                                    device);
}

cudaError_t cudaDeviceSynchronize(void) {
  return vgre::api::CUDAInterceptor::instance().deviceSynchronize();
}

cudaError_t cudaDeviceReset(void) {
  return vgre::api::CUDAInterceptor::instance().deviceReset();
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().setDeviceFlags(flags);
}

cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
  return vgre::api::CUDAInterceptor::instance().getDeviceFlags(flags);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  return vgre::api::CUDAInterceptor::instance().deviceGetPCIBusId(pciBusId, len,
                                                                 device);
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
  return vgre::api::CUDAInterceptor::instance().deviceGetByPCIBusId(device,
                                                                    pciBusId);
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice) {
  return vgre::api::CUDAInterceptor::instance().deviceCanAccessPeer(
      canAccessPeer, device, peerDevice);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().deviceEnablePeerAccess(
      peerDevice, flags);
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
  return vgre::api::CUDAInterceptor::instance().deviceDisablePeerAccess(
      peerDevice);
}

// ── Memory Management ──────────────────────────────────────────────────────

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  return vgre::api::CUDAInterceptor::instance().malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  return vgre::api::CUDAInterceptor::instance().free(devPtr);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind_t kind) {
  return vgre::api::CUDAInterceptor::instance().memcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind_t kind, cudaStream_t stream) {
  // Escalate to true async processing pool via Interceptor
  return vgre::api::CUDAInterceptor::instance().memcpyAsync(dst, src, count,
                                                            kind, stream);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind_t kind) {
  return vgre::api::CUDAInterceptor::instance().memcpy2D(
      dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind_t kind, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memcpy2DAsync(
      dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count, unsigned int advice, int device) {
  return vgre::api::CUDAInterceptor::instance().memAdvise(devPtr, count, static_cast<int>(advice), device);
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memPrefetchAsync(devPtr, count, dstDevice, stream);
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                           int srcDevice, size_t count) {
  return vgre::api::CUDAInterceptor::instance().memcpyPeer(
      dst, dstDevice, src, srcDevice, count);
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memcpyPeerAsync(
      dst, dstDevice, src, srcDevice, count, stream);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  return vgre::api::CUDAInterceptor::instance().memset(devPtr, value, count);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memsetAsync(devPtr, value,
                                                            count, stream);
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                            size_t height) {
  return vgre::api::CUDAInterceptor::instance().mallocPitch(devPtr, pitch,
                                                            width, height);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().hostAlloc(pHost, size, flags);
}

cudaError_t cudaFreeHost(void *pHost) {
  return vgre::api::CUDAInterceptor::instance().freeHost(pHost);
}

cudaError_t cudaHostRegister(void *pHost, size_t size, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().hostRegister(pHost, size,
                                                             flags);
}

cudaError_t cudaHostUnregister(void *pHost) {
  return vgre::api::CUDAInterceptor::instance().hostUnregister(pHost);
}

// ── Stream Management ──────────────────────────────────────────────────────

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  return vgre::api::CUDAInterceptor::instance().streamCreate(pStream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().streamCreateWithFlags(pStream,
                                                                      flags);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
  return vgre::api::CUDAInterceptor::instance().streamCreateWithPriority(
      pStream, flags, priority);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamSynchronize(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamQuery(stream);
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  return vgre::api::CUDAInterceptor::instance().deviceGetStreamPriorityRange(
      leastPriority, greatestPriority);
}

// ── Event Management ───────────────────────────────────────────────────────

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return vgre::api::CUDAInterceptor::instance().eventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().eventCreateWithFlags(event,
                                                                     flags);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().eventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return vgre::api::CUDAInterceptor::instance().eventSynchronize(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end) {
  return vgre::api::CUDAInterceptor::instance().eventElapsedTime(ms, start,
                                                                 end);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return vgre::api::CUDAInterceptor::instance().eventDestroy(event);
}

// ── Device Attributes / Version / Memory Info ────────────────────────────

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
  return vgre::api::CUDAInterceptor::instance().deviceGetAttribute(value, attr,
                                                                   device);
}

cudaError_t cudaDriverGetVersion(int *version) {
  return vgre::api::CUDAInterceptor::instance().driverGetVersion(version);
}

cudaError_t cudaRuntimeGetVersion(int *version) {
  return vgre::api::CUDAInterceptor::instance().runtimeGetVersion(version);
}

cudaError_t cudaMemGetInfo(size_t *freeBytes, size_t *totalBytes) {
  return vgre::api::CUDAInterceptor::instance().memGetInfo(freeBytes,
                                                           totalBytes);
}

// ── Kernel Registration & Launch ───────────────────────────────────────────

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

// ── Cooperative Launch ────────────────────────────────────────────────────

cudaError_t cudaLaunchCooperativeKernel(const void *hostFun, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
  // Cooperative launch uses the same path as regular launch in VGRE,
  // since grid-wide synchronization is handled via serialized block phases.
  return cudaLaunchKernel(hostFun, gridDim, blockDim, args, sharedMem, stream);
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

} // extern "C"
