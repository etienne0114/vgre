/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

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

class CUDAModuleRegistry {
public:
  static CUDAModuleRegistry &instance() {
    static CUDAModuleRegistry registry;
    return registry;
  }

  void **registerFatBinary(void *fatCubin) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Provide a stable pointer handle representing this abstract module
    auto handle = new ModuleHandleWrapper{nextModuleId_++, fatCubin};
    modules_[handle] = {};
    return reinterpret_cast<void **>(handle);
  }

  void unregisterFatBinary(void **handlePtr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!handlePtr)
      return;
    auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
    if (modules_.find(handle) != modules_.end()) {
      // Clean up host-to-name mappings associated with this module
      for (const auto &funcPtr : modules_[handle]) {
        hostToName_.erase(funcPtr);
      }
      modules_.erase(handle);
      delete handle;
    }
  }

  void registerFunction(void **handlePtr, const void *hostFun,
                        const char *deviceName) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string name =
        deviceName ? deviceName
                   : ("vgre_auto_kernel_" + std::to_string(nextFunctionId_++));
    hostToName_[hostFun] = name;

    if (handlePtr) {
      auto *handle = reinterpret_cast<ModuleHandleWrapper *>(handlePtr);
      if (modules_.find(handle) != modules_.end()) {
        modules_[handle].push_back(hostFun);
      }
    }
  }

  std::string lookupKernelName(const void *hostFun) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostToName_.find(hostFun);
    if (it != hostToName_.end()) {
      return it->second;
    }
    return "vgre_dynamic_kernel_" + std::to_string(nextFunctionId_++);
  }

  void registerVariable(const void *hostVar, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (hostVarToDevicePtr_.find(hostVar) == hostVarToDevicePtr_.end()) {
      void *devPtr = nullptr;
      vgre::api::CUDAInterceptor::instance().malloc(&devPtr, size);
      hostVarToDevicePtr_[hostVar] = devPtr;
    }
  }

  void *lookupVariable(const void *hostVar) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = hostVarToDevicePtr_.find(hostVar);
    if (it != hostVarToDevicePtr_.end()) {
      return it->second;
    }
    return nullptr;
  }

private:
  struct ModuleHandleWrapper {
    uint64_t id;
    void *fatCubin;
  };

  std::mutex mutex_;
  uint64_t nextModuleId_ = 1;
  uint64_t nextFunctionId_ = 1;

  // Maps a ModuleHandle to a list of host function pointers registered to it
  std::unordered_map<ModuleHandleWrapper *, std::vector<const void *>> modules_;
  // Maps a host function pointer directly to its kernel string name
  std::unordered_map<const void *, std::string> hostToName_;
  // Maps a host variable pointer to a device-allocated VGRE memory handle
  std::unordered_map<const void *, void *> hostVarToDevicePtr_;
};

// ── Initialization & Error Handling ────────────────────────────────────────

cudaError_t cudaGetLastError(void) {
  return vgre::api::CUDAInterceptor::instance().getLastError();
}

cudaError_t cudaPeekAtLastError(void) {
  return vgre::api::CUDAInterceptor::instance().getLastError();
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

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  return vgre::api::CUDAInterceptor::instance().memset(devPtr, value, count);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memsetAsync(devPtr, value,
                                                            count, stream);
}

// ── Stream Management ──────────────────────────────────────────────────────

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  return vgre::api::CUDAInterceptor::instance().streamCreate(pStream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
  (void)flags;
  return vgre::api::CUDAInterceptor::instance().streamCreate(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamSynchronize(stream);
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
    CUDAModuleRegistry::instance().registerVariable(hostVar, size);
  }
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  *devPtr = CUDAModuleRegistry::instance().lookupVariable(symbol);
  if (!*devPtr)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *hostFun, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  std::string kernelName =
      CUDAModuleRegistry::instance().lookupKernelName(hostFun);

  vgre::dim3 vgreGrid(gridDim.x, gridDim.y, gridDim.z);
  vgre::dim3 vgreBlock(blockDim.x, blockDim.y, blockDim.z);

  // Launch via VGRE interceptor.
  // The source string includes a pseudo-marker '__attribute__((global))' and an
  // empty body. The VGRE LLVMTranslationEngine parses this in matchMatrixOp(),
  // matchVectorOp(), etc., falling back to CPU parallel simulated execution if
  // a pattern is recognized natively.
  return vgre::api::CUDAInterceptor::instance().launchKernel(
      kernelName,
      "__attribute__((global)) void " + kernelName +
          "() { /* VGRE Emulated Block */ }",
      vgreGrid, vgreBlock, args, sharedMem, stream);
}

} // extern "C"
