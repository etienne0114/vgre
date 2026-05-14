// CUDA Driver API — cuda driver module

#include "cuda_driver_internal.h"

extern "C" {

extern "C" void *vgre_register_module_data(const void *data, size_t size);
extern "C" bool vgre_unregister_module_data(void *handle);
extern "C" const char *vgre_get_module_source(void *handle);
extern "C" void *vgre_lookup_symbol(void *handle, const char *name, size_t *size);
extern "C" void *vgre_lookup_texture_ref(void *handle, const char *name);

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    
    const uint32_t ELF_MAGIC = 0x464c457f;
    const uint32_t *header = reinterpret_cast<const uint32_t *>(image);
    
    size_t len = 0;
    if (header[0] == ELF_MAGIC) {
        // Authoritative: Parse ELF headers to find the exact image size
        vgre::common::ELFReader reader(image, 1024 * 1024); // Initial scan size
        len = reader.getTotalSize();
        if (len == 0) len = 8 * 1024 * 1024; // Fallback to conservative estimate
    } else {
        len = std::strlen(reinterpret_cast<const char *>(image));
    }

    if (len == 0) return CUDA_ERROR_INVALID_VALUE;
    void *h = vgre_register_module_data(image, len);
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    *module = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, int *options, void **optionValues) {
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return cuModuleLoadData(module, image);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  if (!dptr || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  void *ptr = vgre_lookup_symbol(hmod, name, bytes);
  if (!ptr) return CUDA_ERROR_UNKNOWN;
  *dptr = ptr;
  return CUDA_SUCCESS;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().moduleLoad(module, fname);
  return toCU(err);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  // If module is from memory PTX, compile source and register kernel on the fly.
  const char *src = vgre_get_module_source(hmod);
  if (src && std::strlen(src) > 0) {
    vgre::KernelId id = 0;
    auto r = vgre::core::RuntimeEngine::instance().registerKernel(name, src, id);
    if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_UNKNOWN;
    *hfunc = static_cast<CUfunction>(id);
    return CUDA_SUCCESS;
  }

  auto err = vgre::api::CUDAInterceptor::instance().moduleGetFunction(hfunc, hmod, name);
  return toCU(err);
}

CUresult cuModuleUnload(CUmodule hmod) {
  if (!hmod) return CUDA_ERROR_INVALID_VALUE;
  if (vgre_unregister_module_data(hmod)) {
    return CUDA_SUCCESS;
  }
  auto err = vgre::api::CUDAInterceptor::instance().moduleUnload(hmod);
  return toCU(err);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
  (void)extra;
  if (f == 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::dim3 grid(gridDimX, gridDimY, gridDimZ);
  vgre::dim3 block(blockDimX, blockDimY, blockDimZ);
  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      static_cast<vgre::KernelId>(f), grid, block, kernelParams, sharedMemBytes, hStream);
  return (r == vgre::VGREResult::SUCCESS) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

} // extern "C"
