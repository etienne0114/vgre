/**
 * VGRE Integration Test — Texture Driver API
 *
 * Verifies that the CUDA Driver API shims for texture objects
 * correctly interact with the internal TextureManager.
 */
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include <cassert>
#include <iostream>
#include <vector>

// Forward declarations for Driver API shims (linked from libvgre_cudart)
extern "C" {
  using CUresult = int;
  using CUdeviceptr = void*;
  using CUtexObject = uint64_t;

  struct CUDA_RESOURCE_DESC {
    int resType;
    union {
      struct {
        void *devPtr;
        size_t sizeInBytes;
        size_t width;
        size_t height;
        size_t pitchInBytes;
      } res;
    } res;
  };

  struct CUDA_TEXTURE_DESC {
    int addressMode[3];
    int filterMode;
    int readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
  };

  CUresult cuInit(unsigned int flags);
  CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
  CUresult cuMemFree(CUdeviceptr dptr);
  CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                             const CUDA_TEXTURE_DESC *pTexDesc, const void *pResViewDesc);
  CUresult cuTexObjectDestroy(CUtexObject texObject);
}

void test_texture_driver_api() {
  std::cout << "\n--- Test: Texture Driver API ---\n";

  CUresult r = cuInit(0);
  assert(r == 0);

  const size_t width = 64;
  const size_t height = 64;
  const size_t size = width * height * sizeof(float);

  CUdeviceptr dptr = nullptr;
  r = cuMemAlloc(&dptr, size);
  assert(r == 0 && dptr != nullptr);

  CUDA_RESOURCE_DESC resDesc{};
  resDesc.resType = 0x00; // CU_RESOURCETYPE_ARRAY or similar
  resDesc.res.res.devPtr = dptr;
  resDesc.res.res.width = width;
  resDesc.res.res.height = height;
  resDesc.res.res.sizeInBytes = size;

  CUDA_TEXTURE_DESC texDesc{};
  texDesc.addressMode[0] = 1; // CLAMP
  texDesc.filterMode = 1;      // LINEAR

  CUtexObject texObj = 0;
  r = cuTexObjectCreate(&texObj, &resDesc, &texDesc, nullptr);
  assert(r == 0 && texObj != 0);
  std::cout << "  ✓ Texture object created: " << texObj << "\n";

  r = cuTexObjectDestroy(texObj);
  assert(r == 0);
  std::cout << "  ✓ Texture object destroyed\n";

  cuMemFree(dptr);
  std::cout << "[PASS] Texture Driver API\n";
}

int main() {
  test_texture_driver_api();
  return 0;
}
