#include "vgre/api/cuda_interceptor.h"

#include <cassert>
#include <iostream>

extern "C" {
struct dim3 {
  unsigned int x, y, z;
};

void **__cudaRegisterFatBinary(void *fatCubin);
vgre::api::cudaError_t
cudaLaunchKernel(const void *hostFun, dim3 gridDim, dim3 blockDim, void **args,
                 size_t sharedMem, vgre::api::cudaStream_t stream);
}

int main() {
  std::cout << "=============================================\n";
  std::cout << "  VGRE Integration Test — CUDART Shim Guard\n";
  std::cout << "=============================================\n";

  // Regression: null fatbinary must be rejected safely.
  assert(__cudaRegisterFatBinary(nullptr) == nullptr);

  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};

  // Regression: null host function must return invalid value.
  auto r = cudaLaunchKernel(nullptr, grid, block, nullptr, 0, 0);
  assert(r == vgre::api::cudaErrorInvalidValue);

  // Regression: unregistered host function must return invalid device function
  // instead of attempting a fallback launch path.
  auto fakeHostFun = reinterpret_cast<const void *>(0x1);
  r = cudaLaunchKernel(fakeHostFun, grid, block, nullptr, 0, 0);
  assert(r == vgre::api::cudaErrorInvalidDeviceFunction);

  std::cout << "[PASS] CUDART shim regression guards validated\n";
  return 0;
}
