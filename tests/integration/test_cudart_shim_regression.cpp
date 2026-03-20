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

  // Regression: null host function must return invalid value.
  assert(cudaLaunchKernel(nullptr, {1, 1, 1}, {1, 1, 1}, nullptr, 0, 0) == vgre::api::cudaErrorInvalidValue);

  // Regression: unregistered host function must return invalid device function
  // instead of attempting a fallback launch path.
  assert(cudaLaunchKernel(reinterpret_cast<const void *>(0x1), {1, 1, 1}, {1, 1, 1}, nullptr, 0, 0) == vgre::api::cudaErrorInvalidDeviceFunction);

  std::cout << "[PASS] CUDART shim regression guards validated\n";
  return 0;
}
