#include "vgre/api/cuda_interceptor.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::api;

void checkCuda(cudaError_t result, const char *func) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error in " << func << ": "
              << CUDAInterceptor::instance().getErrorString(result)
              << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA(ans)                                                        \
  {                                                                            \
    checkCuda((ans), #ans);                                                    \
  }

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — Multi-Device Support" << std::endl;
  std::cout << "=============================================" << std::endl;

  auto &interceptor = CUDAInterceptor::instance();
  CHECK_CUDA(interceptor.init());

  int deviceCount = 0;
  CHECK_CUDA(interceptor.getDeviceCount(&deviceCount));
  std::cout << "Detected " << deviceCount << " virtual GPU device(s)"
            << std::endl;

  if (deviceCount < 2) {
    std::cerr << "Error: Expected at least 2 virtual devices for this test."
              << std::endl;
    return 1;
  }

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp_t prop;
    CHECK_CUDA(interceptor.getDeviceProperties(&prop, i));
    std::cout << "Device " << i << ": " << prop.name
              << " | VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB"
              << std::endl;
  }

  // Step 1: Default device should be 0
  int currentDevice = -1;
  CHECK_CUDA(interceptor.getDevice(&currentDevice));
  std::cout << "Initial current device: " << currentDevice << std::endl;
  assert(currentDevice == 0);

  // Step 2: Switch to device 1
  std::cout << "Switching to device 1..." << std::endl;
  CHECK_CUDA(interceptor.setDevice(1));
  CHECK_CUDA(interceptor.getDevice(&currentDevice));
  std::cout << "Current device after setDevice(1): " << currentDevice
            << std::endl;
  assert(currentDevice == 1);

  // Step 3: Test memory allocation on different devices
  void *ptr0 = nullptr;
  void *ptr1 = nullptr;

  CHECK_CUDA(interceptor.setDevice(0));
  CHECK_CUDA(interceptor.malloc(&ptr0, 1024));
  std::cout << "Allocated 1024 bytes on device 0 at " << ptr0 << std::endl;

  CHECK_CUDA(interceptor.setDevice(1));
  CHECK_CUDA(interceptor.malloc(&ptr1, 1024));
  std::cout << "Allocated 1024 bytes on device 1 at " << ptr1 << std::endl;

  CHECK_CUDA(interceptor.free(ptr0));
  CHECK_CUDA(interceptor.free(ptr1));

  std::cout << "[PASS] Multi-Device Support test" << std::endl;

  return 0;
}
