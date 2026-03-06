#include "vgre/api/cuda_interceptor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::api;

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — UVM Managed Memory  " << std::endl;
  std::cout << "=============================================" << std::endl;

  CUDAInterceptor &cuda = CUDAInterceptor::instance();
  cuda.init();

  const int N = 1024;
  size_t size = N * sizeof(float);
  float *a, *b, *c;

  // Allocate managed memory
  std::cout << "--- Step 1: Allocating managed memory ---" << std::endl;
  if (cuda.mallocManaged((void **)&a, size) != cudaSuccess)
    return 1;
  if (cuda.mallocManaged((void **)&b, size) != cudaSuccess)
    return 1;
  if (cuda.mallocManaged((void **)&c, size) != cudaSuccess)
    return 1;

  // Initialize on host (should trigger page faults)
  std::cout << "--- Step 2: Initializing on host (triggers faults) ---"
            << std::endl;
  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 2);
  }

  // Launch kernel
  std::cout << "--- Step 3: Launching kernel using managed pointers ---"
            << std::endl;
  std::string source = R"(
        extern "C" __global__ void vadd(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    )";

  dim3 grid(4, 1, 1);
  dim3 block(256, 1, 1);
  void *args[] = {&a, &b, &c, (void *)&N};

  if (cuda.launchKernel("vadd", source, grid, block, args) != cudaSuccess) {
    std::cerr << "Kernel launch failed" << std::endl;
    return 1;
  }

  cuda.deviceSynchronize();

  // Verify result on host (without memcpy)
  std::cout << "--- Step 4: Verifying results on host ---" << std::endl;
  int errors = 0;
  for (int i = 0; i < N; i++) {
    float expected = static_cast<float>(i + i * 2);
    if (std::abs(c[i] - expected) > 1e-5) {
      errors++;
      if (errors < 5) {
        std::cout << "Error at " << i << ": got " << c[i] << ", expected "
                  << expected << std::endl;
      }
    }
  }

  if (errors == 0) {
    std::cout << "  ✓ All " << N << " elements verified correctly via UVM"
              << std::endl;
    std::cout << "[PASS] UVM Managed Memory test" << std::endl;
  } else {
    std::cout << "  ✗ Verification failed with " << errors << " errors"
              << std::endl;
    return 1;
  }

  cuda.free(a);
  cuda.free(b);
  cuda.free(c);

  return 0;
}
