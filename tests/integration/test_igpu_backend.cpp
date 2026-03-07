#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace vgre;

const char *vectorAddSource = R"(
extern "C" __global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
)";

int main() {
  vgre_init();

  // Check if iGPU is available natively
  auto resources = advanced::HybridComputeManager::instance().getResources();
  if (!resources.hasIntegratedGPU) {
    std::cout << "[SKIP] Integrated GPU not detected on this system. Cannot "
                 "test native OpenCL iGPU backend."
              << std::endl;
    vgre_shutdown();
    return 0; // Graceful skip
  }

  std::cout << "[TEST] Integrated GPU Available! Device: " << resources.igpuName
            << std::endl;

  const int numElements = 50000;
  const size_t size = numElements * sizeof(float);

  vgre::MemoryHandle d_A, d_B, d_C;
  auto &engine = core::RuntimeEngine::instance();
  auto &memgr = engine.getMemoryManager();

  // Allocate buffers on VGRE (CPUside Unified Memory)
  if (memgr.allocate(size, d_A) != VGREResult::SUCCESS)
    return -1;
  if (memgr.allocate(size, d_B) != VGREResult::SUCCESS)
    return -1;
  if (memgr.allocate(size, d_C) != VGREResult::SUCCESS)
    return -1;

  // Fill host arrays
  float *h_A = reinterpret_cast<float *>(d_A);
  float *h_B = reinterpret_cast<float *>(d_B);

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  // Register kernel
  KernelId kernelId;
  if (engine.registerKernel("vectorAdd", vectorAddSource, kernelId) !=
      VGREResult::SUCCESS) {
    std::cerr << "Failed to register kernel." << std::endl;
    return -1;
  }

  // Setup args
  void *args[] = {&d_A, &d_B, &d_C, (void *)&numElements};

  // Calculate grid
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid, 1, 1);
  dim3 block(threadsPerBlock, 1, 1);

  std::cout << "[TEST] Dispatching vectorAdd to INTEGRATED_GPU..." << std::endl;

  // Launch using the HybridComputeManager pointing specifically at the iGPU
  // backend
  auto result =
      advanced::HybridComputeManager::instance().distributeRegisteredKernel(
          kernelId, grid, block, args, 4, 0,
          advanced::ComputeBackend::INTEGRATED_GPU);

  if (result != VGREResult::SUCCESS) {
    std::cerr << "Failed to dispatch kernel to memory mapped OpenCL backend."
              << std::endl;
    return -1;
  }

  // Verify
  float *h_C = reinterpret_cast<float *>(d_C);
  int errors = 0;
  for (int i = 0; i < numElements; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    if (std::abs(h_C[i] - expected) > 1e-5) {
      std::cerr << "Mismatch at " << i << ": expected " << expected << ", got "
                << h_C[i] << std::endl;
      errors++;
      if (errors > 10)
        break;
    }
  }

  if (errors == 0) {
    std::cout << "[PASS] iGPU OpenCL Zero-Copy Backend executed successfully!"
              << std::endl;
  } else {
    std::cerr << "[FAIL] Output validation failed." << std::endl;
  }

  vgre_shutdown();
  return (errors == 0) ? 0 : -1;
}
