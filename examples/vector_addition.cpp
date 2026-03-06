/**
 * VGRE Example — Vector Addition
 *
 * Demonstrates: C[i] = A[i] + B[i]
 * through the full VGRE pipeline.
 */
#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

int main() {
  std::cout << "╔═══════════════════════════════════════════╗\n";
  std::cout << "║  VGRE Example: Vector Addition            ║\n";
  std::cout << "╚═══════════════════════════════════════════╝\n\n";

  // Initialize CUDA-compatible API
  auto &cuda = api::CUDAInterceptor::instance();
  cuda.init();

  // Query device
  api::cudaDeviceProp_t prop;
  cuda.getDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "VRAM:   " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
  std::cout << "SMs:    " << prop.multiProcessorCount << "\n\n";

  // Problem size
  const int N = 1 << 20; // 1M elements
  std::cout << "Vector size: " << N << " elements (" << N * sizeof(float) / 1024
            << " KB)\n";

  // Host arrays
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    hA[i] = static_cast<float>(i) * 0.001f;
    hB[i] = static_cast<float>(N - i) * 0.001f;
  }

  // Device allocation
  float *dA, *dB, *dC;
  cuda.malloc(reinterpret_cast<void **>(&dA), N * sizeof(float));
  cuda.malloc(reinterpret_cast<void **>(&dB), N * sizeof(float));
  cuda.malloc(reinterpret_cast<void **>(&dC), N * sizeof(float));

  // Host → Device
  cuda.memcpy(dA, hA.data(), N * sizeof(float), api::cudaMemcpyHostToDevice);
  cuda.memcpy(dB, hB.data(), N * sizeof(float), api::cudaMemcpyHostToDevice);

  // Kernel source
  const char *src = R"(
        __global__ void vector_add(float* A, float* B, float* C, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) C[i] = A[i] + B[i];
        }
    )";

  // Launch
  int n = N;
  void *args[] = {&dA, &dB, &dC, &n};
  dim3 block(256);
  dim3 grid((N + 255) / 256);

  std::cout << "Grid:   " << grid.x << " blocks\n";
  std::cout << "Block:  " << block.x << " threads\n\n";

  auto start = std::chrono::steady_clock::now();
  cuda.launchKernel("vector_add", src, grid, block, args);
  cuda.deviceSynchronize();
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Execution time: " << ms << " ms\n";
  std::cout << "Throughput: " << (3.0 * N * sizeof(float)) / (ms / 1000.0) / 1e9
            << " GB/s\n\n";

  // Device → Host
  cuda.memcpy(hC.data(), dC, N * sizeof(float), api::cudaMemcpyDeviceToHost);

  // Verify
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float expected = hA[i] + hB[i];
    if (std::fabs(hC[i] - expected) > 1e-4f) {
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "✓ Verification PASSED (" << N << " elements)\n";
  } else {
    std::cout << "✗ Verification FAILED (" << errors << " errors)\n";
  }

  // Cleanup
  cuda.free(dA);
  cuda.free(dB);
  cuda.free(dC);

  return errors > 0 ? 1 : 0;
}
