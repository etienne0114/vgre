/**
 * VGRE Integration Test — Vector Addition End-to-End
 *
 * Tests the full pipeline:
 *   1. Create virtual GPU device
 *   2. Allocate device memory
 *   3. Copy data host → device
 *   4. Register and launch vector addition kernel
 *   5. Copy results device → host
 *   6. Verify correctness
 */
#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

static constexpr float EPSILON = 1e-5f;

// ── Test via RuntimeEngine directly ────────────────────────────────────────
void test_runtime_engine_vector_add() {
  std::cout << "\n--- Test: RuntimeEngine vector addition ---\n";

  core::RuntimeEngine engine;
  VGREResult r = engine.initialize();
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const int N = 4096;
  std::vector<float> hostA(N), hostB(N), hostC(N, 0.0f);

  for (int i = 0; i < N; ++i) {
    hostA[i] = static_cast<float>(i);
    hostB[i] = static_cast<float>(i) * 2.0f;
  }

  // Allocate device memory
  auto &mm = engine.getMemoryManager();
  MemoryHandle devA = nullptr;
  MemoryHandle devB = nullptr;
  MemoryHandle devC = nullptr;
  r = mm.allocate(N * sizeof(float), devA);
  (void)r;
  assert(r == VGREResult::SUCCESS && devA != nullptr);
  r = mm.allocate(N * sizeof(float), devB);
  (void)r;
  assert(r == VGREResult::SUCCESS && devB != nullptr);
  r = mm.allocate(N * sizeof(float), devC);
  (void)r;
  assert(r == VGREResult::SUCCESS && devC != nullptr);

  // Copy to device
  r = mm.copyHostToDevice(devA, hostA.data(), N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);
  r = mm.copyHostToDevice(devB, hostB.data(), N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  // Kernel source
  const char *kernelSource = R"(
        __global__ void vector_add(float* A, float* B, float* C, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) {
                C[i] = A[i] + B[i];
            }
        }
    )";

  // Launch
  int n = N;
  void *args[] = {&devA, &devB, &devC, &n};
  dim3 blockDim(256);
  dim3 gridDim((N + 255) / 256);

  auto start = std::chrono::steady_clock::now();
  r = engine.launchKernel("vector_add", kernelSource, gridDim, blockDim, args);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  engine.synchronize();
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "  Execution time: " << ms << " ms\n";

  // Copy back
  r = mm.copyDeviceToHost(hostC.data(), devC, N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  // Verify
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float expected = hostA[i] + hostB[i];
    if (std::fabs(hostC[i] - expected) > EPSILON) {
      if (errors < 5) {
        std::cerr << "  Mismatch at " << i << ": " << hostC[i]
                  << " != " << expected << "\n";
      }
      errors++;
    }
  }

  assert(errors == 0);
  std::cout << "  ✓ All " << N << " elements verified correctly\n";

  // Cleanup
  mm.free(devA);
  mm.free(devB);
  mm.free(devC);
  engine.shutdown();

  std::cout << "[PASS] RuntimeEngine vector addition\n";
}

// ── Test via CUDA Interceptor API ──────────────────────────────────────────
void test_cuda_api_vector_add() {
  std::cout << "\n--- Test: CUDA API vector addition ---\n";

  auto &cuda = api::CUDAInterceptor::instance();
  api::cudaError_t err = cuda.init();
  (void)err;
  assert(err == api::cudaSuccess);

  // Query device
  int count = 0;
  err = cuda.getDeviceCount(&count);
  (void)err;
  assert(err == api::cudaSuccess);
  assert(count >= 1);

  api::cudaDeviceProp_t prop;
  err = cuda.getDeviceProperties(&prop, 0);
  (void)err;
  assert(err == api::cudaSuccess);
  std::cout << "  Device: " << prop.name << "\n";
  std::cout << "  VRAM: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
  std::cout << "  SMs: " << prop.multiProcessorCount << "\n";

  // Allocate
  const int N = 2048;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  err = cuda.malloc(reinterpret_cast<void **>(&dA), N * sizeof(float));
  (void)err;
  assert(err == api::cudaSuccess && dA != nullptr);
  err = cuda.malloc(reinterpret_cast<void **>(&dB), N * sizeof(float));
  (void)err;
  assert(err == api::cudaSuccess && dB != nullptr);
  err = cuda.malloc(reinterpret_cast<void **>(&dC), N * sizeof(float));
  (void)err;
  assert(err == api::cudaSuccess && dC != nullptr);

  // Host arrays
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    hA[i] = static_cast<float>(i) * 0.5f;
    hB[i] = static_cast<float>(N - i) * 0.3f;
  }

  // H2D
  err = cuda.memcpy(dA, hA.data(), N * sizeof(float),
                    api::cudaMemcpyHostToDevice);
  (void)err;
  assert(err == api::cudaSuccess);
  err = cuda.memcpy(dB, hB.data(), N * sizeof(float),
                    api::cudaMemcpyHostToDevice);
  (void)err;
  assert(err == api::cudaSuccess);

  // Launch
  const char *src = R"(
        __global__ void vadd(float* A, float* B, float* C, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) C[i] = A[i] + B[i];
        }
    )";

  int n = N;
  void *args[] = {&dA, &dB, &dC, &n};
  err = cuda.launchKernel("vadd", src, dim3((N + 255) / 256), dim3(256), args);
  (void)err;
  assert(err == api::cudaSuccess);
  err = cuda.deviceSynchronize();
  (void)err;
  assert(err == api::cudaSuccess);

  // D2H
  err = cuda.memcpy(hC.data(), dC, N * sizeof(float),
                    api::cudaMemcpyDeviceToHost);
  (void)err;
  assert(err == api::cudaSuccess);

  // Verify
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float expected = hA[i] + hB[i];
    if (std::fabs(hC[i] - expected) > EPSILON) {
      errors++;
    }
  }
  (void)errors;
  assert(errors == 0);
  std::cout << "  ✓ All " << N << " elements correct via CUDA API\n";

  // Cleanup
  cuda.free(dA);
  cuda.free(dB);
  cuda.free(dC);

  std::cout << "[PASS] CUDA API vector addition\n";
}

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — Vector Addition    " << std::endl;
  std::cout << "=============================================" << std::endl;

  test_runtime_engine_vector_add();
  test_cuda_api_vector_add();

  std::cout << "\n✓ All integration tests passed!" << std::endl;
  return 0;
}
