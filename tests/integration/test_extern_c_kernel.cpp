/**
 * VGRE Integration Test — extern "C" kernel parsing
 *
 * Ensures KernelParser can handle extern "C" qualifiers in kernel source.
 */
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

static constexpr float EPSILON = 1e-5f;

void test_extern_c_kernel() {
  std::cout << "\n--- Test: extern \"C\" kernel parsing ---\n";

  core::RuntimeEngine engine;
  VGREResult r = engine.initialize();
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const int N = 2048;
  std::vector<float> hostA(N), hostB(N), hostC(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    hostA[i] = static_cast<float>(i);
    hostB[i] = static_cast<float>(i) * 3.0f;
  }

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

  r = mm.copyHostToDevice(devA, hostA.data(), N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);
  r = mm.copyHostToDevice(devB, hostB.data(), N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const char *kernelSource = R"(
    extern "C" __global__ void vadd_extern_c(float* A, float* B, float* C, int N) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < N) {
        C[i] = A[i] + B[i];
      }
    }
  )";

  int n = N;
  void *args[] = {&devA, &devB, &devC, &n};
  dim3 blockDim(256);
  dim3 gridDim((N + 255) / 256);

  r = engine.launchKernel("vadd_extern_c", kernelSource, gridDim, blockDim,
                          args);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  engine.synchronize();

  r = mm.copyDeviceToHost(hostC.data(), devC, N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

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
  std::cout << "  ✓ extern \"C\" kernel parsed and executed\n";

  mm.free(devA);
  mm.free(devB);
  mm.free(devC);
  engine.shutdown();

  std::cout << "[PASS] extern \"C\" kernel parsing\n";
}

int main() {
  test_extern_c_kernel();
  return 0;
}
