/**
 * VGRE Integration Test — static __shared__ support
 *
 * Verifies that static __shared__ arrays are backed by shared memory even
 * when sharedMem size is not explicitly provided at launch.
 */
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

static constexpr float EPSILON = 1e-4f;

void test_static_shared_sum() {
  std::cout << "\n--- Test: static __shared__ sum ---\n";

  core::RuntimeEngine engine;
  VGREResult r = engine.initialize();
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const int threadsPerBlock = 128;
  const int blocks = 2;

  auto &mm = engine.getMemoryManager();
  MemoryHandle devOut = nullptr;
  r = mm.allocate(blocks * sizeof(float), devOut);
  (void)r;
  assert(r == VGREResult::SUCCESS && devOut != nullptr);

  const char *kernelSource = R"(
    __global__ void static_shared_sum(float* out) {
      __shared__ float smem[128];
      int tid = threadIdx.x;
      smem[tid] = (float)tid;
      __syncthreads();
      if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
          sum += smem[i];
        }
        out[blockIdx.x] = sum;
      }
    }
  )";

  void *args[] = {&devOut};
  dim3 blockDim(threadsPerBlock);
  dim3 gridDim(blocks);

  // sharedMem size intentionally 0 to validate static shared auto-allocation
  r = engine.launchKernel("static_shared_sum", kernelSource, gridDim, blockDim,
                          args, 0);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  engine.synchronize();

  std::vector<float> hostOut(blocks, 0.0f);
  r = mm.copyDeviceToHost(hostOut.data(), devOut, blocks * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const float expected = (threadsPerBlock - 1) * threadsPerBlock / 2.0f;
  for (int i = 0; i < blocks; ++i) {
    if (std::fabs(hostOut[i] - expected) > EPSILON) {
      std::cerr << "  Mismatch at block " << i << ": " << hostOut[i]
                << " != " << expected << "\n";
    }
    assert(std::fabs(hostOut[i] - expected) <= EPSILON);
  }

  std::cout << "  ✓ static __shared__ sum verified\n";

  mm.free(devOut);
  engine.shutdown();

  std::cout << "[PASS] static __shared__ support\n";
}

int main() {
  test_static_shared_sum();
  return 0;
}
