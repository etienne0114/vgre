/**
 * VGRE Integration Test — __syncthreads correctness
 *
 * Ensures that shared memory + __syncthreads behaves correctly for blocks
 * with <= 256 threads (auto block-threading path).
 */
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

static constexpr float EPSILON = 1e-4f;

void test_syncthreads_block_sum() {
  std::cout << "\n--- Test: __syncthreads block sum ---\n";

  core::RuntimeEngine engine;
  VGREResult r = engine.initialize();
  (void)r;
  assert(r == VGREResult::SUCCESS);

  const int threadsPerBlock = 128;
  const int blocks = 4;

  // Output: one value per block
  auto &mm = engine.getMemoryManager();
  MemoryHandle devOut = nullptr;
  r = mm.allocate(blocks * sizeof(float), devOut);
  (void)r;
  assert(r == VGREResult::SUCCESS && devOut != nullptr);

  const char *kernelSource = R"(
    __global__ void block_sum(float* out) {
      float* smem = (float*)sharedMem;
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

  const size_t sharedMemBytes = threadsPerBlock * sizeof(float);
  r = engine.launchKernel("block_sum", kernelSource, gridDim, blockDim, args,
                          sharedMemBytes);
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

  std::cout << "  ✓ __syncthreads block sum verified\n";

  mm.free(devOut);
  engine.shutdown();

  std::cout << "[PASS] __syncthreads correctness\n";
}

int main() {
  test_syncthreads_block_sum();
  return 0;
}
