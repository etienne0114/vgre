/**
 * VGRE Example — Matrix Multiplication
 *
 * Demonstrates: C = A × B (NxN matrices)
 * using 2D grid/block layout.
 */
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;

int main() {
  std::cout << "╔═══════════════════════════════════════════╗\n";
  std::cout << "║  VGRE Example: Matrix Multiplication      ║\n";
  std::cout << "╚═══════════════════════════════════════════╝\n\n";

  core::RuntimeEngine engine;
  engine.initialize();

  const auto &props = engine.getDevice().getProperties();
  std::cout << "Device: " << props.name << "\n\n";

  // Matrix size
  const int N = 256;
  std::cout << "Matrix: " << N << "x" << N << " ("
            << N * N * sizeof(float) / 1024 << " KB each)\n";

  // Host matrices
  std::vector<float> hA(N * N), hB(N * N), hC(N * N, 0.0f);

  for (int i = 0; i < N * N; ++i) {
    hA[i] = static_cast<float>(i % N) * 0.01f;
    hB[i] = static_cast<float>(i / N) * 0.01f;
  }

  // Device memory
  auto &mm = engine.getMemoryManager();
  MemoryHandle dA, dB, dC;
  mm.allocate(N * N * sizeof(float), dA);
  mm.allocate(N * N * sizeof(float), dB);
  mm.allocate(N * N * sizeof(float), dC);

  mm.copyHostToDevice(dA, hA.data(), N * N * sizeof(float));
  mm.copyHostToDevice(dB, hB.data(), N * N * sizeof(float));

  // Kernel source with 2D indexing
  const char *src = R"(
        __global__ void matmul(float* A, float* B, float* C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < N && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    )";

  int n = N;
  void *args[] = {&dA, &dB, &dC, &n};
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (N + 15) / 16);

  std::cout << "Grid:  " << grid.x << "x" << grid.y << " blocks\n";
  std::cout << "Block: " << block.x << "x" << block.y << " threads\n\n";

  auto start = std::chrono::steady_clock::now();
  engine.launchKernel("matmul", src, grid, block, args);
  engine.synchronize();
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
  std::cout << "Execution time: " << ms << " ms\n";
  std::cout << "Performance:    " << gflops << " GFLOPS\n\n";

  // Copy back
  mm.copyDeviceToHost(hC.data(), dC, N * N * sizeof(float));

  // Verify a few elements (full verification is expensive for large N)
  bool correct = true;
  for (int row = 0; row < std::min(N, 4); ++row) {
    for (int col = 0; col < std::min(N, 4); ++col) {
      float expected = 0.0f;
      for (int k = 0; k < N; ++k) {
        expected += hA[row * N + k] * hB[k * N + col];
      }
      if (std::fabs(hC[row * N + col] - expected) > 0.1f) {
        std::cerr << "Mismatch at [" << row << "][" << col
                  << "]: " << hC[row * N + col] << " vs " << expected << "\n";
        correct = false;
      }
    }
  }

  if (correct) {
    std::cout << "✓ Spot-check verification PASSED\n";
  } else {
    std::cout << "✗ Verification FAILED\n";
  }

  // Cleanup
  mm.free(dA);
  mm.free(dB);
  mm.free(dC);
  engine.shutdown();

  return correct ? 0 : 1;
}
