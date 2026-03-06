/**
 * VGRE Test — LD_PRELOAD Shim Validation
 *
 * This test uses STANDARD CUDA API calls (not VGRE ones).
 * It will be compiled WITHOUT linking to VGRE.
 * We will run it with LD_PRELOAD=libvgre_cudart.so to prove
 * the interception works.
 */

#include <cassert>
#include <iostream>
#include <vector>

// Standard CUDART definitions that frameworks expect
extern "C" {
typedef int cudaError_t;
const cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};

struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int warpSize;
  int multiProcessorCount;
  int major;
  int minor;
  int clockRate;
  size_t totalConstMem;
};

cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);
}

int main() {
  std::cout << "--- Testing Standard CUDART Interception ---\n";

  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    std::cerr << "Failed to get device count. Did you forget LD_PRELOAD?\n";
    return 1;
  }

  std::cout << "Detected " << count << " Virtual Devices!\n";

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "VRAM:   " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

  // Test Memory Interception
  const int N = 1000;
  std::vector<float> h_A(N, 42.0f);
  std::vector<float> h_B(N, 0.0f);

  float *d_A = nullptr;
  err = cudaMalloc((void **)&d_A, N * sizeof(float));
  assert(err == cudaSuccess);
  assert(d_A != nullptr);

  std::cout << "Allocated " << N * sizeof(float) << " bytes at vptr: " << d_A
            << "\n";

  err = cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  std::cout << "Host -> Device copy successful.\n";

  err = cudaMemcpy(h_B.data(), d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);

  std::cout << "Device -> Host copy successful. Checking data...\n";

  assert(h_B[0] == 42.0f);
  assert(h_B[N - 1] == 42.0f);

  cudaFree(d_A);
  std::cout << "Freed memory.\n";

  cudaDeviceSynchronize();

  std::cout
      << "[PASS] All intercepted CUDART calls worked flawlessly via VGRE!\n";

  return 0;
}
