#include "vgre/api/cuda_interceptor.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace vgre;
using namespace vgre::api;

void checkCuda(cudaError_t result, const char *func) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error in " << func << ": " << (int)result << std::endl;
    exit(1);
  }
}

#define CHECK_CUDA(ans)                                                        \
  {                                                                            \
    checkCuda((ans), #ans);                                                    \
  }

// vec_add: C[i] = A[i] + B[i]
const char *vec_add_source = R"(
extern "C" __global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)";

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — Stream Concurrency" << std::endl;
  std::cout << "=============================================" << std::endl;

  auto &interceptor = CUDAInterceptor::instance();
  CHECK_CUDA(interceptor.init());

  const int N = 100000;
  size_t size = N * sizeof(float);

  // ── Allocate device memory for two independent vector additions ──
  float *d_a1, *d_b1, *d_c1;
  float *d_a2, *d_b2, *d_c2;
  CHECK_CUDA(interceptor.malloc((void **)&d_a1, size));
  CHECK_CUDA(interceptor.malloc((void **)&d_b1, size));
  CHECK_CUDA(interceptor.malloc((void **)&d_c1, size));
  CHECK_CUDA(interceptor.malloc((void **)&d_a2, size));
  CHECK_CUDA(interceptor.malloc((void **)&d_b2, size));
  CHECK_CUDA(interceptor.malloc((void **)&d_c2, size));

  // ── Initialize input data ──
  for (int i = 0; i < N; ++i) {
    d_a1[i] = static_cast<float>(i);
    d_b1[i] = static_cast<float>(i * 2);
    d_a2[i] = static_cast<float>(i * 3);
    d_b2[i] = static_cast<float>(i * 4);
    d_c1[i] = 0.0f;
    d_c2[i] = 0.0f;
  }

  cudaStream_t stream1 = 0;
  cudaStream_t stream2 = 0;
  CHECK_CUDA(interceptor.streamCreate(&stream1));
  CHECK_CUDA(interceptor.streamCreate(&stream2));

  std::cout << "Launching vec_add on Stream 1 and Stream 2..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  dim3 grid((N + 255) / 256, 1, 1);
  dim3 block(256, 1, 1);
  int n = N;

  // Launch on stream 1: C1 = A1 + B1
  void *args1[] = {&d_a1, &d_b1, &d_c1, &n};
  CHECK_CUDA(interceptor.launchKernel("vec_add", vec_add_source, grid, block,
                                      args1, 0, stream1));

  // Launch on stream 2: C2 = A2 + B2
  void *args2[] = {&d_a2, &d_b2, &d_c2, &n};
  CHECK_CUDA(interceptor.launchKernel("vec_add", vec_add_source, grid, block,
                                      args2, 0, stream2));

  std::cout << "Kernels launched asynchronously. Synchronizing..." << std::endl;

  CHECK_CUDA(interceptor.streamSynchronize(stream1));
  CHECK_CUDA(interceptor.streamSynchronize(stream2));
  assert(interceptor.streamSynchronize(999999) != cudaSuccess);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "Total execution time: " << diff.count() << " seconds"
            << std::endl;

  // ── Verify correctness ──
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float expected1 = static_cast<float>(i) + static_cast<float>(i * 2);
    float expected2 = static_cast<float>(i * 3) + static_cast<float>(i * 4);

    if (std::fabs(d_c1[i] - expected1) > 1e-3f) {
      if (errors < 5)
        std::cerr << "Stream1 error at " << i << ": got " << d_c1[i]
                  << " expected " << expected1 << std::endl;
      errors++;
    }
    if (std::fabs(d_c2[i] - expected2) > 1e-3f) {
      if (errors < 5)
        std::cerr << "Stream2 error at " << i << ": got " << d_c2[i]
                  << " expected " << expected2 << std::endl;
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "[PASS] Stream Concurrency — all " << N
              << " elements verified correct on both streams!" << std::endl;
  } else {
    std::cerr << "[FAIL] Stream Concurrency — " << errors
              << " verification errors" << std::endl;
    return 1;
  }

  interceptor.free(d_a1);
  interceptor.free(d_b1);
  interceptor.free(d_c1);
  interceptor.free(d_a2);
  interceptor.free(d_b2);
  interceptor.free(d_c2);
  CHECK_CUDA(interceptor.streamDestroy(stream1));
  CHECK_CUDA(interceptor.streamDestroy(stream2));

  return 0;
}
