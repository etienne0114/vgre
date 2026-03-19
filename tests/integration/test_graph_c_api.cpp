#include "vgre/api/vgre_c_api.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — C API Graphs       " << std::endl;
  std::cout << "=============================================" << std::endl;

  if (vgre_init() != VGRE_SUCCESS) {
    std::cerr << "vgre_init failed" << std::endl;
    return 1;
  }

  const int N = 512;
  size_t size = N * sizeof(float);
  std::vector<float> h_a(N, 1.0f);
  std::vector<float> h_b(N, 2.0f);
  std::vector<float> h_c(N, 0.0f);

  void *d_a = nullptr;
  void *d_b = nullptr;
  void *d_c = nullptr;
  if (vgre_malloc(&d_a, size) != VGRE_SUCCESS ||
      vgre_malloc(&d_b, size) != VGRE_SUCCESS ||
      vgre_malloc(&d_c, size) != VGRE_SUCCESS) {
    std::cerr << "vgre_malloc failed" << std::endl;
    return 1;
  }

  vgre_memcpy(d_a, h_a.data(), size, VGRE_MEMCPY_HOST_TO_DEVICE);
  vgre_memcpy(d_b, h_b.data(), size, VGRE_MEMCPY_HOST_TO_DEVICE);

  const char *source = R"(
    extern "C" __global__ void vadd_capi(float* a, float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        c[i] = a[i] + b[i];
      }
    }
  )";

  uint64_t kernel_id = 0;
  if (vgre_register_kernel("vadd_capi", source, &kernel_id) != VGRE_SUCCESS) {
    std::cerr << "vgre_register_kernel failed" << std::endl;
    return 1;
  }

  uint64_t graph = 0;
  if (vgre_graphCreate(&graph) != VGRE_SUCCESS) {
    std::cerr << "vgre_graphCreate failed" << std::endl;
    return 1;
  }

  uint32_t grid[3] = {2, 1, 1};
  uint32_t block[3] = {256, 1, 1};

  void *args[] = {&d_a, &d_b, &d_c, (void *)&N};
  uint8_t arg_types[] = {VGRE_ARG_POINTER, VGRE_ARG_POINTER, VGRE_ARG_POINTER,
                         VGRE_ARG_INT32};

  uint64_t n1 = 0;
  uint64_t n2 = 0;
  uint64_t n3 = 0;
  vgre_graphAddKernelNodeEx(graph, kernel_id, "vadd_capi", grid, block, args,
                            arg_types, 4, nullptr, 0, &n1);
  uint64_t deps1[] = {n1};
  vgre_graphAddKernelNodeEx(graph, kernel_id, "vadd_capi", grid, block, args,
                            arg_types, 4, deps1, 1, &n2);
  uint64_t deps2[] = {n2};
  vgre_graphAddKernelNodeEx(graph, kernel_id, "vadd_capi", grid, block, args,
                            arg_types, 4, deps2, 1, &n3);

  uint64_t exec = 0;
  if (vgre_graphInstantiate(graph, &exec) != VGRE_SUCCESS) {
    std::cerr << "vgre_graphInstantiate failed" << std::endl;
    return 1;
  }

  vgre_memset(d_c, 0, size);
  vgre_graphLaunch(exec, 0);
  vgre_stream_synchronize(0);

  vgre_memcpy(h_c.data(), d_c, size, VGRE_MEMCPY_DEVICE_TO_HOST);

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (std::abs(h_c[i] - 3.0f) > 1e-5f) {
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "  ✓ C API graph execution verified" << std::endl;
  } else {
    std::cerr << "  ✗ Verification failed with " << errors << " errors"
              << std::endl;
    return 1;
  }

  vgre_graphExecDestroy(exec);
  vgre_graphDestroy(graph);
  vgre_free(d_a);
  vgre_free(d_b);
  vgre_free(d_c);
  vgre_shutdown();

  std::cout << "[PASS] C API Graphs test" << std::endl;
  return 0;
}
