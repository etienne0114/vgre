#include "vgre/api/vgre_c_api.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// A complex kernel that requires true C++ compilation (loops and conditionals)
const char *COMPLEX_KERNEL = R"(
extern "C" __global__ void complex_op(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = data[i];
        // Complex loop that regex would fail to match
        for (int j = 0; j < 5; ++j) {
            if (val > 10.0f) {
                val = sqrtf(val) + 1.0f;
            } else {
                val = val * 2.0f - 1.0f;
            }
        }
        data[i] = val;
    }
}
)";

int main() {
  vgre_init();

  const int N = 1024;
  size_t size = N * sizeof(float);

  void *h_data = malloc(size);
  float *f_data = (float *)h_data;
  for (int i = 0; i < N; ++i) {
    f_data[i] = (float)i;
  }

  void *d_data;
  vgre_malloc(&d_data, size);
  vgre_memcpy(d_data, h_data, size, VGRE_MEMCPY_HOST_TO_DEVICE);

  // Register the kernel first
  uint64_t kernel_id;
  int reg_res = vgre_register_kernel("complex_op", COMPLEX_KERNEL, &kernel_id);
  if (reg_res != VGRE_SUCCESS) {
    std::cerr << "Kernel registration FAILED!" << std::endl;
    return 1;
  }

  // Launch complex kernel on CPU (AUTO/CPU backend will use
  // LLVMTranslationEngine)
  void *args[] = {&d_data, (int *)&N};
  uint32_t gridDim[3] = {8, 1, 1};
  uint32_t blockDim[3] = {128, 1, 1};

  // Launch using the registered ID
  int launch_res =
      vgre_launch_kernel(kernel_id, gridDim, blockDim, args, 2, 0, 0);
  if (launch_res != VGRE_SUCCESS) {
    std::cerr << "Kernel launch FAILED with code " << launch_res << "!"
              << std::endl;
    return 1;
  }
  vgre_synchronize();

  vgre_memcpy(h_data, d_data, size, VGRE_MEMCPY_DEVICE_TO_HOST);

  // Verify a few values
  for (int i = 0; i < 10; ++i) {
    float expected = (float)i;
    for (int j = 0; j < 5; ++j) {
      if (expected > 10.0f) {
        expected = sqrtf(expected) + 1.0f;
      } else {
        expected = expected * 2.0f - 1.0f;
      }
    }
    std::cout << "Index " << i << ": Original " << i << " -> Result "
              << f_data[i] << " (Expected " << expected << ")" << std::endl;

    if (std::abs(f_data[i] - expected) > 1e-4f) {
      std::cerr << "Verification FAILED at index " << i << "!" << std::endl;
      return 1;
    }
  }

  std::cout << "Complex JIT Kernel verification SUCCESSFUL!" << std::endl;

  vgre_free(d_data);
  free(h_data);
  vgre_shutdown();
  return 0;
}
