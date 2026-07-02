/**
 * Test: runtime block-threading toggle (vgre_set_block_threads)
 *
 * Verifies that the public C API flips the process-wide flag that
 * JIT-generated launchers consult (vgre_jit_block_threads_enabled), i.e. the
 * call is a real runtime control, not a logged no-op, and that a sync kernel
 * still produces correct results with the flag in both positions.
 */

#include "vgre/api/vgre_c_api.h"
#include "vgre/runtime/gpu_thread_context.h"

#include <cassert>
#include <cstdio>
#include <cstring>

static const char *kSyncKernel = R"(
extern "C" __global__ void block_sum(const float* in, float* out, int n) {
    extern __shared__ float buf[];
    int t = threadIdx.x;
    buf[t] = (t < n) ? in[t] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) buf[t] += buf[t + s];
        __syncthreads();
    }
    if (t == 0) out[0] = buf[0];
}
)";

static int run_block_sum(float *expected) {
  uint64_t kid = 0;
  if (vgre_register_kernel("block_sum", kSyncKernel, &kid) != VGRE_SUCCESS)
    return 1;

  const int n = 64;
  float hostIn[n];
  float sum = 0.f;
  for (int i = 0; i < n; ++i) { hostIn[i] = 1.0f + i; sum += hostIn[i]; }
  *expected = sum;

  void *dIn = nullptr, *dOut = nullptr;
  if (vgre_malloc(&dIn, sizeof(hostIn)) != VGRE_SUCCESS) return 1;
  if (vgre_malloc(&dOut, sizeof(float)) != VGRE_SUCCESS) return 1;
  if (vgre_memcpy(dIn, hostIn, sizeof(hostIn), VGRE_MEMCPY_HOST_TO_DEVICE) != VGRE_SUCCESS)
    return 1;

  int nArg = n;
  void *args[] = {&dIn, &dOut, &nArg};
  uint32_t grid[3] = {1, 1, 1}, block[3] = {n, 1, 1};
  if (vgre_launch_kernel(kid, grid, block, args, 3,
                         n * sizeof(float), 0) != VGRE_SUCCESS)
    return 1;
  if (vgre_synchronize() != VGRE_SUCCESS) return 1;

  float result = 0.f;
  if (vgre_memcpy(&result, dOut, sizeof(float), VGRE_MEMCPY_DEVICE_TO_HOST) != VGRE_SUCCESS)
    return 1;
  vgre_free(dIn);
  vgre_free(dOut);

  if (result != sum) {
    std::fprintf(stderr, "[FAIL] block_sum = %f, expected %f\n", result, sum);
    return 1;
  }
  return 0;
}

int main() {
  std::printf("\n--- Test: block-threads runtime toggle ---\n");

  if (vgre_init() != VGRE_SUCCESS) {
    std::fprintf(stderr, "[FAIL] vgre_init\n");
    return 1;
  }

  // The C API call must actually flip the flag the JIT launchers read.
  using vgre::runtime::vgre_jit_block_threads_enabled;
  assert(vgre_set_block_threads(1) == VGRE_SUCCESS);
  assert(vgre_jit_block_threads_enabled() == 1);
  assert(vgre_set_block_threads(0) == VGRE_SUCCESS);
  assert(vgre_jit_block_threads_enabled() == 0);
  assert(vgre_set_block_threads(1) == VGRE_SUCCESS);
  assert(vgre_jit_block_threads_enabled() == 1);
  std::printf("[PASS] vgre_set_block_threads flips the runtime flag\n");

  // A __syncthreads() reduction must be correct with the flag in both
  // positions (multi-thread blocks always use threaded dispatch; the flag
  // additionally covers single-thread sync-kernel blocks).
  float expected = 0.f;
  assert(vgre_set_block_threads(1) == VGRE_SUCCESS);
  if (run_block_sum(&expected) != 0) return 1;
  std::printf("[PASS] sync-kernel correct with block threads enabled (sum=%g)\n",
              expected);

  assert(vgre_set_block_threads(0) == VGRE_SUCCESS);
  if (run_block_sum(&expected) != 0) return 1;
  std::printf("[PASS] sync-kernel correct with block threads disabled (sum=%g)\n",
              expected);

  vgre_shutdown();
  std::printf("[PASS] block-threads runtime toggle\n");
  return 0;
}
