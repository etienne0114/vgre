/**
 * Test: HIP/ROCm compatibility layer (src/hip/hip_runtime.cpp)
 *
 * End-to-end verification that a real HIP program runs on VGRE:
 *   device queries → memory round-trips → streams/async → events →
 *   hipModuleLoadData(kernel source) → hipModuleGetFunction →
 *   hipModuleLaunchKernel → numerically-verified result.
 *
 * Every hip* entry delegates to VGRE's real cuda* / cu* implementation; this
 * test proves the delegation is ABI-correct (handle marshalling, memcpy kinds,
 * stream ids) rather than re-testing the underlying compute paths.
 */

#include "vgre/hip/hip_runtime.h"

// Tests build in Release (-DNDEBUG); keep the value asserts real.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_HIP(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "[FAIL] %s → %d (%s) at %s:%d\n", #expr, _e,        \
                   hipGetErrorString(_e), __FILE__, __LINE__);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char *kVecAddSource = R"(
extern "C" __global__ void hip_vec_add(const float* a, const float* b,
                                       float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
)";

int main() {
  std::printf("\n--- Test: HIP runtime compatibility layer ---\n");

  // ── Device management ──────────────────────────────────────────────────
  int count = 0;
  CHECK_HIP(hipGetDeviceCount(&count));
  assert(count >= 1);

  CHECK_HIP(hipSetDevice(0));
  int dev = -1;
  CHECK_HIP(hipGetDevice(&dev));
  assert(dev == 0);

  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, 0));
  assert(prop.name[0] != '\0');
  assert(prop.maxThreadsPerBlock >= 256);
  assert(prop.warpSize == 32);
  assert(prop.totalGlobalMem > 0);
  std::printf("[PASS] device: %s (%s), %d SMs\n", prop.name, prop.gcnArchName,
              prop.multiProcessorCount);

  int maxThreads = 0;
  CHECK_HIP(hipDeviceGetAttribute(&maxThreads,
                                  hipDeviceAttributeMaxThreadsPerBlock, 0));
  assert(maxThreads == prop.maxThreadsPerBlock);
  std::printf("[PASS] device queries\n");

  // ── Memory round-trip ──────────────────────────────────────────────────
  const int n = 4096;
  const size_t bytes = n * sizeof(float);
  std::vector<float> hostA(n), hostB(n), hostC(n, 0.f);
  for (int i = 0; i < n; ++i) { hostA[i] = 0.5f * i; hostB[i] = 2.0f * i; }

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_HIP(hipMalloc(&dA, bytes));
  CHECK_HIP(hipMalloc(&dB, bytes));
  CHECK_HIP(hipMalloc(&dC, bytes));
  CHECK_HIP(hipMemcpy(dA, hostA.data(), bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dB, hostB.data(), bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(dC, 0, bytes));

  // D2D + D2H round-trip check
  CHECK_HIP(hipMemcpyDtoD(dC, dA, bytes));
  CHECK_HIP(hipMemcpyDtoH(hostC.data(), dC, bytes));
  for (int i = 0; i < n; ++i) assert(hostC[i] == hostA[i]);
  std::printf("[PASS] memory round-trip (H2D/D2D/D2H/memset)\n");

  size_t freeB = 0, totalB = 0;
  CHECK_HIP(hipMemGetInfo(&freeB, &totalB));
  assert(totalB > 0 && freeB <= totalB);

  // ── Streams + async ops ────────────────────────────────────────────────
  hipStream_t stream = nullptr;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemsetAsync(dC, 0, bytes, stream));
  CHECK_HIP(hipMemcpyAsync(dC, dB, bytes, hipMemcpyDeviceToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  assert(hipStreamQuery(stream) == hipSuccess);
  CHECK_HIP(hipMemcpy(hostC.data(), dC, bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) assert(hostC[i] == hostB[i]);
  std::printf("[PASS] stream create/async/sync/query\n");

  // ── Events ─────────────────────────────────────────────────────────────
  hipEvent_t start = nullptr, stop = nullptr;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreateWithFlags(&stop, hipEventDefault));
  CHECK_HIP(hipEventRecord(start, stream));
  CHECK_HIP(hipMemcpyAsync(dC, dA, bytes, hipMemcpyDeviceToDevice, stream));
  CHECK_HIP(hipEventRecord(stop, stream));
  CHECK_HIP(hipEventSynchronize(stop));
  float ms = -1.f;
  CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
  assert(ms >= 0.f);
  CHECK_HIP(hipEventDestroy(start));
  CHECK_HIP(hipEventDestroy(stop));
  std::printf("[PASS] events record/sync/elapsed (%.4f ms)\n", ms);

  // ── Module load + JIT kernel launch ────────────────────────────────────
  hipModule_t mod = nullptr;
  CHECK_HIP(hipModuleLoadData(&mod, kVecAddSource));
  hipFunction_t fn = nullptr;
  CHECK_HIP(hipModuleGetFunction(&fn, mod, "hip_vec_add"));
  assert(fn != nullptr);

  int nArg = n;
  void *params[] = {&dA, &dB, &dC, &nArg};
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;
  CHECK_HIP(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1,
                                  0, stream, params, nullptr));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(hostC.data(), dC, bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    float expect = hostA[i] + hostB[i];
    if (std::fabs(hostC[i] - expect) > 1e-6f) {
      std::fprintf(stderr, "[FAIL] c[%d] = %f, expected %f\n", i, hostC[i], expect);
      return 1;
    }
  }
  std::printf("[PASS] hipModule JIT kernel launch (vec_add %d elems verified)\n", n);

  CHECK_HIP(hipModuleUnload(mod));
  CHECK_HIP(hipStreamDestroy(stream));
  CHECK_HIP(hipFree(dA));
  CHECK_HIP(hipFree(dB));
  CHECK_HIP(hipFree(dC));

  // ── Pinned host memory ─────────────────────────────────────────────────
  void *pinned = nullptr;
  CHECK_HIP(hipHostMalloc(&pinned, 1024, hipHostMallocDefault));
  std::memset(pinned, 0xAB, 1024);
  CHECK_HIP(hipHostFree(pinned));
  std::printf("[PASS] pinned host memory\n");

  std::printf("[PASS] HIP runtime compatibility layer\n");
  return 0;
}
