// Track Z (Zero-Burden Engine): the real CUDA-on-CPU C-ABI path routed through
// the execution backend. With VGRE_EXEC_BACKEND=interpreter, vgre_register_kernel
// compiles the CUDA-C source with VGRE's own front-end and vgre_launch_kernel
// runs it on the interpreter tier — no LLVM/Clang for this kernel. Verifies the
// numerical result AND that the backend path (high-range kernel id) was taken.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/api/vgre_c_api.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static const char* kVecAdd = R"(
extern "C" __global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
)";

int main() {
    // Select the interpreter backend BEFORE init so registration routes to it.
    setenv("VGRE_EXEC_BACKEND", "interpreter", 1);

    CHECK(vgre_init() == VGRE_SUCCESS, "vgre_init");

    const int N = 256;
    const size_t bytes = N * sizeof(float);
    std::vector<float> hA(N), hB(N), hC(N, -1.0f), ref(N);
    for (int i = 0; i < N; ++i) { hA[i] = i * 0.25f; hB[i] = 100.0f - i; ref[i] = hA[i] + hB[i]; }

    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK(vgre_malloc(&dA, bytes) == VGRE_SUCCESS, "malloc A");
    CHECK(vgre_malloc(&dB, bytes) == VGRE_SUCCESS, "malloc B");
    CHECK(vgre_malloc(&dC, bytes) == VGRE_SUCCESS, "malloc C");

    CHECK(vgre_memcpy(dA, hA.data(), bytes, VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D A");
    CHECK(vgre_memcpy(dB, hB.data(), bytes, VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D B");

    uint64_t kid = 0;
    CHECK(vgre_register_kernel("vecAdd", kVecAdd, &kid) == VGRE_SUCCESS, "register vecAdd");
#ifdef VGRE_ENABLE_JIT
    // JIT build: VGRE_EXEC_BACKEND routes registration through the side backend
    // dispatch, whose kernel ids live in a high range (proves the front-end/
    // interpreter path — not the LLVM JIT — handled this kernel).
    CHECK(kid >= 0x4000000000000000ULL, "kernel registered on the backend dispatch");
#else
    // No-LLVM build: the C-ABI routes through the engine's single kernel registry
    // (which itself runs the from-scratch backend — there is no LLVM at all here).
    CHECK(kid != 0, "kernel registered (engine backend, no LLVM)");
#endif

    int n = N;
    void* args[] = {&dA, &dB, &dC, &n};
    uint32_t grid[3]  = {(uint32_t)((N + 63) / 64), 1, 1};
    uint32_t block[3] = {64, 1, 1};
    CHECK(vgre_launch_kernel(kid, grid, block, args, 4, 0, 0) == VGRE_SUCCESS, "launch vecAdd on backend");
    CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");

    CHECK(vgre_memcpy(hC.data(), dC, bytes, VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H C");

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) maxErr = std::fmax(maxErr, std::fabs(hC[i] - ref[i]));
    CHECK(maxErr < 1e-4f, "vecAdd via C-ABI + interpreter backend matches reference");
    std::printf("  backend-dispatch vecAdd max error = %.3e  (kid=0x%llx)\n",
                maxErr, (unsigned long long)kid);

    vgre_free(dA); vgre_free(dB); vgre_free(dC);
    vgre_shutdown();

    if (g_fail == 0)
        std::printf("PASS: C-ABI kernel launch routed through the interpreter backend (no LLVM)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
