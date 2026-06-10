/**
 * VGRE Integration Test — device-side cuRAND (Track W)
 *
 * Verifies that a JIT-compiled __global__ kernel can call curand_init,
 * curand_uniform and curand_normal (the device cuRAND API provided by
 * cpu_cuda_env.h → cuda_device_libs/curand_kernel.h) and produce statistically
 * sound streams: uniform samples lie in (0,1] with mean ≈ 0.5, normal samples
 * have mean ≈ 0 and stddev ≈ 1.  Also checks per-thread determinism: the same
 * (seed, sequence) reproduces the same value.
 */
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vgre;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Device-side cuRAND Tests (Track W) ===\n");

    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) {
        printf("  FAIL  runtime init\n");
        return 1;
    }

    const int threadsPerBlock = 64;
    const int blocks          = 64;
    const int N               = threadsPerBlock * blocks;  // 4096 samples

    auto& mm = engine.getMemoryManager();
    MemoryHandle devUni = nullptr, devNrm = nullptr;
    bool okAlloc =
        mm.allocate(N * sizeof(float), devUni) == VGREResult::SUCCESS &&
        mm.allocate(N * sizeof(float), devNrm) == VGREResult::SUCCESS;
    check("device buffers allocated", okAlloc && devUni && devNrm);

    const char* kernelSource = R"(
        __global__ void curand_kernel_test(float* uni, float* nrm,
                                           unsigned long long seed) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            curandState state;
            curand_init(seed, tid, 0, &state);
            uni[tid] = curand_uniform(&state);
            nrm[tid] = curand_normal(&state);
        }
    )";

    unsigned long long seed = 1234ULL;
    void* args[] = {&devUni, &devNrm, &seed};
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocks);

    VGREResult r = engine.launchKernel("curand_kernel_test", kernelSource,
                                       gridDim, blockDim, args, 0);
    check("kernel launch succeeded", r == VGREResult::SUCCESS);
    engine.synchronize();

    std::vector<float> uni(N, 0.0f), nrm(N, 0.0f);
    mm.copyDeviceToHost(uni.data(), devUni, N * sizeof(float));
    mm.copyDeviceToHost(nrm.data(), devNrm, N * sizeof(float));

    // ── Uniform: range (0,1] and mean ≈ 0.5 ──────────────────────────────────
    bool rangeOk = true;
    double usum = 0.0;
    for (float v : uni) {
        if (v <= 0.0f || v > 1.0001f) rangeOk = false;
        usum += v;
    }
    double umean = usum / N;
    check("uniform samples in (0,1]", rangeOk);
    check("uniform mean ~ 0.5", std::fabs(umean - 0.5) < 0.03);

    // Not all identical (the generator actually varies per thread).
    bool varies = false;
    for (int i = 1; i < N; ++i) if (uni[i] != uni[0]) { varies = true; break; }
    check("uniform stream varies across threads", varies);

    // ── Normal: mean ≈ 0, stddev ≈ 1 ─────────────────────────────────────────
    double nsum = 0.0;
    for (float v : nrm) nsum += v;
    double nmean = nsum / N;
    double nvar = 0.0;
    for (float v : nrm) nvar += (v - nmean) * (v - nmean);
    nvar /= N;
    double nstd = std::sqrt(nvar);
    check("normal mean ~ 0", std::fabs(nmean) < 0.1);
    check("normal stddev ~ 1", std::fabs(nstd - 1.0) < 0.15);

    // ── Determinism: same (seed, seq) → same value ───────────────────────────
    MemoryHandle devUni2 = nullptr, devNrm2 = nullptr;
    mm.allocate(N * sizeof(float), devUni2);
    mm.allocate(N * sizeof(float), devNrm2);
    void* args2[] = {&devUni2, &devNrm2, &seed};
    engine.launchKernel("curand_kernel_test", kernelSource, gridDim, blockDim, args2, 0);
    engine.synchronize();
    std::vector<float> uni2(N, 0.0f);
    mm.copyDeviceToHost(uni2.data(), devUni2, N * sizeof(float));
    bool deterministic = true;
    for (int i = 0; i < N; ++i) if (uni[i] != uni2[i]) { deterministic = false; break; }
    check("same seed reproduces same stream", deterministic);

    mm.free(devUni);  mm.free(devNrm);
    mm.free(devUni2); mm.free(devNrm2);
    engine.shutdown();

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
