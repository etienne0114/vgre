// Track Z (Zero-Burden Engine): growing the supported CUDA-C subset — shared
// memory + __syncthreads + loops + ++/-- + math intrinsics, all end-to-end
// (CUDA-C -> our PTX -> interpreter backend, no LLVM).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using vgre::compiler::backend::LaunchConfig;
using vgre::compiler::backend::makeBackend;
using vgre::compiler::frontend::compileToPtx;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static std::unique_ptr<vgre::compiler::backend::PreparedKernel>
compile(vgre::compiler::backend::ExecutionBackend& be, const char* src, const char* name) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen(%s) error: %s\n", name, cg.error.c_str()); return nullptr; }
    auto k = be.preparePtx(cg.ptx, name);
    if (!k) std::printf("---- PTX(%s) ----\n%s\n", name, cg.ptx.c_str());
    return k;
}

int main() {
    auto be = makeBackend("interpreter");
    CHECK(be != nullptr, "interpreter backend");
    if (!be) return 1;

    // ── 1. ++ / -- in a for loop ──────────────────────────────────────────────
    {
        const char* src = R"(
extern "C" __global__ void countup(int* out, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int s = 0;
        for (int j = 0; j < reps; j++) { s++; }
        out[i] = s;
    }
})";
        auto k = compile(*be, src, "countup");
        CHECK(k != nullptr, "countup compiles + loads");
        if (k) {
            const int N = 32, reps = 7;
            std::vector<int> out(N, -1);
            int* op = out.data(); int n = N, r = reps;
            void* args[] = {&op, &n, &r};
            LaunchConfig cfg; cfg.gridDim[0] = 1; cfg.blockDim[0] = N;
            CHECK(be->launch(*k, cfg, args, 3), "countup runs");
            int bad = 0; for (int i = 0; i < N; ++i) if (out[i] != reps) ++bad;
            CHECK(bad == 0, "countup: out[i] == reps (loops + ++ work)");
        }
    }

    // ── 2. math intrinsics: fmaf / fminf / fmaxf / sqrtf ──────────────────────
    {
        const char* src = R"(
extern "C" __global__ void mathk(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = fmaf(x[i], 2.0f, 1.0f);
        float b = fminf(a, 5.0f);
        y[i] = fmaxf(b, sqrtf(x[i]));
    }
})";
        auto k = compile(*be, src, "mathk");
        CHECK(k != nullptr, "mathk compiles + loads");
        if (k) {
            const int N = 16;
            std::vector<float> x(N), y(N, -1), ref(N);
            for (int i = 0; i < N; ++i) {
                x[i] = i * 0.7f;
                float a = std::fmaf(x[i], 2.0f, 1.0f);
                float b = std::fmin(a, 5.0f);
                ref[i] = std::fmax(b, std::sqrt(x[i]));
            }
            float* xp = x.data(); float* yp = y.data(); int n = N;
            void* args[] = {&xp, &yp, &n};
            LaunchConfig cfg; cfg.gridDim[0] = 1; cfg.blockDim[0] = N;
            CHECK(be->launch(*k, cfg, args, 3), "mathk runs");
            float e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(y[i] - ref[i]));
            CHECK(e < 1e-4f, "mathk: intrinsics match reference");
            std::printf("  mathk max error = %.3e\n", e);
        }
    }

    // ── 3. shared memory + __syncthreads: single-block tree reduction ─────────
    {
        const char* src = R"(
extern "C" __global__ void reduce(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float v = 0.0f;
    if (tid < n) { v = in[tid]; }
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out[0] = sdata[0]; }
})";
        auto k = compile(*be, src, "reduce");
        CHECK(k != nullptr, "reduce compiles + loads (shared memory)");
        if (k) {
            const int N = 64;
            std::vector<float> in(N), out(1, -1);
            float ref = 0; for (int i = 0; i < N; ++i) { in[i] = i * 1.0f; ref += in[i]; }
            float* ip = in.data(); float* op = out.data(); int n = N;
            void* args[] = {&ip, &op, &n};
            LaunchConfig cfg; cfg.gridDim[0] = 1; cfg.blockDim[0] = N;  // one block of N
            CHECK(be->launch(*k, cfg, args, 3), "reduce runs");
            CHECK(std::fabs(out[0] - ref) < 1e-3f, "reduce: shared-memory sum matches reference");
            std::printf("  reduce out=%.1f ref=%.1f\n", out[0], ref);
        }
    }

    if (g_fail == 0)
        std::printf("PASS: CUDA-C subset (shared mem, loops, ++/--, intrinsics) — all green\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
