// Track Z (Zero-Burden Engine): a real matrix-multiply kernel — the canonical GPU
// workload — compiled by the from-scratch front-end (no LLVM) and run on BOTH
// execution tiers, verified against a CPU reference. This exercises 2D grid/block
// indexing, an inner for-loop accumulator, flattened 2D addressing, and a
// compound && guard — a genuine kernel, not a toy.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// C[MxN] = A[MxK] * B[KxN], one thread per output element (row-major, flattened).
static const char* kGemm = R"(
extern "C" __global__ void gemm(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k = k + 1) {
            sum = sum + A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
})";

static void run(bool compiled, const char* label,
                const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M, int N, int K) {
    const float* Ap = A.data(); const float* Bp = B.data(); float* Cp = C.data();
    int m = M, n = N, k = K;
    void* args[] = {&Ap, &Bp, &Cp, &m, &n, &k};
    // 16x16 blocks tiling the NxM output grid (x=col, y=row).
    uint32_t bx = 16, by = 16;
    uint32_t gx = (N + bx - 1) / bx, gy = (M + by - 1) / by;

    if (compiled) {
        std::string err;
        auto ck = CompiledKernel::compileSource(kGemm, "gemm", err);
        CHECK(ck != nullptr, "gemm compiles on the compiled tier");
        if (!ck) { std::printf("  compile: %s\n", err.c_str()); return; }
        Extent g{gx, gy, 1}, b{bx, by, 1};
        CHECK(ck->launch(g, b, args, 6), "gemm runs on the compiled tier");
    } else {
        auto cg = compileToPtx(kGemm, "gemm");
        CHECK(cg.ok, "gemm compiles to PTX (no LLVM)");
        if (!cg.ok) { std::printf("  codegen: %s\n", cg.error.c_str()); return; }
        auto beI = be::makeBackend("interpreter");
        auto kk = beI->preparePtx(cg.ptx, "gemm");
        CHECK(kk != nullptr, "gemm PTX loads into the interpreter");
        if (!kk) return;
        be::LaunchConfig lc;
        lc.gridDim[0] = gx; lc.gridDim[1] = gy;
        lc.blockDim[0] = bx; lc.blockDim[1] = by;
        CHECK(beI->launch(*kk, lc, args, 6), "gemm runs on the interpreter tier");
    }
    (void)label;
}

int main() {
    const int M = 20, N = 24, K = 16;
    std::vector<float> A(M * K), B(K * N), ref(M * N, 0.0f);
    for (int i = 0; i < M * K; ++i) A[i] = std::sin(0.1f * i) + 0.5f;
    for (int i = 0; i < K * N; ++i) B[i] = std::cos(0.07f * i) - 0.3f;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
            ref[r * N + c] = s;
        }

    std::vector<float> Ci(M * N, -1.0f), Cc(M * N, -1.0f);
    run(false, "interpreter", A, B, Ci, M, N, K);
    run(true, "compiled", A, B, Cc, M, N, K);

    float ei = 0, ec = 0;
    for (int i = 0; i < M * N; ++i) {
        ei = std::fmax(ei, std::fabs(Ci[i] - ref[i]));
        ec = std::fmax(ec, std::fabs(Cc[i] - ref[i]));
    }
    CHECK(ei < 1e-3f, "interpreter GEMM matches CPU reference");
    CHECK(ec < 1e-4f, "compiled GEMM matches CPU reference");
    std::printf("  GEMM %dx%dx%d  interp_err=%.3e  compiled_err=%.3e\n", M, N, K, ei, ec);

    if (g_fail == 0)
        std::printf("PASS: real GEMM kernel on BOTH tiers (CUDA-C -> our compiler -> run, no LLVM)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
