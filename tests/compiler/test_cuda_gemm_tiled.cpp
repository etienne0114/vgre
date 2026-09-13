// Track Z (Zero-Burden Engine): a TILED matrix multiply — the canonical
// shared-memory + __syncthreads cooperative kernel — compiled by the from-scratch
// front-end (no LLVM) and run on the Tier-0 interpreter (which schedules the
// CTA's threads cooperatively across the barrier). Verified vs a CPU reference.
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

// 16x16 tiled GEMM. Each block cooperatively stages a 16-wide K-tile of A and B
// into shared memory, syncs, and accumulates. Literal 16 (no macros — the lexer
// skips preprocessor lines).
static const char* kTiled = R"(
extern "C" __global__ void gemm_tiled(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    int ntiles = (K + 15) / 16;
    for (int t = 0; t < ntiles; t = t + 1) {
        int aCol = t * 16 + tx;
        int bRow = t * 16 + ty;
        if (row < M && aCol < K) As[ty*16 + tx] = A[row*K + aCol];
        else As[ty*16 + tx] = 0.0f;
        if (bRow < K && col < N) Bs[ty*16 + tx] = B[bRow*N + col];
        else Bs[ty*16 + tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; k = k + 1) {
            sum = sum + As[ty*16 + k] * Bs[k*16 + tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = sum;
})";

int main() {
    const int M = 40, N = 32, K = 48;   // sizes not multiples of 16 -> exercises the tile guards
    std::vector<float> A(M * K), B(K * N), ref(M * N, 0.0f), C(M * N, -1.0f);
    for (int i = 0; i < M * K; ++i) A[i] = std::sin(0.05f * i) + 0.25f;
    for (int i = 0; i < K * N; ++i) B[i] = std::cos(0.03f * i) + 0.1f;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
            ref[r * N + c] = s;
        }

    auto cg = compileToPtx(kTiled, "gemm_tiled");
    CHECK(cg.ok, "tiled GEMM compiles to PTX (no LLVM)");
    if (!cg.ok) { std::printf("  codegen: %s\n", cg.error.c_str()); return 1; }

    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, "gemm_tiled");
    CHECK(k != nullptr, "tiled GEMM PTX loads (shared mem + bar.sync)");
    if (!k) { std::printf("---- PTX ----\n%s\n", cg.ptx.c_str()); return 1; }

    const float* Ap = A.data(); const float* Bp = B.data(); float* Cp = C.data();
    int m = M, n = N, kk = K;
    void* args[] = {&Ap, &Bp, &Cp, &m, &n, &kk};
    be::LaunchConfig lc;
    lc.gridDim[0] = (N + 15) / 16; lc.gridDim[1] = (M + 15) / 16;
    lc.blockDim[0] = 16; lc.blockDim[1] = 16;
    CHECK(beI->launch(*k, lc, args, 6), "tiled GEMM runs (cooperative CTA)");

    float e = 0;
    for (int i = 0; i < M * N; ++i) e = std::fmax(e, std::fabs(C[i] - ref[i]));
    CHECK(e < 1e-3f, "tiled GEMM matches CPU reference");
    std::printf("  tiled GEMM %dx%dx%d  max_err=%.3e\n", M, N, K, e);

    if (g_fail == 0)
        std::printf("PASS: tiled shared-memory GEMM (no LLVM) — cooperative __syncthreads works\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
