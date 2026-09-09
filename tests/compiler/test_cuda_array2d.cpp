// Track Z (Zero-Burden Engine): multi-dimensional array syntax. Stock CUDA
// kernels declare tiles as `__shared__ float As[16][16]` and index them
// `As[ty][tx]` — the from-scratch front-end now flattens N-D declarations and
// N-D indexing to row-major offsets, so textbook kernels compile verbatim (no
// hand-flattening). Verified on:
//   * a tiled GEMM with 2D __shared__ tiles (interpreter tier)
//   * a per-thread 2D local array (BOTH tiers)
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
#include <cstdio>
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

// Textbook tiled GEMM — 2D __shared__ tiles, natural As[ty][tx] indexing.
static const char* kGemm2D = R"(
extern "C" __global__ void gemm2d(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    int ntiles = (K + 15) / 16;
    for (int t = 0; t < ntiles; t = t + 1) {
        int aCol = t * 16 + tx;
        int bRow = t * 16 + ty;
        if (row < M && aCol < K) As[ty][tx] = A[row*K + aCol];
        else As[ty][tx] = 0.0f;
        if (bRow < K && col < N) Bs[ty][tx] = B[bRow*N + col];
        else Bs[ty][tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; k = k + 1) {
            sum = sum + As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = sum;
})";

// Per-thread 2D local array m[2][3] — runs on the compiled tier (no barrier).
// out[i] = sum_{r<2,c<3} (in[i]*(r+1) + c) = 9*in[i] + 6.
static const char* kLocal2D = R"(
extern "C" __global__ void local2d(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float m[2][3];
        int r;
        int c;
        for (r = 0; r < 2; r = r + 1)
            for (c = 0; c < 3; c = c + 1)
                m[r][c] = in[i] * (r + 1) + c;
        float s = 0.0f;
        for (r = 0; r < 2; r = r + 1)
            for (c = 0; c < 3; c = c + 1)
                s = s + m[r][c];
        out[i] = s;
    }
})";

int main() {
    // ── tiled GEMM with 2D shared tiles (interpreter) ────────────────────────
    {
        const int M = 32, N = 32, K = 32;
        std::vector<float> A(M * K), B(K * N), C(M * N, 0);
        for (int i = 0; i < M * K; ++i) A[i] = std::sin(0.1f * i);
        for (int i = 0; i < K * N; ++i) B[i] = std::cos(0.07f * i);
        auto cg = compileToPtx(kGemm2D, "gemm2d");
        CHECK(cg.ok, "2D __shared__ tiled GEMM compiles");
        if (!cg.ok) { std::printf("  codegen: %s\n", cg.error.c_str()); return 1; }
        auto beI = be::makeBackend("interpreter");
        auto k = beI->preparePtx(cg.ptx, "gemm2d");
        CHECK(k != nullptr, "gemm2d PTX loads");
        if (k) {
            void* Ap = A.data(); void* Bp = B.data(); void* Cp = C.data();
            int m = M, n = N, kk = K;
            void* args[] = {&Ap, &Bp, &Cp, &m, &n, &kk};
            be::LaunchConfig lc;
            lc.gridDim[0] = N / 16; lc.gridDim[1] = M / 16;
            lc.blockDim[0] = 16; lc.blockDim[1] = 16;
            CHECK(beI->launch(*k, lc, args, 6), "gemm2d runs");
            double maxerr = 0;
            for (int r = 0; r < M; ++r) for (int cc = 0; cc < N; ++cc) {
                double ref = 0; for (int p = 0; p < K; ++p) ref += (double)A[r*K+p] * B[p*N+cc];
                maxerr = std::max(maxerr, std::fabs(ref - C[r*N+cc]));
            }
            CHECK(maxerr < 1e-3, "gemm2d matches CPU reference");
            std::printf("  gemm2d max error = %.3e\n", maxerr);
        }
    }

    // ── per-thread 2D local array on BOTH tiers ──────────────────────────────
    {
        const int Nn = 256;
        std::vector<float> in(Nn), out(Nn, 0);
        for (int i = 0; i < Nn; ++i) in[i] = (i % 9) * 0.5f - 2.0f;
        void* op = out.data(); void* ip = in.data(); int n = Nn;
        void* args[] = {&op, &ip, &n};
        auto refOk = [&]() {
            for (int i = 0; i < Nn; ++i)
                if (std::fabs(out[i] - (9.0f * in[i] + 6.0f)) > 1e-4f) return false;
            return true;
        };

        auto cg = compileToPtx(kLocal2D, "local2d");
        CHECK(cg.ok, "2D local array compiles");
        auto beI = be::makeBackend("interpreter");
        auto k = cg.ok ? beI->preparePtx(cg.ptx, "local2d") : nullptr;
        if (k) {
            be::LaunchConfig lc; lc.gridDim[0] = (Nn + 63) / 64; lc.blockDim[0] = 64;
            CHECK(beI->launch(*k, lc, args, 3), "local2d runs (interpreter)");
            CHECK(refOk(), "local2d == 9*in + 6 (interpreter)");
        }

        std::fill(out.begin(), out.end(), 0.0f);
        std::string err;
        auto ck = CompiledKernel::compileSource(kLocal2D, "local2d", err);
        CHECK(ck != nullptr, "2D local array compiles on the compiled tier");
        if (ck) {
            Extent g{(uint32_t)((Nn + 63) / 64), 1, 1}, b{64, 1, 1};
            CHECK(ck->launch(g, b, args, 3), "local2d runs (compiled tier)");
            CHECK(refOk(), "local2d == 9*in + 6 (compiled tier)");
        }
    }

    if (g_fail == 0)
        std::printf("PASS: multi-dimensional arrays (2D __shared__ tiles + 2D local, both tiers)\n");
    return g_fail ? 1 : 0;
}
