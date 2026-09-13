// Track Z (Zero-Burden Engine): a real FLASH-ATTENTION kernel — the defining
// transformer workload — compiled by the from-scratch front-end (no LLVM) and
// run on the Tier-0 interpreter, verified against a CPU softmax-attention
// reference. This is the capstone that combines EVERYTHING the subset now has:
//   * __shared__ K/V tiles + __syncthreads() cooperative loading
//   * per-thread LOCAL register arrays (acc[D], qreg[D]) for the online softmax
//   * running max / running sum with rescale (the "flash" trick, no N^2 scores)
//   * expf / fmaxf intrinsics, nested loops, 2D flattened addressing
// One block per query tile; one thread per query; D = TILE = 8.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

// D = 8 (head dim), TILE = 8 (== blockDim.x). scale = 1/sqrt(8) = 0.35355339.
static const char* kFlash = R"(
extern "C" __global__ void flash_attn(float* O, const float* Q,
                                      const float* K, const float* V,
                                      int Nq, int Nk) {
    int q  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    __shared__ float Ks[64];
    __shared__ float Vs[64];
    float acc[8];
    float qreg[8];
    int d;
    for (d = 0; d < 8; d = d + 1) acc[d] = 0.0f;
    if (q < Nq) { for (d = 0; d < 8; d = d + 1) qreg[d] = Q[q*8 + d]; }
    else        { for (d = 0; d < 8; d = d + 1) qreg[d] = 0.0f; }
    float m = -1000000.0f;
    float l = 0.0f;
    float scale = 0.353553390f;
    int ntiles = (Nk + 7) / 8;
    int kt;
    for (kt = 0; kt < ntiles; kt = kt + 1) {
        int krow = kt * 8 + tx;
        int j;
        if (krow < Nk) {
            for (j = 0; j < 8; j = j + 1) {
                Ks[tx*8 + j] = K[krow*8 + j];
                Vs[tx*8 + j] = V[krow*8 + j];
            }
        } else {
            for (j = 0; j < 8; j = j + 1) { Ks[tx*8 + j] = 0.0f; Vs[tx*8 + j] = 0.0f; }
        }
        __syncthreads();
        int kk;
        for (kk = 0; kk < 8; kk = kk + 1) {
            int kidx = kt * 8 + kk;
            if (kidx < Nk) {
                float s = 0.0f;
                for (d = 0; d < 8; d = d + 1) s = s + qreg[d] * Ks[kk*8 + d];
                s = s * scale;
                float mnew = fmaxf(m, s);
                float corr = expf(m - mnew);
                float p = expf(s - mnew);
                l = l * corr + p;
                for (d = 0; d < 8; d = d + 1) acc[d] = acc[d] * corr + p * Vs[kk*8 + d];
                m = mnew;
            }
        }
        __syncthreads();
    }
    if (q < Nq) {
        float inv = 1.0f / l;
        for (d = 0; d < 8; d = d + 1) O[q*8 + d] = acc[d] * inv;
    }
}
)";

int main() {
    const int D = 8, Nq = 24, Nk = 20;   // non-multiples of 8 exercise the tail guards
    std::vector<float> Q(Nq * D), K(Nk * D), V(Nk * D), O(Nq * D, 0);
    for (int i = 0; i < Nq * D; ++i) Q[i] = std::sin(0.3f * i) * 0.5f;
    for (int i = 0; i < Nk * D; ++i) K[i] = std::cos(0.2f * i) * 0.5f;
    for (int i = 0; i < Nk * D; ++i) V[i] = std::sin(0.1f * i + 1.0f);

    auto cg = compileToPtx(kFlash, "flash_attn");
    CHECK(cg.ok, "flash-attention kernel compiles (no LLVM)");
    if (!cg.ok) { std::printf("  codegen: %s\n", cg.error.c_str()); return 1; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, "flash_attn");
    CHECK(k != nullptr, "flash-attention PTX loads");
    if (!k) { std::printf("%s\n", cg.ptx.c_str()); return 1; }

    void* Op = O.data(); void* Qp = Q.data(); void* Kp = K.data(); void* Vp = V.data();
    int nq = Nq, nk = Nk;
    void* args[] = {&Op, &Qp, &Kp, &Vp, &nq, &nk};
    be::LaunchConfig lc;
    lc.gridDim[0] = (Nq + 7) / 8;   // one block per 8 queries
    lc.blockDim[0] = 8;
    CHECK(beI->launch(*k, lc, args, 6), "flash-attention runs on the interpreter");

    // CPU reference: standard softmax attention, O[q] = sum_j softmax(QK^T*scale)_j V[j].
    const float scale = 0.353553390f;
    double maxerr = 0;
    for (int q = 0; q < Nq; ++q) {
        std::vector<float> sc(Nk);
        float mx = -1e30f;
        for (int j = 0; j < Nk; ++j) {
            float s = 0; for (int d = 0; d < D; ++d) s += Q[q*D+d] * K[j*D+d];
            sc[j] = s * scale; if (sc[j] > mx) mx = sc[j];
        }
        float sum = 0; for (int j = 0; j < Nk; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < D; ++d) {
            float o = 0; for (int j = 0; j < Nk; ++j) o += sc[j] / sum * V[j*D+d];
            maxerr = std::max(maxerr, std::fabs((double)o - O[q*D+d]));
        }
    }
    CHECK(maxerr < 2e-3, "flash-attention output matches CPU softmax attention");
    std::printf("  flash-attention max error vs reference = %.3e\n", maxerr);

    if (g_fail == 0)
        std::printf("PASS: flash-attention (shared tiles + local acc + online softmax, no LLVM)\n");
    return g_fail ? 1 : 0;
}
