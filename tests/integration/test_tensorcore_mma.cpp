// Track 9 — warp-collective tensor-core mma.sync correctness.
//
// The raw-PTX mma.sync.* helpers (vgre_mma_*) are WARP-COLLECTIVE: a single lane
// holds only a *fragment* of A/B/C (e.g. m16n8k16.f16: 8 of A's 256 elements), so
// its 4 outputs cannot be computed in isolation. VGRE runs them with one OS
// thread per lane sharing a per-warp scratch buffer + a block barrier — each lane
// deposits its fragment, barriers, the full tile GEMM is reconstructed, and this
// lane's outputs are scattered back.
//
// This test drives the SAME runtime path the JIT codegen sets up for `usesMma`
// kernels: it dispatches 32 lanes via vgre_jit_block_dispatch (which installs the
// block barrier exactly as a real kernel launch does), points each lane's TLS at
// a shared fragment buffer, and calls the helper. The host packs A/B/C into
// fragments using the canonical PTX ISA layout (ISA 9.7.14.4.x) and checks the
// reassembled D == A·B + C from an INDEPENDENT reference. A wrong fragment→lane
// map yields a wrong D — so this proves the layout, not just "it ran".

#include "vgre/compiler/wmma_emulation.h"   // vgre_mma_* (warp-collective helpers)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
  void vgre_jit_block_dispatch(int, void (*)(int, void*), void*);
  void** vgre_jit_get_mma_buffer();
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockDim();
}

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static inline uint16_t f16bits(float f) { return vgre_cuda::__half(f).__x; }
static inline float    f16val (float f) { return float(vgre_cuda::__half(f)); }
static inline uint32_t pack2(uint16_t lo, uint16_t hi) {
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
static inline uint32_t packs8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// Shared per-warp fragment scratch the codegen would allocate for a usesMma
// kernel; all 32 lanes point their TLS at it.
static uint32_t g_mma_scratch[32 * 16];

// ── m16n8k16 f16·f16→f32 ─────────────────────────────────────────────────────
struct F16Job {
    const uint32_t *Areg, *Breg; const float *Creg; float *Dreg;  // [32×..]
};
static void f16_lane(int tid, void *arg) {
    auto *j = static_cast<F16Job *>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx()  = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()   = vgre::dim3(32, 1, 1);
    const uint32_t *A = j->Areg + tid * 4;
    const uint32_t *B = j->Breg + tid * 2;
    const float    *C = j->Creg + tid * 4;
    float d0, d1, d2, d3;
    vgre_mma_m16n8k16_f32_f16(d0, d1, d2, d3, A[0], A[1], A[2], A[3], B[0], B[1],
                              C[0], C[1], C[2], C[3]);
    float *D = j->Dreg + tid * 4;
    D[0] = d0; D[1] = d1; D[2] = d2; D[3] = d3;
}

static bool runF16() {
    const int M = 16, N = 8, K = 16;
    std::mt19937 rng(20260611);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), C(M * N);
    for (auto &x : A) x = f16val(nd(rng));
    for (auto &x : B) x = f16val(nd(rng));
    for (auto &x : C) x = nd(rng);

    std::vector<uint32_t> Areg(32 * 4), Breg(32 * 2);
    std::vector<float>    Creg(32 * 4), Dreg(32 * 4, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        auto Ab = [&](int m, int k){ return f16bits(A[m * K + k]); };
        Areg[lane*4+0] = pack2(Ab(g,   2*t+0), Ab(g,   2*t+1));
        Areg[lane*4+1] = pack2(Ab(g+8, 2*t+0), Ab(g+8, 2*t+1));
        Areg[lane*4+2] = pack2(Ab(g,   2*t+8), Ab(g,   2*t+9));
        Areg[lane*4+3] = pack2(Ab(g+8, 2*t+8), Ab(g+8, 2*t+9));
        auto Bb = [&](int k, int n){ return f16bits(B[k * N + n]); };
        Breg[lane*2+0] = pack2(Bb(2*t+0, g), Bb(2*t+1, g));
        Breg[lane*2+1] = pack2(Bb(2*t+8, g), Bb(2*t+9, g));
        Creg[lane*4+0] = C[(g)   * N + 2*t+0]; Creg[lane*4+1] = C[(g)   * N + 2*t+1];
        Creg[lane*4+2] = C[(g+8) * N + 2*t+0]; Creg[lane*4+3] = C[(g+8) * N + 2*t+1];
    }

    F16Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(32, f16_lane, &job);

    std::vector<float> D(M * N, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        D[(g)   * N + 2*t+0] = Dreg[lane*4+0]; D[(g)   * N + 2*t+1] = Dreg[lane*4+1];
        D[(g+8) * N + 2*t+0] = Dreg[lane*4+2]; D[(g+8) * N + 2*t+1] = Dreg[lane*4+3];
    }
    double maxErr = 0.0; int at = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = C[m * N + n];
            for (int kk = 0; kk < K; ++kk) ref += f16val(A[m*K+kk]) * f16val(B[kk*N+n]);
            double e = std::fabs(D[m * N + n] - ref);
            if (e > maxErr) { maxErr = e; at = m * N + n; }
        }
    printf("  [info] m16n8k16.f16 max abs err vs A*B+C = %.2e at (%d,%d): D=%.4f\n",
           maxErr, at / N, at % N, D[at]);
    return maxErr < 1e-2;
}

// ── m16n8k32 s8·s8→s32 ───────────────────────────────────────────────────────
struct S8Job {
    const uint32_t *Areg, *Breg; const int *Creg; int *Dreg;
};
static void s8_lane(int tid, void *arg) {
    auto *j = static_cast<S8Job *>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx()  = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()   = vgre::dim3(32, 1, 1);
    const uint32_t *A = j->Areg + tid * 4;
    const uint32_t *B = j->Breg + tid * 2;
    const int      *C = j->Creg + tid * 4;
    int d0, d1, d2, d3;
    vgre_mma_m16n8k32_s8(d0, d1, d2, d3, A[0], A[1], A[2], A[3], B[0], B[1],
                         C[0], C[1], C[2], C[3]);
    int *D = j->Dreg + tid * 4;
    D[0] = d0; D[1] = d1; D[2] = d2; D[3] = d3;
}

static bool runS8() {
    const int M = 16, N = 8, K = 32;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> di(-8, 7);
    std::vector<int8_t> A(M * K), B(K * N);
    std::vector<int>    C(M * N);
    for (auto &x : A) x = (int8_t)di(rng);
    for (auto &x : B) x = (int8_t)di(rng);
    for (auto &x : C) x = di(rng);

    std::vector<uint32_t> Areg(32 * 4), Breg(32 * 2);
    std::vector<int>      Creg(32 * 4), Dreg(32 * 4, 0);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        auto Ae = [&](int m, int kc){ return A[m * K + kc]; };
        Areg[lane*4+0] = packs8(Ae(g,   4*t+0), Ae(g,   4*t+1), Ae(g,   4*t+2), Ae(g,   4*t+3));
        Areg[lane*4+1] = packs8(Ae(g+8, 4*t+0), Ae(g+8, 4*t+1), Ae(g+8, 4*t+2), Ae(g+8, 4*t+3));
        Areg[lane*4+2] = packs8(Ae(g,   4*t+16),Ae(g,   4*t+17),Ae(g,   4*t+18),Ae(g,   4*t+19));
        Areg[lane*4+3] = packs8(Ae(g+8, 4*t+16),Ae(g+8, 4*t+17),Ae(g+8, 4*t+18),Ae(g+8, 4*t+19));
        auto Be = [&](int kc, int n){ return B[kc * N + n]; };
        Breg[lane*2+0] = packs8(Be(4*t+0, g),  Be(4*t+1, g),  Be(4*t+2, g),  Be(4*t+3, g));
        Breg[lane*2+1] = packs8(Be(4*t+16, g), Be(4*t+17, g), Be(4*t+18, g), Be(4*t+19, g));
        Creg[lane*4+0] = C[(g)   * N + 2*t+0]; Creg[lane*4+1] = C[(g)   * N + 2*t+1];
        Creg[lane*4+2] = C[(g+8) * N + 2*t+0]; Creg[lane*4+3] = C[(g+8) * N + 2*t+1];
    }

    S8Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(32, s8_lane, &job);

    std::vector<int> D(M * N, 0);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        D[(g)   * N + 2*t+0] = Dreg[lane*4+0]; D[(g)   * N + 2*t+1] = Dreg[lane*4+1];
        D[(g+8) * N + 2*t+0] = Dreg[lane*4+2]; D[(g+8) * N + 2*t+1] = Dreg[lane*4+3];
    }
    long maxErr = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            long ref = C[m * N + n];
            for (int kk = 0; kk < K; ++kk) ref += (int)A[m*K+kk] * (int)B[kk*N+n];
            maxErr = std::max(maxErr, std::labs(ref - (long)D[m * N + n]));
        }
    printf("  [info] m16n8k32.s8 max abs err vs A*B+C = %ld (exact)\n", maxErr);
    return maxErr == 0;
}

// ── TMA store (shared → global) — the JIT-path helpers in wmma_emulation.h ────
// cp.async.bulk.tensor.2d.global.shared::cta.bulk_group lowers to
// vgre_tma_store_2d_b. Validates the interior box scatter, out-of-bounds clipping
// (real TMA store drops OOB elements — must not overrun the tensor), and that the
// store is the exact inverse of the load (load→store reproduces the tensor).
static void runTmaStore() {
    const int H = 8, W = 8, bh = 4, bw = 4, GUARD = 4;
    std::vector<float> G(H * W + GUARD, -1.0f);
    for (int i = 0; i < GUARD; ++i) G[H * W + i] = 7777.0f;   // guard cells past the tensor

    VgreTMADescriptor d{};
    d.baseAddr  = G.data();
    d.elemBytes = sizeof(float);
    d.dim[0] = W; d.dim[1] = H;                                // global bounds
    d.stride[0] = (uint32_t)(W * sizeof(float));               // row byte stride (2d convention)
    d.boxDim[0] = bw; d.boxDim[1] = bh;

    // 1) Interior store lands the box at (col=2,row=1); other cells untouched.
    float tile[bh * bw];
    for (int i = 0; i < bh * bw; ++i) tile[i] = 100.0f + i;
    vgre_tma_store_2d_b(&d, tile, /*col*/2, /*row*/1);
    bool ok1 = (G[0] == -1.0f);
    for (int i = 0; i < bh; ++i)
        for (int j = 0; j < bw; ++j)
            if (G[(1 + i) * W + (2 + j)] != tile[i * bw + j]) ok1 = false;
    check("TMA 2d store writes the interior box to global", ok1);

    // 2) Box overhanging the tensor at (col=6,row=6): only [6..7]x[6..7] valid; the
    //    OOB quadrant must be dropped and the guard cells must be intact.
    std::fill(G.begin(), G.begin() + H * W, -1.0f);
    for (int i = 0; i < bh * bw; ++i) tile[i] = 500.0f + i;
    vgre_tma_store_2d_b(&d, tile, 6, 6);
    bool ok2 = true;
    for (int i = 0; i < bh; ++i)
        for (int j = 0; j < bw; ++j) {
            int gr = 6 + i, gc = 6 + j;
            if (gr < H && gc < W && G[gr * W + gc] != tile[i * bw + j]) ok2 = false;
        }
    for (int i = 0; i < GUARD; ++i) if (G[H * W + i] != 7777.0f) ok2 = false;   // no overrun
    check("TMA 2d store clips out-of-bounds (no overrun, real TMA padding)", ok2);

    // 3) load→store round-trip reproduces the whole tensor exactly.
    std::vector<float> src(H * W), dstG(H * W, 0.0f);
    for (int i = 0; i < H * W; ++i) src[i] = (float)i * 0.5f - 3.0f;
    VgreTMADescriptor ds = d; ds.baseAddr = src.data();
    VgreTMADescriptor dd = d; dd.baseAddr = dstG.data();
    float box[bh * bw];
    for (int y = 0; y < H; y += bh)
        for (int x = 0; x < W; x += bw) {
            vgre_tma_load_2d_b(box, &ds, x, y);
            vgre_tma_store_2d_b(&dd, box, x, y);
        }
    bool ok3 = true;
    for (int i = 0; i < H * W; ++i) if (dstG[i] != src[i]) ok3 = false;
    check("TMA load->store round-trip reproduces the tensor exactly", ok3);
}

int main() {
    printf("=== Tensor-core mma.sync warp-collective correctness (Track 9) ===\n");
    check("m16n8k16 f16*f16->f32 matches A*B+C reference", runF16());
    check("m16n8k32 s8*s8->s32 matches A*B+C reference (exact)", runS8());
    runTmaStore();
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
