// Phase 3, Track P3-4 — Hopper warp-group MMA (wgmma) correctness.
//
// wgmma is WARP-GROUP-collective: the m64×N accumulator is distributed across all
// 128 lanes (each holds N/2 fp32) and A/B live in shared memory. This test drives
// 128 lanes through the real collective helper (each computing only its fragment
// from the shared tiles), reassembles the full D via the PTX-ISA fragment layout,
// and checks: (1) the layout is a BIJECTION (every output element written exactly
// once — proves the fragment map), (2) D == an independent A·B GEMM, and (3) the
// accumulator is read-modify-write (D = A·B + C).

#include "vgre/compiler/wmma_emulation.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
  void vgre_jit_block_dispatch(int, void (*)(int, void*), void*);
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockDim();
}

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static inline uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static inline float    bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

struct Job { int N; uint64_t descA, descB; float cInit; float* frags; };  // frags: [128][N/2]

static void lane_job(int tid, void* arg) {
    auto* j = static_cast<Job*>(arg);
    *vgre_jit_get_threadIdx() = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()  = vgre::dim3(128, 1, 1);
    const int nFrag = j->N / 2;
    float frag[128];
    for (int e = 0; e < nFrag; ++e) frag[e] = j->cInit;        // uniform C (RMW test)
    vgre_wgmma_wg_bf16(frag, j->N, j->descA, j->descB);
    for (int e = 0; e < nFrag; ++e) j->frags[(size_t)tid * nFrag + e] = frag[e];
}

static bool run_case(int N, float cInit) {
    const int M = 64, K = 16;
    // 16-byte-aligned bf16 tiles so (ptr>>4)<<4 recovers the pointer.
    auto* A = static_cast<uint16_t*>(std::aligned_alloc(16, ((size_t)M * K * 2 + 15) & ~15ull));
    auto* B = static_cast<uint16_t*>(std::aligned_alloc(16, ((size_t)K * N * 2 + 15) & ~15ull));
    std::mt19937 rng(N * 131 + 7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Af(M * K), Bf(K * N);
    for (int i = 0; i < M * K; ++i) { Af[i] = bf2f(f2bf(nd(rng))); A[i] = f2bf(Af[i]); }
    for (int i = 0; i < K * N; ++i) { Bf[i] = bf2f(f2bf(nd(rng))); B[i] = f2bf(Bf[i]); }

    Job job;
    job.N = N;
    job.descA = (uint64_t)(uintptr_t)A >> 4;
    job.descB = (uint64_t)(uintptr_t)B >> 4;
    job.cInit = cInit;
    std::vector<float> frags((size_t)128 * (N / 2), 0.0f);
    job.frags = frags.data();
    vgre_jit_block_dispatch(128, lane_job, &job);

    // Reassemble D[64][N] from the lane fragments via the PTX-ISA layout, counting
    // writes per element (must be exactly 1 → bijective layout).
    std::vector<float> D((size_t)M * N, 0.0f);
    std::vector<int>   wrote((size_t)M * N, 0);
    const int nFrag = N / 2;
    for (int wgLane = 0; wgLane < 128; ++wgLane)
        for (int e = 0; e < nFrag; ++e) {
            int row, col; detail::wgmma_frag_rc(wgLane, e, &row, &col);
            D[(size_t)row * N + col] = frags[(size_t)wgLane * nFrag + e];
            wrote[(size_t)row * N + col]++;
        }
    bool bijection = true;
    for (int i = 0; i < M * N; ++i) if (wrote[i] != 1) bijection = false;

    // Independent reference: D = A·B + cInit.
    double maxErr = 0.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = cInit;
            for (int k = 0; k < K; ++k) acc += Af[m * K + k] * Bf[k * N + n];
            maxErr = std::max(maxErr, (double)std::fabs(D[(size_t)m * N + n] - acc));
        }
    std::free(A); std::free(B);
    printf("  [info] N=%d cInit=%.1f: bijection=%d max abs err=%.2e\n", N, cInit, bijection, maxErr);
    return bijection && maxErr < 1e-2;
}

int main() {
    printf("=== Hopper wgmma warp-group collective MMA (Track P3-4) ===\n");
    check("m64n64k16 bf16: distributed accumulator == A·B (C=0)",  run_case(64, 0.0f));
    check("m64n128k16 bf16: distributed accumulator == A·B (C=0)", run_case(128, 0.0f));
    check("m64n256k16 bf16: accumulator read-modify-write (C=5)",  run_case(256, 5.0f));
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
