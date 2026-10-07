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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_WIN32)
static inline void* aligned_alloc_portable(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static inline void aligned_free_portable(void* p) { _aligned_free(p); }
#else
static inline void* aligned_alloc_portable(size_t alignment, size_t size) {
    return std::aligned_alloc(alignment, size);
}
static inline void aligned_free_portable(void* p) { std::free(p); }
#endif

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

static inline uint16_t f2bf(float f) { return vgre::runtime::fp32_to_bf16(f); }
static inline float    bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static inline uint16_t f2half(float f) { return vgre_cuda::__half(f).__x; }
static inline float    half2f(uint16_t b) { return vgre_mma_detail::f16b(b); }

using WgmmaFn = void (*)(float*, int, uint64_t, uint64_t);
struct Job { int N; uint64_t descA, descB; float cInit; float* frags; WgmmaFn mma; };

static void lane_job(int tid, void* arg) {
    auto* j = static_cast<Job*>(arg);
    *vgre_jit_get_threadIdx() = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()  = vgre::dim3(128, 1, 1);
    const int nFrag = j->N / 2;
    float frag[128];
    for (int e = 0; e < nFrag; ++e) frag[e] = j->cInit;        // uniform C (RMW test)
    j->mma(frag, j->N, j->descA, j->descB);
    for (int e = 0; e < nFrag; ++e) j->frags[(size_t)tid * nFrag + e] = frag[e];
}

using Encode16Fn = uint16_t (*)(float);
using Decode16Fn = float (*)(uint16_t);

static bool run_16bit_case(const char* type, int N, float cInit,
                           Encode16Fn encode, Decode16Fn decode, WgmmaFn mma) {
    const int M = 64, K = 16;
    // 16-byte-aligned tiles so (ptr>>4)<<4 recovers the descriptor pointer.
    auto* A = static_cast<uint16_t*>(aligned_alloc_portable(16, ((size_t)M * K * 2 + 15) & ~15ull));
    auto* B = static_cast<uint16_t*>(aligned_alloc_portable(16, ((size_t)K * N * 2 + 15) & ~15ull));
    std::mt19937 rng(N * 131 + 7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Af(M * K), Bf(K * N);
    for (int i = 0; i < M * K; ++i) { A[i] = encode(nd(rng)); Af[i] = decode(A[i]); }
    for (int i = 0; i < K * N; ++i) { B[i] = encode(nd(rng)); Bf[i] = decode(B[i]); }

    Job job{N, (uint64_t)(uintptr_t)A >> 4, (uint64_t)(uintptr_t)B >> 4,
            cInit, nullptr, mma};
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
    aligned_free_portable(A); aligned_free_portable(B);
    printf("  [info] %s N=%d cInit=%.1f: bijection=%d max abs err=%.2e\n",
           type, N, cInit, bijection, maxErr);
    return bijection && maxErr < 1e-2;
}

static bool run_tf32_case() {
    constexpr int M = 64, N = 256, K = 8;
    auto* A = static_cast<float*>(aligned_alloc_portable(16, M * K * sizeof(float)));
    auto* B = static_cast<float*>(aligned_alloc_portable(16, K * N * sizeof(float)));
    if (!A || !B) { aligned_free_portable(A); aligned_free_portable(B); return false; }
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A[m * K + k] = static_cast<float>(((m * 17 + k * 11) % 129) - 64) / 8.0f;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B[k * N + n] = static_cast<float>(((k * 23 + n * 7) % 97) - 48) / 8.0f;

    Job job{N, (uint64_t)(uintptr_t)A >> 4, (uint64_t)(uintptr_t)B >> 4,
            0.75f, nullptr, vgre_wgmma_wg_tf32};
    std::vector<float> frags((size_t)128 * (N / 2), 0.0f);
    job.frags = frags.data();
    job.cInit = 0.75f;
    vgre_jit_block_dispatch(128, lane_job, &job);

    std::vector<float> D((size_t)M * N, 0.0f);
    std::vector<int> wrote((size_t)M * N, 0);
    for (int lane = 0; lane < 128; ++lane)
        for (int e = 0; e < N / 2; ++e) {
            int row, col; detail::wgmma_frag_rc(lane, e, &row, &col);
            D[(size_t)row * N + col] = frags[(size_t)lane * (N / 2) + e];
            ++wrote[(size_t)row * N + col];
        }
    bool bijection = true;
    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i) bijection = bijection && wrote[i] == 1;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = job.cInit;
            for (int k = 0; k < K; ++k) ref += A[m * K + k] * B[k * N + n];
            maxErr = std::max(maxErr, (double)std::fabs(D[(size_t)m * N + n] - ref));
        }
    aligned_free_portable(A); aligned_free_portable(B);
    printf("  [info] tf32 N=%d: bijection=%d max abs err=%.2e\n", N, bijection, maxErr);
    return bijection && maxErr < 1e-3;
}

static bool test_f16_n128_scalar_bounds() {
    constexpr int M = 64, N = 128, K = 16;
    auto* A = static_cast<uint16_t*>(aligned_alloc_portable(16, M * K * sizeof(uint16_t)));
    auto* B = static_cast<uint16_t*>(aligned_alloc_portable(16, K * N * sizeof(uint16_t)));
    if (!A || !B) { aligned_free_portable(A); aligned_free_portable(B); return false; }
    for (int i = 0; i < M * K; ++i) A[i] = f2half(static_cast<float>((i % 13) - 6) / 8.0f);
    for (int i = 0; i < K * N; ++i) B[i] = f2half(static_cast<float>((i % 11) - 5) / 8.0f);

    constexpr float kGuard = 1234.5f;
    std::vector<float> D(2 * M * N, kGuard);
    std::fill(D.begin(), D.begin() + M * N, 0.0f);
    vgre_wgmma_m64n128k16_f16_f32(D.data(),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(A)) >> 4,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(B)) >> 4);

    double maxErr = 0.0;
    bool guardIntact = true;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k)
                ref += half2f(A[m * K + k]) * half2f(B[k * N + n]);
            maxErr = std::max(maxErr, static_cast<double>(std::fabs(D[m * N + n] - ref)));
        }
    for (size_t i = M * N; i < D.size(); ++i) guardIntact = guardIntact && D[i] == kGuard;
    aligned_free_portable(A); aligned_free_portable(B);
    std::printf("  [info] scalar f16 m64n128k16 max abs err=%.2e guard=%d\n", maxErr, guardIntact);
    return maxErr < 1e-4 && guardIntact;
}

int main() {
    printf("=== Hopper wgmma warp-group collective MMA (Track P3-4) ===\n");
    check("m64n64k16 bf16: distributed accumulator == A·B (C=0)",
          run_16bit_case("bf16", 64, 0.0f, f2bf, bf2f, vgre_wgmma_wg_bf16));
    check("m64n128k16 bf16: distributed accumulator == A·B (C=0)",
          run_16bit_case("bf16", 128, 0.0f, f2bf, bf2f, vgre_wgmma_wg_bf16));
    check("m64n256k16 bf16: accumulator read-modify-write (C=5)",
          run_16bit_case("bf16", 256, 5.0f, f2bf, bf2f, vgre_wgmma_wg_bf16));
    check("m64n128k16 f16: distributed accumulator == A·B (C=0)",
          run_16bit_case("f16", 128, 0.0f, f2half, half2f, vgre_wgmma_wg_f16));
    check("m64n256k16 f16: accumulator read-modify-write (C=5)",
          run_16bit_case("f16", 256, 5.0f, f2half, half2f, vgre_wgmma_wg_f16));
    check("m64n256k8 tf32: distributed accumulator == A·B+C", run_tf32_case());
    check("scalar f16 m64n128k16 preserves rows outside the 64-row tile",
          test_f16_n128_scalar_bounds());
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
