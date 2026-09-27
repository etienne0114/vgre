// nvcuda::wmma emulation (wmma_emulation.h) — verifies the tile MMA
// C = A·B + C is correct. VGRE holds each fragment as a whole tile in one
// thread, so mma_sync is self-contained (no warp/JIT harness needed). On a CPU
// with AVX-512 this exercises the AVX-512 mma kernel (mma_avx512, N==16); on the
// AMX path (amxEnabled) it exercises mma_amx_bf16_16x16x16. Run under Intel SDE
// (`sde64 -spr`) to actually execute those paths — this host has neither.

#include "vgre/compiler/wmma_emulation.h"

#include <cmath>
#include <cstdio>
#include <vector>

// A namespace ALIAS, not `using namespace nvcuda::wmma` — the latter pulls
// nvcuda::wmma::detail into the global scope of this TU, which makes the header's
// own global-scope templates (they reference the unrelated global ::detail as
// `detail::`) ambiguous under clang-cl. The alias avoids that.
namespace wm = nvcuda::wmma;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    constexpr int M = 16, N = 16, K = 16;
    std::vector<float> A(M * K), B(K * N), Cin(M * N), Cout(M * N), Ref(M * N);
    for (int i = 0; i < M * K; ++i) A[i] = 0.1f * (float)((i % 11) - 5);
    for (int i = 0; i < K * N; ++i) B[i] = 0.1f * (float)((i % 7) - 3);
    for (int i = 0; i < M * N; ++i) Cin[i] = 0.05f * (float)((i % 5) - 2);

    // Independent scalar reference: Ref = A·B + Cin (row-major).
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = Cin[m * N + n];
            for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
            Ref[m * N + n] = acc;
        }

    // WMMA path: load, mma, store.
    wm::fragment<wm::matrix_a, M, N, K, float, wm::row_major> a;
    wm::fragment<wm::matrix_b, M, N, K, float, wm::row_major> b;
    wm::fragment<wm::accumulator, M, N, K, float> c;
    wm::load_matrix_sync(a, A.data(), K);
    wm::load_matrix_sync(b, B.data(), N);
    wm::load_matrix_sync(c, Cin.data(), N, wm::mem_row_major);
    wm::mma_sync(c, a, b, c);
    wm::store_matrix_sync(Cout.data(), c, N, wm::mem_row_major);

    float maxd = 0.f;
    for (int i = 0; i < M * N; ++i) { float d = std::fabs(Cout[i] - Ref[i]); if (d > maxd) maxd = d; }
    // FP32 accumulation (AVX-512/AMX-bf16 or scalar) — small tolerance.
    CHECK(maxd < 1e-2f, "wmma 16x16x16 mma_sync == scalar reference");
    std::printf("wmma mma_sync 16x16x16 max|Δ| vs scalar = %g\n", maxd);

    if (g_fail == 0) std::printf("test_wmma_mma: ALL PASS\n");
    else             std::printf("test_wmma_mma: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
