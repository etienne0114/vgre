// Phase 3, Track P3-7 — Blackwell tcgen05 Tensor Memory (TMEM) data path.
//
// Validates the TMEM model (alloc/dealloc, tile scatter/gather in the (lane,
// column) layout, K-loop accumulate-in-place, and generic ld/st/cp word moves),
// then drives two real tcgen05 GEMMs entirely through TMEM — alloc the
// accumulator, run the MMA per K-tile into TMEM (the existing tensor-core math),
// then tcgen05.ld the result out — and checks both match an independent reference
// over the SAME quantized operands. A wrong TMEM layout, a missing accumulate, or
// a bad address yields a wrong GEMM, so a match proves the data path.

#include "vgre/core/tmem.h"
#include "vgre/compiler/wmma_emulation.h"   // tcgen05/wgmma MMA helpers + codecs

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace vgre::tmem;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static uint16_t f32_to_bf16(float f) {           // round-to-nearest-even
    uint32_t x; std::memcpy(&x, &f, 4);
    return (uint16_t)((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}
static float bf16_to_f32(uint16_t v) {
    uint32_t f = (uint32_t)v << 16; float r; std::memcpy(&r, &f, 4); return r;
}

int main() {
    printf("=== Blackwell tcgen05 Tensor Memory (TMEM) data path (P3-7) ===\n");

    // ── 1. TMEM memory model ─────────────────────────────────────────────────
    {
        TensorMemory tm;
        uint32_t a0 = tm.alloc(256);
        uint32_t a1 = tm.alloc(128);
        bool ok = (addr_col(a0) == 0) && (addr_col(a1) == 256) &&
                  (tm.columns_used() == 384);
        check("tcgen05.alloc bump-allocates distinct column ranges", ok);

        // store_tile / load_tile round-trip a [4x8] tile at an allocated address.
        float src[4 * 8], dst[4 * 8];
        for (int i = 0; i < 32; ++i) src[i] = (float)(i + 1);
        tm.store_tile(a1, src, 4, 8);
        tm.load_tile(dst, a1, 4, 8);
        bool rt = std::memcmp(src, dst, sizeof src) == 0;
        // ...and it landed at the right (lane,column): element (m,n) at lane m, col 256+n.
        bool placed = bf16_to_f32(0) == 0.0f;  // (silence) — explicit check below
        placed = (*reinterpret_cast<const float*>(tm.word(2, 256 + 3)) == src[2 * 8 + 3]);
        check("store_tile/load_tile round-trip in the (lane,column) layout", rt && placed);

        // accumulate_tile: D += tile.
        tm.accumulate_tile(a1, src, 4, 8);
        tm.load_tile(dst, a1, 4, 8);
        bool acc = true;
        for (int i = 0; i < 32; ++i) if (dst[i] != 2.0f * src[i]) acc = false;
        check("accumulate_tile performs D += tile (K-loop accumulate)", acc);

        // generic ld/st word moves (32x32b.x2 fragment: 8 lanes x 2 words).
        uint32_t regsIn[8 * 2], regsOut[8 * 2];
        for (int i = 0; i < 16; ++i) regsIn[i] = 0xA5A50000u + i;
        uint32_t a2 = tm.alloc(2);
        tm.st(a2, regsIn, 8, 2);
        tm.ld(regsOut, a2, 8, 2);
        check("tcgen05.st/.ld round-trip a register fragment", std::memcmp(regsIn, regsOut, sizeof regsIn) == 0);

        // tcgen05.cp: stage SMEM → TMEM.
        uint32_t smem[8 * 2]; for (int i = 0; i < 16; ++i) smem[i] = 0xCAFE0000u + i;
        uint32_t a3 = tm.alloc(2);
        tm.cp_from_smem(a3, smem, 8, 2);
        tm.ld(regsOut, a3, 8, 2);
        check("tcgen05.cp stages SMEM into TMEM", std::memcmp(smem, regsOut, sizeof smem) == 0);

        // dealloc is LIFO.
        int before = tm.columns_used();
        tm.dealloc(a3, 2);
        check("tcgen05.dealloc releases the allocation (LIFO)", tm.columns_used() == before - 2);
    }

    // ── 2. tcgen05 BF16 GEMM (M=64,N=256,K=32) through TMEM, K-loop accumulate ─
    {
        const int M = 64, N = 256, K = 32, KT = 16;     // BF16 K-tile = 16
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        std::vector<uint16_t> A(M * K), B(K * N);
        for (auto& x : A) x = f32_to_bf16(nd(rng));
        for (auto& x : B) x = f32_to_bf16(nd(rng));

        TensorMemory tm;
        uint32_t acc = tm.alloc(N);                     // FP32 accumulator in TMEM

        std::vector<uint16_t> Asub(M * KT), Bsub(KT * N);
        std::vector<float> temp(M * N);
        for (int k0 = 0; k0 < K; k0 += KT) {            // K-loop: mma → accumulate in TMEM
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < KT; ++k) Asub[m * KT + k] = A[m * K + (k0 + k)];
            for (int k = 0; k < KT; ++k)
                for (int n = 0; n < N; ++n) Bsub[k * N + n] = B[(k0 + k) * N + n];
            std::fill(temp.begin(), temp.end(), 0.0f);
            vgre_wgmma_m64n256k16_bf16_f32(temp.data(),
                vgre_make_wgmma_desc(Asub.data()), vgre_make_wgmma_desc(Bsub.data()));
            tm.accumulate_tile(acc, temp.data(), M, N);  // D += A_kt · B_kt
        }
        std::vector<float> out(M * N);
        tm.load_tile(out.data(), acc, M, N);             // tcgen05.ld result out of TMEM

        double maxErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k) ref += bf16_to_f32(A[m * K + k]) * bf16_to_f32(B[k * N + n]);
                maxErr = std::max(maxErr, (double)std::fabs(out[m * N + n] - ref));
            }
        printf("  [info] tcgen05 BF16 GEMM via TMEM max abs err = %.2e\n", maxErr);
        check("tcgen05 BF16 GEMM through TMEM == reference", maxErr < 1e-2);
    }

    // ── 3. tcgen05 FP8 (E4M3) GEMM (M=64,N=256,K=32) through TMEM ─────────────
    {
        const int M = 64, N = 256, K = 32;
        std::mt19937 rng(5);
        std::vector<uint8_t> A(M * K), B(K * N);
        // Small representable E4M3 values (exp <= 6 ⇒ never NaN).
        auto e4 = [&]{ return (uint8_t)(((rng() & 1u) << 7) | ((rng() % 7u) << 3) | (rng() & 0x7u)); };
        for (auto& x : A) x = e4();
        for (auto& x : B) x = e4();

        TensorMemory tm;
        uint32_t acc = tm.alloc(N);
        std::vector<float> temp(M * N, 0.0f);
        vgre_tcgen05_m64n256k32_e4m3_f32(temp.data(),
            vgre_make_wgmma_desc(A.data()), vgre_make_wgmma_desc(B.data()));
        tm.store_tile(acc, temp.data(), M, N);
        std::vector<float> out(M * N);
        tm.load_tile(out.data(), acc, M, N);

        double maxErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k)
                    ref += detail::fp8e4m3_to_f32(A[m * K + k])
                         * detail::fp8e4m3_to_f32(B[k * N + n]);
                maxErr = std::max(maxErr, (double)std::fabs(out[m * N + n] - ref));
            }
        printf("  [info] tcgen05 FP8(E4M3) GEMM via TMEM max abs err = %.2e\n", maxErr);
        check("tcgen05 FP8 E4M3 GEMM through TMEM == reference", maxErr < 1e-3);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
