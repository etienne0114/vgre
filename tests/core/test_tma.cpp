// Phase 3, Track P3-5 — Tensor Memory Accelerator (TMA).
// Validates the tensor-map box copy (strided global gather → box-contiguous
// SMEM), out-of-bounds zero-fill at tile boundaries, mbarrier transaction-byte
// completion, and an end-to-end TMA-tiled GEMM (load A/B tiles via TMA, GEMM the
// SMEM tiles, compare to a direct GEMM).

#include "vgre/core/tma.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::tma;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Build a row-major float tensor map [H,W] with box [bh,bw].
static TensorMap map2d(const float* g, int H, int W, int bh, int bw) {
    uint64_t dim[2]    = {(uint64_t)H, (uint64_t)W};
    uint64_t stride[2] = {(uint64_t)W * sizeof(float), sizeof(float)};  // row-major
    uint32_t box[2]    = {(uint32_t)bh, (uint32_t)bw};
    return tensor_map_encode_tiled(g, 2, dim, stride, box, sizeof(float));
}

int main() {
    printf("=== Tensor Memory Accelerator (TMA) ===\n");
    const int H = 10, W = 12, bh = 4, bw = 4;
    std::vector<float> G(H * W);
    for (int i = 0; i < H * W; ++i) G[i] = (float)i;
    TensorMap tm = map2d(G.data(), H, W, bh, bw);

    // 1. Interior box copy == the global sub-tile.
    {
        std::vector<float> smem(bh * bw, -1.0f);
        Mbarrier mb; mbarrier_init(mb); mbarrier_expect_tx(mb, box_bytes(tm));
        int32_t coord[2] = {2, 3};                          // start (row 2, col 3)
        cp_async_bulk_tensor(smem.data(), tm, coord, &mb);
        bool ok = true;
        for (int i = 0; i < bh; ++i)
            for (int j = 0; j < bw; ++j)
                if (smem[i * bw + j] != G[(2 + i) * W + (3 + j)]) ok = false;
        check("interior box copy == global sub-tile", ok);
        check("mbarrier completes after the transfer", mbarrier_try_wait(mb));
        check("mbarrier transaction bytes == box bytes", mb.arrivedBytes == box_bytes(tm));
    }

    // 2. Boundary box (partly out of bounds) zero-fills the OOB region.
    {
        std::vector<float> smem(bh * bw, -1.0f);
        Mbarrier mb; mbarrier_init(mb);
        int32_t coord[2] = {8, 10};   // rows 8..11 (8,9 valid; 10,11 OOB); cols 10..13 (10,11 valid)
        cp_async_bulk_tensor(smem.data(), tm, coord, &mb);
        bool ok = true;
        for (int i = 0; i < bh; ++i)
            for (int j = 0; j < bw; ++j) {
                int gr = 8 + i, gc = 10 + j;
                float expect = (gr < H && gc < W) ? G[gr * W + gc] : 0.0f;
                if (smem[i * bw + j] != expect) ok = false;
            }
        check("boundary box zero-fills out-of-bounds elements (real TMA padding)", ok);
    }

    // 3. Plain bulk copy.
    {
        std::vector<float> src(64), dst(64, 0.0f);
        for (int i = 0; i < 64; ++i) src[i] = (float)(i + 100);
        Mbarrier mb; mbarrier_init(mb);
        cp_async_bulk(dst.data(), src.data(), 64 * sizeof(float), &mb);
        bool ok = (dst == src) && mb.arrivedBytes == 64 * sizeof(float);
        check("plain cp.async.bulk copies contiguously + completes tx", ok);
    }

    // 4. End-to-end TMA-tiled GEMM: stream A and B tiles via TMA into SMEM, GEMM
    //    the SMEM tiles, accumulate, and compare to a direct GEMM.
    {
        const int M = 8, N = 8, K = 16, T = 4;            // K-tiles of width T
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), Cref(M * N, 0.0f);
        for (auto& x : A) x = nd(rng);
        for (auto& x : B) x = nd(rng);

        // Tensor maps: A box [M,T] over [M,K]; B box [T,N] over [K,N].
        uint64_t aDim[2]={(uint64_t)M,(uint64_t)K}, aStr[2]={(uint64_t)K*4,4}; uint32_t aBox[2]={(uint32_t)M,(uint32_t)T};
        uint64_t bDim[2]={(uint64_t)K,(uint64_t)N}, bStr[2]={(uint64_t)N*4,4}; uint32_t bBox[2]={(uint32_t)T,(uint32_t)N};
        TensorMap tmA = tensor_map_encode_tiled(A.data(), 2, aDim, aStr, aBox, 4);
        TensorMap tmB = tensor_map_encode_tiled(B.data(), 2, bDim, bStr, bBox, 4);

        std::vector<float> sA(M * T), sB(T * N);
        for (int k0 = 0; k0 < K; k0 += T) {
            Mbarrier ma, mb; mbarrier_init(ma); mbarrier_init(mb);
            mbarrier_expect_tx(ma, box_bytes(tmA)); mbarrier_expect_tx(mb, box_bytes(tmB));
            int32_t ca[2] = {0, k0}, cb[2] = {k0, 0};
            cp_async_bulk_tensor(sA.data(), tmA, ca, &ma);    // A[:, k0:k0+T]
            cp_async_bulk_tensor(sB.data(), tmB, cb, &mb);    // B[k0:k0+T, :]
            if (!mbarrier_try_wait(ma) || !mbarrier_try_wait(mb)) { check("TMA tiles ready", false); break; }
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < T; ++kk) acc += sA[m * T + kk] * sB[kk * N + n];
                    C[m * N + n] += acc;
                }
        }
        double maxErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
                maxErr = std::max(maxErr, (double)std::fabs(C[m * N + n] - acc));
            }
        printf("  [info] TMA-tiled GEMM max abs err = %.2e\n", maxErr);
        check("TMA-tiled GEMM (A/B streamed via TMA) == direct GEMM", maxErr < 1e-4);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
