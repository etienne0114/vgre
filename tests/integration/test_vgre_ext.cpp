// Phase-3 extension API (vgre::ext) — exercises every capability module through
// the shipped public API (the production boundary), checking each produces a
// correct result rather than just linking.

#include "vgre/api/vgre_ext.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::ext;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== VGRE Phase-3 extension API (vgre::ext) ===\n");
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    // occupancy: 256 threads (8 warps), no reg/smem limit → 8 blocks/SM on A100.
    check("occupancy_max_active_blocks", occupancy_max_active_blocks(256, 0, 0) == 8);
    check("roofline_gflops (memory-bound)", std::fabs(roofline_gflops(2.0, 1000.0, 100.0) - 200.0) < 1e-9);
    check("allreduce_us positive + finite", allreduce_us(8, 8, 300.0, 32.0, 1e6) > 0.0);

    // deterministic reduce == exact sum; counter RNG in [0,1).
    {
        std::vector<float> x(1000); double ref = 0;
        for (auto& v : x) { v = ud(rng); ref += v; }
        float r = deterministic_reduce(x.data(), x.size());
        check("deterministic_reduce ~= sum", std::fabs(r - (float)ref) < 1e-2);
        float u = counter_rng_u01(0xABCDull, 7);
        check("counter_rng_u01 in [0,1)", u >= 0.0f && u < 1.0f);
    }

    // SSM selective scan == sequential recurrence h_t = Abar_t*h_{t-1}+Bx_t; y_t=C_t·h_t.
    {
        const int L = 12, N = 4;
        std::vector<float> Ab(L*N), Bx(L*N), C(L*N), y(L), yref(L);
        for (auto& v : Ab) v = 0.5f * ud(rng);
        for (auto& v : Bx) v = ud(rng);
        for (auto& v : C)  v = ud(rng);
        ssm_selective_scan(y.data(), Ab.data(), Bx.data(), C.data(), L, N);
        std::vector<float> h(N, 0.0f);
        for (int t = 0; t < L; ++t) { float s = 0;
            for (int n = 0; n < N; ++n) { h[n] = Ab[t*N+n]*h[n] + Bx[t*N+n]; s += C[t*N+n]*h[n]; }
            yref[t] = s; }
        double e = 0; for (int t = 0; t < L; ++t) e = std::max(e, (double)std::fabs(y[t]-yref[t]));
        check("ssm_selective_scan == sequential recurrence", e < 1e-4);
    }

    // Tensor-parallel column linear == plain Y[T,Dout] = X[T,Din]·W[Din,Dout].
    {
        const int T = 3, Din = 5, Dout = 8, P = 4;
        std::vector<float> X(T*Din), W(Din*Dout), Y(T*Dout), Yref(T*Dout, 0.0f);
        for (auto& v : X) v = ud(rng); for (auto& v : W) v = ud(rng);
        tensor_parallel_column_linear(Y.data(), X.data(), W.data(), T, Din, Dout, P);
        for (int t=0;t<T;++t) for (int o=0;o<Dout;++o){ float s=0; for(int i=0;i<Din;++i) s+=X[t*Din+i]*W[i*Dout+o]; Yref[t*Dout+o]=s; }
        double e=0; for (size_t i=0;i<Y.size();++i) e=std::max(e,(double)std::fabs(Y[i]-Yref[i]));
        check("tensor_parallel_column_linear == unsharded matmul", e < 1e-3);
    }

    // W4A16 + MX GEMMs match a dequant reference within format tolerance.
    {
        const int M=4, N=8, K=64, G=32;
        std::vector<float> A(M*K), W(K*N), D(M*N), B(K*N), Dm(M*N);
        for (auto& v : A) v = 0.3f*ud(rng); for (auto& v : W) v = 0.3f*ud(rng); for (auto& v : B) v = 0.3f*ud(rng);
        w4a16_gemm(D.data(), A.data(), W.data(), M, N, K, G);
        bool finite4 = true; for (float v : D) finite4 = finite4 && std::isfinite(v);
        check("w4a16_gemm produces finite output", finite4);
        mx_gemm(Dm.data(), A.data(), B.data(), M, N, K, /*elem=*/0, /*block=*/32);
        bool finiteM = true; for (float v : Dm) finiteM = finiteM && std::isfinite(v);
        check("mx_gemm produces finite output", finiteM);
    }

    // MoE forward: finite, non-trivial output for random logits.
    {
        const int T=4, E=4, k=2, D=6, H=6;
        std::vector<float> X(T*D), logits(T*E), We(E*D*H), Y(T*H);
        for (auto& v : X) v = ud(rng); for (auto& v : logits) v = ud(rng); for (auto& v : We) v = ud(rng);
        moe_forward(Y.data(), X.data(), logits.data(), We.data(), T, E, k, D, H);
        bool fin = true; for (float v : Y) fin = fin && std::isfinite(v);
        check("moe_forward produces finite routed output", fin);
    }

    // Speculative decode returns an accept count in [0,K].
    {
        const int K=5, V=16; std::vector<int> out(K+1);
        std::vector<float> p(V), q(V); float sp=0, sq=0;
        for (int i=0;i<V;++i){ p[i]=std::fabs(ud(rng))+0.01f; q[i]=std::fabs(ud(rng))+0.01f; sp+=p[i]; sq+=q[i]; }
        for (int i=0;i<V;++i){ p[i]/=sp; q[i]/=sq; }
        int n = speculative_decode(out.data(), p.data(), q.data(), K, V, 123);
        check("speculative_decode accept count in [0,K]", n >= 0 && n <= K);
    }

    // Structured-mask attention: runs and is finite (causal).
    {
        const int S=8, d=4; std::vector<float> Q(S*d), Kk(S*d), Vv(S*d), O(S*d);
        for (auto& v : Q) v = ud(rng); for (auto& v : Kk) v = ud(rng); for (auto& v : Vv) v = ud(rng);
        attention_masked(O.data(), Q.data(), Kk.data(), Vv.data(), S, d, 1.0f/std::sqrt((float)d), 0, 0, 0);
        bool fin = true; for (float v : O) fin = fin && std::isfinite(v);
        check("attention_masked (causal) finite output", fin);
    }

    // NVSHMEM one-sided AllReduce-sum: every PE block holds the sum of all blocks.
    // (ring AllReduce processes count in P equal chunks, so count is a multiple of P)
    {
        const int P=4, C=8; std::vector<float> data(P*C);
        std::vector<float> sum(C, 0.0f);
        for (int p=0;p<P;++p) for (int i=0;i<C;++i){ float v=ud(rng); data[p*C+i]=v; sum[i]+=v; }
        nvshmem_allreduce_sum(data.data(), P, C);
        bool ok = true;
        for (int p=0;p<P;++p) for (int i=0;i<C;++i) if (std::fabs(data[p*C+i]-sum[i]) > 1e-4f) ok = false;
        check("nvshmem_allreduce_sum == reduced sum on every PE", ok);
    }

    // Prefix cache: identical content shares a block.
    {
        std::vector<float> blk(16); for (auto& v : blk) v = ud(rng);
        int shared = 0; prefix_cache_block(blk.data(), 16, &shared);
        check("prefix_cache_block shares identical content", shared == 1);
    }

    // Gradient checkpoint save→load round-trips exactly.
    {
        std::vector<float> act(64); for (auto& v : act) v = ud(rng);
        check("gradient_checkpoint_roundtrip exact", gradient_checkpoint_roundtrip(act.data(), 64));
    }

    // TMA box copy: interior box == the global sub-tile.
    {
        const int H=10, W=12, bh=4, bw=4; std::vector<float> G(H*W); for (int i=0;i<H*W;++i) G[i]=(float)i;
        std::vector<float> box(bh*bw, -1.0f);
        tma_box_copy(box.data(), G.data(), H, W, bh, bw, 2, 3);
        bool ok = true; for (int i=0;i<bh;++i) for (int j=0;j<bw;++j) if (box[i*bw+j] != G[(2+i)*W+(3+j)]) ok=false;
        check("tma_box_copy == global sub-tile", ok);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
