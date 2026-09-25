// Phase 3, Track P3-8 — speculative decoding.
// The defining guarantee: the first accepted token is distributed EXACTLY as the
// target model p[0], regardless of the draft q. We verify this statistically
// (total-variation distance of the empirical distribution → 0), confirm that a
// matched draft yields many acceptances (the speedup), and that a perfect draft
// (q==p) accepts every token (count = K+1).

#include "vgre/core/speculative.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::serving;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static void softmax(std::vector<float>& v) {
    float mx = v[0]; for (float x : v) mx = std::max(mx, x);
    float s = 0; for (auto& x : v) { x = std::exp(x - mx); s += x; }
    for (auto& x : v) x /= s;
}

int main() {
    printf("=== Speculative decoding (draft + verify) ===\n");
    const int K = 4, V = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // Build target p[(K+1)·V] and a DIFFERENT draft q[K·V].
    std::vector<float> p((K + 1) * V), q(K * V);
    for (int i = 0; i <= K; ++i) { std::vector<float> row(V);
        for (auto& x : row) x = nd(rng); softmax(row);
        for (int v = 0; v < V; ++v) p[i * V + v] = row[v]; }
    for (int i = 0; i < K; ++i) { std::vector<float> row(V);
        for (auto& x : row) x = nd(rng) * 1.3f + 0.2f; softmax(row);
        for (int v = 0; v < V; ++v) q[i * V + v] = row[v]; }

    // ── Distribution equivalence: empirical first-token dist == p[0] ─────────
    const int N = 400000;
    std::vector<long> hist(V, 0);
    std::vector<int> out(K + 1);
    long totalAccepted = 0;
    for (int s = 0; s < N; ++s) {
        int n = speculative_decode(out.data(), p.data(), q.data(), K, V, rng);
        hist[out[0]]++;
        totalAccepted += n;
    }
    double tv = 0.0;   // total-variation distance
    for (int v = 0; v < V; ++v) tv += std::fabs((double)hist[v] / N - p[v]);
    tv *= 0.5;
    printf("  [info] first-token TV distance from target p[0] = %.4f\n", tv);
    check("first output token is distributed as the target p[0]", tv < 0.01);

    double avgTokens = (double)totalAccepted / N;
    printf("  [info] mean tokens/step = %.3f (>1 ⇒ speedup)\n", avgTokens);
    check("speculative step emits >1 token on average (speedup)", avgTokens > 1.0);

    // ── Perfect draft (q == p) accepts every proposal: count == K+1 ──────────
    {
        std::vector<float> qp(K * V);
        for (int i = 0; i < K; ++i) for (int v = 0; v < V; ++v) qp[i * V + v] = p[i * V + v];
        bool allFull = true;
        for (int s = 0; s < 2000; ++s) {
            int n = speculative_decode(out.data(), p.data(), qp.data(), K, V, rng);
            if (n != K + 1) allFull = false;
        }
        check("a perfect draft (q==p) accepts all K and emits the bonus token", allFull);
    }

    // ── Tree speculative verification (SpecInfer / Medusa-style) ─────────────
    // A branching draft tree: root → 3 children, child 0 → 2 grandchildren.
    // The defining guarantee is unchanged — the first emitted token is distributed
    // exactly as the target at the root — while the tree accepts a longer path on
    // average than a linear chain of the same depth.
    {
        const int M = 6;                       // nodes: 0=root, 1/2/3=children, 4/5=child-0's kids
        const int parent[M] = {-1, 0, 0, 0, 1, 1};
        // Random target + draft distribution per node (different from each other).
        std::vector<float> pT(M * V), qD(M * V);
        for (int i = 0; i < M; ++i) {
            std::vector<float> pr(V), qr(V);
            for (auto& x : pr) x = nd(rng);           softmax(pr);
            for (auto& x : qr) x = nd(rng) * 1.3f + 0.2f; softmax(qr);
            for (int v = 0; v < V; ++v) { pT[i * V + v] = pr[v]; qD[i * V + v] = qr[v]; }
        }

        // Distribution equivalence: resample the tree's tokens from the draft each
        // trial (as a real draft would propose), verify, histogram the first token.
        const int Nt = 400000;
        std::vector<long> th(V, 0);
        std::vector<int> tok(M, -1), to(M + 2);
        long treeAccepted = 0;
        for (int s = 0; s < Nt; ++s) {
            for (int i = 1; i < M; ++i)   // sample each node's token from its parent's draft row
                tok[i] = sample_categorical(&qD[parent[i] * V], V, rng);
            int n = tree_speculative_decode(to.data(), parent, tok.data(), pT.data(), qD.data(), M, V, rng);
            th[to[0]]++;
            treeAccepted += n;
        }
        double ttv = 0.0;
        for (int v = 0; v < V; ++v) ttv += std::fabs((double)th[v] / Nt - pT[v]);
        ttv *= 0.5;
        printf("  [info] tree first-token TV distance from target root = %.4f\n", ttv);
        check("tree verify: first token distributed as the target root p", ttv < 0.01);
        printf("  [info] tree mean tokens/step = %.3f\n", (double)treeAccepted / Nt);
        check("tree verify emits >1 token on average", (double)treeAccepted / Nt > 1.0);

        // Perfect draft (q==p at every node): the first child of each node is always
        // accepted (ratio 1), so it descends root→node1→node4 (a leaf) and emits a
        // bonus — exactly 3 tokens every time.
        {
            std::vector<float> qp = pT;               // draft == target everywhere
            bool always3 = true;
            for (int s = 0; s < 4000; ++s) {
                for (int i = 1; i < M; ++i) tok[i] = sample_categorical(&qp[parent[i] * V], V, rng);
                int n = tree_speculative_decode(to.data(), parent, tok.data(), pT.data(), qp.data(), M, V, rng);
                if (n != 3) always3 = false;
            }
            check("tree verify: a perfect draft descends the full path (3 tokens)", always3);
        }
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
