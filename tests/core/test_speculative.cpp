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

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
