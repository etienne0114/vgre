// Fault-tolerant training — Byzantine-robust aggregation + elastic membership.

#include "vgre/advanced/fault_tolerance.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}
static bool near(const std::vector<double>& v, double target, double tol) {
    for (double x : v) if (std::fabs(x - target) > tol) return false;
    return true;
}

int main() {
    std::printf("=== Fault-tolerant training (robust aggregation + elastic) ===\n");

    // 8 honest gradients ≈ [1,1,1], 2 Byzantine poison ≈ [100,100,100].
    std::vector<std::vector<double>> grads;
    for (int i = 0; i < 8; ++i) grads.push_back({1.0 + 0.01 * i, 1.0, 1.0 - 0.01 * i});
    grads.push_back({100, 100, 100});   // both poisons pull the SAME direction so
    grads.push_back({120, 120, 120});   // the naive mean is genuinely corrupted
    int f = 2;

    // ── coordinate-wise median resists the outliers ──────────────────────
    {
        auto m = RobustAggregator::median(grads);
        check("median ignores Byzantine outliers (~1)", near(m, 1.0, 0.2));
    }
    // ── trimmed mean (trim 2 each end) drops the poison ──────────────────
    {
        auto t = RobustAggregator::trimmedMean(grads, 2);
        check("trimmed mean ~1 after dropping extremes", near(t, 1.0, 0.2));
    }
    // ── Krum selects an honest gradient ──────────────────────────────────
    {
        int idx = RobustAggregator::krumSelect(grads, f);
        check("Krum picks an honest gradient (index < 8)", idx >= 0 && idx < 8);
        auto k = RobustAggregator::krum(grads, f);
        check("Krum result ~1 (not poisoned)", near(k, 1.0, 0.3));
    }
    // ── Multi-Krum averages honest gradients ─────────────────────────────
    {
        auto mk = RobustAggregator::multiKrum(grads, f, 5);
        check("Multi-Krum result ~1", near(mk, 1.0, 0.3));
    }
    // ── naive mean WOULD be corrupted (sanity: confirms the threat) ──────
    {
        std::vector<double> mean(3, 0.0);
        for (auto& g : grads) for (int d = 0; d < 3; ++d) mean[d] += g[d];
        for (auto& v : mean) v /= grads.size();
        check("naive mean is NOT robust (~0, off from 1)", std::fabs(mean[0] - 1.0) > 0.3);
    }

    // ── Elastic membership ───────────────────────────────────────────────
    {
        ElasticMembership em;
        int r0 = em.join("w0"), r1 = em.join("w1"), r2 = em.join("w2");
        check("ranks assigned densely 0,1,2", r0 == 0 && r1 == 1 && r2 == 2);
        check("world size 3", em.worldSize() == 3);
        uint64_t e3 = em.epoch();
        check("idempotent join returns same rank", em.join("w1") == 1 && em.epoch() == e3);

        check("leave removes worker", em.leave("w1"));
        check("world compacted to 2", em.worldSize() == 2);
        int rank = -1;
        em.getRank("w2", rank);
        check("ranks recompacted dense after leave (w2 → 1)", rank == 1);
        check("epoch bumped on membership change", em.epoch() > e3);
        check("leaving unknown worker is a no-op", !em.leave("ghost"));
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
