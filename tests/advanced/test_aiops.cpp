// AIOps + security analytics — EWMA/MAD anomaly detection + audit-log threats.

#include "vgre/advanced/aiops.h"

#include <cstdio>
#include <vector>

using namespace vgre::advanced;
using vgre::compliance::AuditOutcome;
using vgre::compliance::AuditRecord;
using vgre::compliance::AuditSeverity;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== AIOps + security analytics ===\n");

    // ── EWMA control-chart: steady stream, then a spike ──────────────────
    {
        EwmaAnomalyDetector d(/*alpha*/0.3, /*k*/3.0, /*warmup*/5);
        bool flaggedNormal = false, flaggedSpike = false;
        // steady ~100 with small noise
        double noise[] = {100, 101, 99, 100, 102, 98, 101, 99, 100, 100};
        for (double x : noise) if (d.observe(x)) flaggedNormal = true;
        double baseline = d.mean();                 // baseline BEFORE the spike
        flaggedSpike = d.observe(500.0);            // big spike
        check("steady stream raises no anomaly", !flaggedNormal);
        check("a large spike is flagged", flaggedSpike);
        check("baseline mean tracked ~100 during steady stream", baseline > 95 && baseline < 105);
    }

    // ── MAD robust outliers (resists the outliers themselves) ────────────
    {
        std::vector<double> xs = {10, 11, 9, 10, 12, 10, 9, 11, 200, -150};
        OutlierResult r = Anomaly::madOutliers(xs, 3.5);
        check("MAD finds the two extreme outliers (idx 8,9)",
              r.indices.size() == 2 && r.indices[0] == 8 && r.indices[1] == 9);
        check("MAD median ~10 (robust)", r.median > 9.0 && r.median < 11.0);
    }

    // ── Security analytics over crafted audit records ────────────────────
    {
        std::vector<AuditRecord> events;
        auto add = [&](const char* actor, AuditOutcome o) {
            AuditRecord r; r.actor = actor; r.action = "auth.login"; r.outcome = o; events.push_back(r);
        };
        // attacker: 7 denied logins (brute force). normal users: a few successes.
        for (int i = 0; i < 7; ++i) add("attacker", AuditOutcome::DENIED);
        for (int i = 0; i < 10; ++i) add("alice", AuditOutcome::SUCCESS);
        add("bob", AuditOutcome::SUCCESS);
        add("bob", AuditOutcome::DENIED); // 1 denial — below threshold

        SecurityReport rep = SecurityAnalytics::analyze(events, /*bruteForceThreshold*/5);
        check("total + denied counts correct", rep.totalEvents == 19 && rep.deniedEvents == 8);
        check("brute-force finding raised for attacker only",
              rep.findings.size() == 1 && rep.findings[0].actor == "attacker" &&
              rep.findings[0].kind == "brute_force" && rep.findings[0].count == 7);
        check("posture score reduced below 100", rep.postureScore < 100.0);

        // a clean trail → perfect posture, no findings
        std::vector<AuditRecord> clean;
        for (int i = 0; i < 20; ++i) { AuditRecord r; r.actor = "u"; r.outcome = AuditOutcome::SUCCESS; clean.push_back(r); }
        SecurityReport ok = SecurityAnalytics::analyze(clean);
        check("clean trail → posture 100, no threats", ok.postureScore == 100.0 && ok.findings.empty());
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
