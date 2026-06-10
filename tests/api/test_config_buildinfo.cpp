// Track 5 — config registry validation + vgre_get_build_info().

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/config_registry.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}
static bool has(const std::string &h, const std::string &n) {
    return h.find(n) != std::string::npos;
}

int main() {
    printf("=== Config Registry + Build Info (Track 5) ===\n");

    // ── build_info ───────────────────────────────────────────────────────────
    std::string bi = vgre_get_build_info();
    printf("  [info] build_info = %s\n", bi.c_str());
    check("build_info has version", has(bi, "\"version\""));
    check("build_info has git hash", has(bi, "\"git\""));
    check("build_info has simd level", has(bi, "\"simd\""));
    check("build_info has features block", has(bi, "\"features\"") && has(bi, "\"sqlite\""));

    // ── config validation ────────────────────────────────────────────────────
    // Seed the environment with a mix of valid, out-of-range, bad-enum, and
    // unknown VGRE_* vars, then validate.
    setenv("VGRE_METRICS_PORT", "8080", 1);            // valid
    setenv("VGRE_VIRTUAL_DEVICE_COUNT", "99", 1);      // out of range [1..8]
    setenv("VGRE_BENCHMARK", "MAYBE", 1);              // invalid enum (ON|OFF)
    setenv("VGRE_TOTALLY_UNKNOWN_KNOB", "x", 1);       // unknown VGRE_* var
    setenv("VGRE_PBKDF2_ITERATIONS", "5", 1);          // below min 10000

    int problems = vgre::common::ConfigRegistry::instance().validateAndLog();
    printf("  [info] validateAndLog reported %d problem(s)\n", problems);
    // At least the 4 seeded problems (range×2, enum×1, unknown×1) are flagged.
    check("validation flags the invalid/unknown vars", problems >= 4);

    // Idempotent: a second call does nothing (already validated).
    int again = vgre::common::ConfigRegistry::instance().validateAndLog();
    check("validateAndLog is idempotent (0 on second call)", again == 0);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
