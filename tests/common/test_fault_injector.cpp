// Chaos fault injector — policy behaviour + env parsing + guard macro.

#include "vgre/testing/fault_injector.h"

#include <cstdio>
#include <cstdlib>

using namespace vgre::testing;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// A guarded operation: returns -1 when the "alloc" fault point fires, else 0.
static int guardedAlloc() {
    VGRE_FAULT_GUARD("alloc", -1);
    return 0;
}

int main() {
    std::printf("=== Chaos fault injector ===\n");
    FaultInjector& fi = FaultInjector::instance();

    // ── NONE / unconfigured never fails ──────────────────────────────────
    fi.reset();
    check("unconfigured point never fails", !fi.shouldFail("x"));

    // ── ONCE: fails exactly the first call ───────────────────────────────
    fi.configure("once", FaultPolicy::ONCE);
    check("ONCE fails first call", fi.shouldFail("once"));
    check("ONCE passes thereafter", !fi.shouldFail("once") && !fi.shouldFail("once"));

    // ── AFTER_N: fails once count exceeds N ──────────────────────────────
    fi.configure("aft", FaultPolicy::AFTER_N, 3);
    check("AFTER_N passes calls 1..3",
          !fi.shouldFail("aft") && !fi.shouldFail("aft") && !fi.shouldFail("aft"));
    check("AFTER_N fails call 4+", fi.shouldFail("aft") && fi.shouldFail("aft"));

    // ── EVERY_N: fails every Nth call ────────────────────────────────────
    fi.configure("evn", FaultPolicy::EVERY_N, 2);
    bool e[4] = {fi.shouldFail("evn"), fi.shouldFail("evn"), fi.shouldFail("evn"), fi.shouldFail("evn")};
    check("EVERY_N fails on 2nd,4th only", !e[0] && e[1] && !e[2] && e[3]);

    // ── ALWAYS ───────────────────────────────────────────────────────────
    fi.configure("alw", FaultPolicy::ALWAYS);
    check("ALWAYS fails every call", fi.shouldFail("alw") && fi.shouldFail("alw"));

    // ── RATE ~ 0.5 over many calls (deterministic stream) ────────────────
    fi.configure("rt", FaultPolicy::RATE, 0.5);
    int fails = 0;
    for (int i = 0; i < 1000; ++i) if (fi.shouldFail("rt")) ++fails;
    check("RATE 0.5 fails ~half (400..600)", fails > 400 && fails < 600);

    // ── guard macro ──────────────────────────────────────────────────────
    fi.reset();
    fi.configure("alloc", FaultPolicy::ONCE);
    check("guard returns fault value on injected fault", guardedAlloc() == -1);
    check("guard returns normal value otherwise", guardedAlloc() == 0);

    // ── latency ──────────────────────────────────────────────────────────
    fi.setLatency("slow", 250);
    check("latency configured + read back", fi.latencyMs("slow") == 250 && fi.latencyMs("none") == 0);

    // ── env parsing ──────────────────────────────────────────────────────
    fi.reset();
    setenv("VGRE_FAULT_INJECT", "net:always;io:after=2;db:once", 1);
    fi.loadFromEnv();
    check("env net:always", fi.shouldFail("net"));
    check("env io:after=2", !fi.shouldFail("io") && !fi.shouldFail("io") && fi.shouldFail("io"));
    check("env db:once", fi.shouldFail("db") && !fi.shouldFail("db"));

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
