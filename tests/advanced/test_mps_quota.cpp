// Track P — MPS per-client resource quota tests.
//
// 1. loadMPSPolicy parses a JSON policy file (maxMemoryBytes / maxThreadFraction
//    / maxQueueDepth), with defaults for missing/invalid fields.
// 2. End-to-end: a client's allocations are capped at maxMemoryBytes by the MPS
//    daemon, and freeing memory refunds the quota.

#include "vgre/advanced/mps_control.h"
#include "vgre/api/vgre_c_api.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>  // getpid

using namespace vgre::mps;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static std::string writeTempPolicy(const std::string& json) {
    std::string path = "/tmp/vgre_mps_policy_" + std::to_string(::getpid()) + ".json";
    std::ofstream f(path);
    f << json;
    f.close();
    return path;
}

int main() {
    printf("=== MPS Per-Client Quota Tests (Track P) ===\n");
    vgre_init();  // bring up the runtime so the MPS daemon's allocator works

    // ── 1. loadMPSPolicy: valid file ──────────────────────────────────────────
    {
        std::string p = writeTempPolicy(
            "{ \"maxMemoryBytes\": 1024, \"maxThreadFraction\": 0.5, "
            "\"maxQueueDepth\": 8 }");
        VgreMPSPolicy pol = loadMPSPolicy(p);
        check("loadMPSPolicy maxMemoryBytes", pol.maxMemoryBytes == 1024);
        check("loadMPSPolicy maxThreadFraction",
              std::fabs(pol.maxThreadFraction - 0.5f) < 1e-6f);
        check("loadMPSPolicy maxQueueDepth", pol.maxQueueDepth == 8);
        std::remove(p.c_str());
    }

    // ── 2. loadMPSPolicy: missing/empty path → defaults ──────────────────────
    {
        VgreMPSPolicy def = loadMPSPolicy("");
        check("loadMPSPolicy default maxMemoryBytes", def.maxMemoryBytes == UINT64_MAX);
        check("loadMPSPolicy default maxThreadFraction",
              std::fabs(def.maxThreadFraction - 1.0f) < 1e-6f);
        check("loadMPSPolicy default maxQueueDepth", def.maxQueueDepth == 256);
    }

    // ── 3. loadMPSPolicy: partial file keeps defaults for absent fields ───────
    {
        std::string p = writeTempPolicy("{ \"maxMemoryBytes\": 4096 }");
        VgreMPSPolicy pol = loadMPSPolicy(p);
        check("partial policy: set field applied", pol.maxMemoryBytes == 4096);
        check("partial policy: absent field default",
              pol.maxQueueDepth == 256 &&
              std::fabs(pol.maxThreadFraction - 1.0f) < 1e-6f);
        std::remove(p.c_str());
    }

    // ── 4. End-to-end memory quota enforcement ───────────────────────────────
    // Policy must be installed before the daemon services its first MALLOC,
    // because the policy is loaded once on first use.
    {
        std::string p = writeTempPolicy("{ \"maxMemoryBytes\": 1024 }");
        vgre_set_config("VGRE_MPS_POLICY_FILE", p.c_str());

        MPSServer server("");
        bool started = server.start();
        check("MPS server started", started);

        if (started) {
            vgre_set_config("VGRE_MPS_PIPE", server.socketPath().c_str());
            MPSClient* c = MPSClient::instance();
            bool connected = c && c->isConnected();
            check("MPS client connected", connected);

            if (connected) {
                uint64_t p1 = c->malloc(512);
                check("alloc 512 within quota succeeds", p1 != 0);

                uint64_t p2 = c->malloc(1024);  // 512 + 1024 > 1024 → denied
                check("alloc 1024 over quota denied", p2 == 0);

                check("free succeeds", c->free(p1));

                uint64_t p3 = c->malloc(1024);  // quota refunded → fits exactly
                check("alloc 1024 after free succeeds", p3 != 0);
                if (p3) c->free(p3);

                c->disconnect();
            }
            server.stop();
        }
        std::remove(p.c_str());
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
