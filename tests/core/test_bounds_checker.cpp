// Device-memory bounds checker — red-zone canaries detect over/under-flow.

#include "vgre/core/memory/bounds_checker.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace vgre::core;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== Device-memory bounds checker (red zones) ===\n");
    BoundsChecker& bc = BoundsChecker::instance();
    bc.enable(true);
    bc.setRedzoneBytes(16);

    // ── clean in-bounds use → OK on check + free ─────────────────────────
    {
        const size_t N = 256;
        auto* p = static_cast<uint8_t*>(bc.guardedAlloc(N));
        check("guardedAlloc returns a pointer", p != nullptr);
        std::memset(p, 0x11, N);             // write exactly within bounds
        check("in-bounds write keeps red zones intact", bc.check(p) == GuardStatus::Ok);
        check("clean free reports OK", bc.guardedFree(p) == GuardStatus::Ok);
    }

    // ── overflow: write 1 byte past the end → detected ───────────────────
    {
        const size_t N = 100;
        auto* p = static_cast<uint8_t*>(bc.guardedAlloc(N));
        p[N] = 0xFF;                          // 1 byte past the user region
        check("overflow detected by check()", bc.check(p) == GuardStatus::Overflow);
        check("overflow detected at free()", bc.guardedFree(p) == GuardStatus::Overflow);
    }

    // ── underflow: write 1 byte before the start → detected ──────────────
    {
        const size_t N = 64;
        auto* p = static_cast<uint8_t*>(bc.guardedAlloc(N));
        p[-1] = 0xFF;                         // 1 byte before the user region
        check("underflow detected", bc.check(p) == GuardStatus::Underflow);
        bc.guardedFree(p);
    }

    // ── unknown pointer ──────────────────────────────────────────────────
    {
        int local = 0;
        check("unknown pointer reported", bc.check(&local) == GuardStatus::UnknownPtr);
    }

    // ── checkAll over several live allocations ───────────────────────────
    {
        auto* a = static_cast<uint8_t*>(bc.guardedAlloc(32));
        auto* b = static_cast<uint8_t*>(bc.guardedAlloc(32));
        auto* c = static_cast<uint8_t*>(bc.guardedAlloc(32));
        b[32 + 3] = 0x01;                     // corrupt b's trailing red zone
        std::vector<std::pair<const void*, GuardStatus>> bad;
        size_t n = bc.checkAll(bad);
        check("checkAll finds exactly the corrupted allocation",
              n == 1 && !bad.empty() && bad[0].first == b && bad[0].second == GuardStatus::Overflow);
        bc.guardedFree(a); bc.guardedFree(b); bc.guardedFree(c);
        check("all freed → no live allocations", bc.liveCount() == 0);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
