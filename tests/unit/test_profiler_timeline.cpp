// Profiler timestamp-ordered timeline (skip_list-backed) — time-window queries.
//
// Records profile events out of timestamp order, then verifies the skip-list
// timeline returns the events in a [startMs,endMs] window in ascending time order
// (and excludes those outside it) — the production use of vgre::SkipList::range.

#include "vgre/advanced/runtime_profiler.h"

#include <cstdio>
#include <vector>

using vgre::advanced::RuntimeProfiler;
using vgre::advanced::ProfileEvent;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static void rec(RuntimeProfiler& p, const char* name, uint64_t tms) {
    ProfileEvent e;
    e.kernelName = name;
    e.timestamp_ms = tms;
    p.recordEvent(e);
}

int main() {
    printf("=== Profiler timeline (skip_list time-window query) ===\n");
    RuntimeProfiler& p = RuntimeProfiler::instance();
    p.clear();
    p.setEnabled(true);

    // Record out of order; timestamps in ms.
    rec(p, "k_300", 300);
    rec(p, "k_100", 100);
    rec(p, "k_500", 500);
    rec(p, "k_200", 200);
    rec(p, "k_100b", 100);   // duplicate ms — must still be retained + ordered
    rec(p, "k_400", 400);

    // Window [150, 450] → k_200, k_300, k_400 in ascending time order.
    {
        auto w = p.getEventsInWindow(150, 450);
        bool ok = w.size() == 3 &&
                  w[0].timestamp_ms == 200 && w[1].timestamp_ms == 300 && w[2].timestamp_ms == 400;
        check("window [150,450] returns the 3 in-range events, time-ordered", ok);
    }

    // Inclusive endpoints: window [100,100] → both 100-ms events.
    {
        auto w = p.getEventsInWindow(100, 100);
        check("window [100,100] is inclusive (both same-ms events)", w.size() == 2);
    }

    // Full window covers everything, ascending.
    {
        auto w = p.getEventsInWindow(0, 1000);
        bool ordered = true;
        for (size_t i = 1; i < w.size(); ++i)
            if (w[i].timestamp_ms < w[i-1].timestamp_ms) ordered = false;
        check("full window returns all 6 events in ascending time order",
              w.size() == 6 && ordered);
    }

    // Empty window outside the range.
    check("window above all events is empty", p.getEventsInWindow(600, 700).empty());

    // clear() resets the timeline.
    p.clear();
    check("clear() empties the timeline", p.getEventsInWindow(0, 1000).empty());

    p.setEnabled(false);
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
