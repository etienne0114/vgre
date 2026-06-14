// Work-stealing thread pool — correctness under the conditions the engine needs:
// parallel coverage, nested (reentrant) parallelFor, many concurrent jobs, and a
// stress loop that would expose races/deadlocks.

#include "vgre/xla/thread_pool.h"

#include <atomic>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

using vgre::xla::ThreadPool;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== work-stealing thread pool ===\n");
    ThreadPool pool(4);
    check("concurrency = workers + caller", pool.concurrency() == 5);

    // 1) every index processed exactly once
    {
        const int64_t N = 100000;
        std::vector<int> hits(N, 0);
        pool.parallelFor(N, 256, [&](int64_t i) { hits[i] += 1; });
        bool ok = true;
        for (int64_t i = 0; i < N; ++i) ok &= (hits[i] == 1);
        check("each index visited exactly once", ok);
    }

    // 2) correct parallel sum (work actually distributes)
    {
        const int64_t N = 1 << 20;
        std::vector<int64_t> v(N);
        std::iota(v.begin(), v.end(), 1);
        std::atomic<int64_t> sum{0};
        pool.parallelFor(N, 4096, [&](int64_t i) { sum.fetch_add(v[i]); });
        check("parallel sum correct", sum.load() == N * (N + 1) / 2);
    }

    // 3) nested / reentrant parallelFor (a task spawns its own parallel loop)
    {
        const int64_t R = 64, C = 1000;
        std::atomic<int64_t> total{0};
        pool.parallelFor(R, 4, [&](int64_t) {
            pool.parallelFor(C, 64, [&](int64_t) { total.fetch_add(1); });
        });
        check("nested parallelFor (reentrant)", total.load() == R * C);
    }

    // 4) many threads each run independent jobs concurrently
    {
        const int NT = 8;
        std::atomic<int64_t> grand{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < NT; ++t) {
            ts.emplace_back([&] {
                for (int rep = 0; rep < 50; ++rep) {
                    std::atomic<int64_t> local{0};
                    pool.parallelFor(5000, 128, [&](int64_t) { local.fetch_add(1); });
                    grand.fetch_add(local.load());
                }
            });
        }
        for (auto& th : ts) th.join();
        check("concurrent independent jobs", grand.load() == (int64_t)NT * 50 * 5000);
    }

    // 5) tiny / edge counts and the serial path
    {
        std::atomic<int64_t> n{0};
        pool.parallelFor(0, 16, [&](int64_t) { n.fetch_add(1); });
        pool.parallelFor(1, 16, [&](int64_t) { n.fetch_add(1); });
        pool.parallelFor(3, 16, [&](int64_t) { n.fetch_add(1); });  // single chunk → serial
        check("edge counts (0,1,3)", n.load() == 0 + 1 + 3);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
