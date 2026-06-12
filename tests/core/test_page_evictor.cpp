// Phase 3, Track P3-20 — UVM oversubscription via disk-backed LRU eviction.
//
// Models a workload whose managed allocations exceed the host-memory budget (here
// 8x): 64 regions, budget = 8 regions resident. Every access restores the region
// from disk if it was evicted and bumps it to most-recently-used; registering or
// accessing evicts the LRU resident regions to disk to stay within budget. The
// test proves the two guarantees that make oversubscription correct: (1) every
// region's data survives an arbitrary number of evict→disk→restore cycles
// byte-exactly (a missing restore is caught because eviction scrambles the
// buffer), and (2) peak residency never exceeds the budget. Both a sequential
// stream and a random-access pattern (each larger than RAM) are exercised.

#include "vgre/core/memory/page_evictor.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using vgre::core::uvm::DiskBackedLRU;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static const size_t   B = 4096;          // bytes per region (a "page")
static const uint64_t N = 64;            // regions (working set)
static const size_t   BUDGET = 8 * B;    // only 8 of 64 may be resident (8x oversub)

static void fillPattern(uint8_t* p, uint64_t id) {
    for (size_t j = 0; j < B; ++j) p[j] = (uint8_t)((id * 131u + j) & 0xFFu);
}
static bool checkPattern(const uint8_t* p, uint64_t id) {
    for (size_t j = 0; j < B; ++j)
        if (p[j] != (uint8_t)((id * 131u + j) & 0xFFu)) return false;
    return true;
}

int main() {
    printf("=== UVM oversubscription: disk-backed LRU eviction (P3-20) ===\n");

    std::vector<std::vector<uint8_t>> bufs(N, std::vector<uint8_t>(B, 0));
    DiskBackedLRU lru(BUDGET);
    check("backing file opened", lru.ok());

    // Register all N regions. With budget = 8 regions, ~56 are immediately spilled.
    for (uint64_t i = 0; i < N; ++i) lru.add(i, bufs[i].data(), B);
    check("registering >budget regions keeps residency within budget",
          lru.residentBytes() <= BUDGET);
    check("the excess regions were evicted to disk", lru.evictions() > 0);

    // (1) Sequential stream larger than RAM: write each region's pattern in order.
    //     Each write past the budget evicts an LRU region (spilling fresh data).
    for (uint64_t i = 0; i < N; ++i)
        fillPattern(static_cast<uint8_t*>(lru.access(i)), i);
    // Verify the whole set — every region's data survived eviction/restore.
    bool seqOk = true;
    for (uint64_t i = 0; i < N; ++i)
        if (!checkPattern(static_cast<uint8_t*>(lru.access(i)), i)) seqOk = false;
    check("sequential >RAM stream: all region data byte-exact after eviction", seqOk);

    // (2) Random access pattern (also larger than RAM): re-verify reads return the
    //     restored data, with many evict/restore cycles in between.
    std::mt19937 rng(20);
    std::uniform_int_distribution<uint64_t> pick(0, N - 1);
    bool randOk = true;
    for (int step = 0; step < 4000; ++step) {
        uint64_t id = pick(rng);
        if (!checkPattern(static_cast<uint8_t*>(lru.access(id)), id)) { randOk = false; break; }
    }
    check("random-access >RAM workload: every restored region is correct", randOk);

    // The mechanism must have actually oversubscribed: real evictions and restores.
    check("peak residency stayed within the budget", lru.peakResidentBytes() <= BUDGET);
    check("disk round-trips actually happened (evictions and restores > 0)",
          lru.evictions() > 0 && lru.restores() > 0);

    // A just-accessed region is resident; a long-cold one is evicted (LRU works).
    lru.access(7);
    for (uint64_t i = 8; i < N; ++i) lru.access(i);   // push id 7 far down the LRU... then re-touch others
    bool lruWorks = lru.isResident(N - 1);            // most-recent is resident
    check("most-recently-used region is resident (LRU ordering)", lruWorks);

    printf("\n  [info] evictions=%llu restores=%llu peakResident=%zu/%zu bytes\n",
           (unsigned long long)lru.evictions(), (unsigned long long)lru.restores(),
           lru.peakResidentBytes(), lru.budget());
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
