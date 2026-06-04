// QUEUE-58: Treiber lock-free stack unit tests.
//
// Verifies:
//   1. Single-threaded LIFO order: push 5 items, pop all 5 in LIFO order.
//   2. Empty pop returns false; empty() agrees.
//   3. Concurrent correctness: 8 threads × 1000 push then 8 threads × 1000 pop,
//      verify zero loss and zero duplicate (total items conserved, each value
//      appears exactly once).

#include "vgre/core/treiber_stack.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

using vgre::core::TreiberStack;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::abort();
    }
}

// ─── Test 1: LIFO order ───────────────────────────────────────────────────────
static void test_lifo_order() {
    TreiberStack<int> s;

    check(s.empty(), "new stack should be empty");

    for (int i = 1; i <= 5; ++i) s.push(i);

    check(!s.empty(), "stack should not be empty after pushes");

    int v;
    for (int i = 5; i >= 1; --i) {
        bool ok = s.pop(v);
        check(ok, "pop should return true when stack is non-empty");
        check(v == i, "LIFO order violated");
    }

    check(s.empty(), "stack should be empty after popping all items");
}

// ─── Test 2: Empty pop ────────────────────────────────────────────────────────
static void test_empty_pop() {
    TreiberStack<int> s;
    int v = -1;
    bool ok = s.pop(v);
    check(!ok, "pop on empty stack should return false");
    check(v == -1, "pop on empty stack should not modify out parameter");
}

// ─── Test 3: Concurrent push/pop — zero loss, zero duplicate ──────────────────
static void test_concurrent() {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 1000;
    constexpr int kTotal = kThreads * kPerThread;

    TreiberStack<int> s;

    // Phase 1: 8 threads each push kPerThread unique values.
    // Thread t pushes values [t*kPerThread, (t+1)*kPerThread).
    {
        std::vector<std::thread> pushers;
        pushers.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            pushers.emplace_back([&s, t]() {
                const int base = t * kPerThread;
                for (int i = 0; i < kPerThread; ++i)
                    s.push(base + i);
            });
        }
        for (auto& th : pushers) th.join();
    }

    // Phase 2: 8 threads each pop until they have drained kPerThread items total
    // (no thread-local quota; we use a shared atomic counter).
    std::vector<int> collected;
    collected.reserve(kTotal);
    std::mutex collect_mu;
    std::atomic<int> popped{0};

    {
        std::vector<std::thread> poppers;
        poppers.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            poppers.emplace_back([&]() {
                std::vector<int> local;
                local.reserve(kPerThread);
                int v;
                while (true) {
                    int claimed = popped.fetch_add(1, std::memory_order_relaxed);
                    if (claimed >= kTotal) {
                        // Return the increment — we over-claimed.
                        popped.fetch_sub(1, std::memory_order_relaxed);
                        break;
                    }
                    // Spin until an item is available (another thread may not have
                    // pushed yet when this thread over-races, but here all pushes
                    // are done before poppers start, so pop() must succeed quickly).
                    while (!s.pop(v)) { /* busy wait — should be very short */ }
                    local.push_back(v);
                }
                std::lock_guard<std::mutex> lk(collect_mu);
                collected.insert(collected.end(), local.begin(), local.end());
            });
        }
        for (auto& th : poppers) th.join();
    }

    check(static_cast<int>(collected.size()) == kTotal,
          "total items collected != kTotal (zero-loss violated)");

    // Verify no duplicates: sort and check for consecutive equal values.
    std::sort(collected.begin(), collected.end());
    for (int i = 0; i < kTotal; ++i) {
        check(collected[i] == i, "missing or duplicated value (zero-duplicate violated)");
    }

    check(s.empty(), "stack should be empty after draining all items");
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    test_lifo_order();
    test_empty_pop();
    test_concurrent();

    std::fprintf(stdout, "PASS: all TreiberStack tests passed (QUEUE-58)\n");
    return 0;
}
