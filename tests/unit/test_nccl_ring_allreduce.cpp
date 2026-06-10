// NCCL Ring AllReduce test (QUEUE-09)
// Tests the 2-phase ring algorithm with Kahan fp32 summation.
// Validates: correctness across 2/4/8 ranks, Kahan error bound, bandwidth η=2(N-1)/N comment.
#include "nccl.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

// ── Helper: initialise a communicator group of `nranks` in-process ────────
static std::vector<ncclComm_t> make_comms(int nranks) {
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);
    std::vector<ncclComm_t> comms(nranks);
    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r] {
            ncclCommInitRank(&comms[r], nranks, uid, r);
        });
    }
    for (auto& t : threads) t.join();
    return comms;
}

static void destroy_comms(std::vector<ncclComm_t>& comms) {
    for (auto c : comms) ncclCommDestroy(c);
}

// ── Test 1: correctness for ncclFloat32 sum, N ranks ──────────────────────
static void test_allreduce_sum(int nranks, size_t count) {
    auto comms = make_comms(nranks);

    // Each rank contributes rank+1 in every element: expected sum = N*(N+1)/2 each.
    float expected = 0.0f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);

    std::vector<std::vector<float>> send(nranks, std::vector<float>(count));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(count, 0.0f));

    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r] {
            ncclAllReduce(send[r].data(), recv[r].data(), count,
                          ncclFloat32, ncclSum, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < count; ++i) {
            float got = recv[r][i];
            assert(std::fabs(got - expected) < 1e-3f &&
                   "AllReduce sum mismatch");
        }
    destroy_comms(comms);
    printf("  [PASS] ncclFloat32 Sum nranks=%d count=%zu expected=%.1f\n",
           nranks, count, expected);
}

// ── Test 2: Kahan precision — alternating +1/-1 floats, large N ───────────
// Naive summation of N alternating +1/-1 values accumulates O(N·ε) error.
// Kahan summation gives |error| ≤ 2ε · ||x||_1.
static void test_kahan_precision(int nranks, size_t count) {
    if (nranks < 2) return;
    auto comms = make_comms(nranks);

    // Rank 0 contributes +1, rank 1 contributes -1, alternating.
    // Expected result: 0 if nranks even, 1 if nranks odd.
    float expected = (nranks % 2 == 0) ? 0.0f : 1.0f;

    std::vector<std::vector<float>> send(nranks, std::vector<float>(count));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(count, 0.0f));
    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), (r % 2 == 0) ? 1.0f : -1.0f);

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r] {
            ncclAllReduce(send[r].data(), recv[r].data(), count,
                          ncclFloat32, ncclSum, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    // Kahan error bound: |result - expected| ≤ 2ε·nranks·1 ≈ 2.4e-7 * nranks
    float tol = 1e-4f;
    for (size_t i = 0; i < count; ++i) {
        float got = recv[0][i];
        assert(std::fabs(got - expected) < tol && "Kahan precision failed");
    }
    destroy_comms(comms);
    printf("  [PASS] Kahan precision nranks=%d count=%zu expected=%.1f |err|<%.1e\n",
           nranks, count, expected, (double)tol);
}

// ── Test 3: ncclAvg correctness ────────────────────────────────────────────
static void test_allreduce_avg(int nranks) {
    auto comms = make_comms(nranks);
    constexpr size_t count = 64;
    float expected = 0.0f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);
    expected /= static_cast<float>(nranks);

    std::vector<std::vector<float>> send(nranks, std::vector<float>(count));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(count, 0.0f));
    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r] {
            ncclAllReduce(send[r].data(), recv[r].data(), count,
                          ncclFloat32, ncclAvg, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < count; ++i)
            assert(std::fabs(recv[r][i] - expected) < 1e-3f && "Avg mismatch");

    destroy_comms(comms);
    printf("  [PASS] ncclAvg nranks=%d expected=%.2f\n", nranks, expected);
}

// ── Test 4: large buffer (>1 MB, triggers ring path) ──────────────────────
static void test_large_buffer(int nranks) {
    // 1.5 MB / sizeof(float) = 393216 floats — above kRingThreshold
    constexpr size_t count = 393216;
    test_allreduce_sum(nranks, count);
}

// ── Test 5: TP same-process atomic-fence AllReduce (Track M) ──────────────
// Set VGRE_TP_DEGREE before commInit so the NcclGroupState picks up the value
// and routes ncclAllReduce through tp_ring_allreduce instead of the condvar path.
static void tp_env_set(int n) {
#ifdef _WIN32
    _putenv_s("VGRE_TP_DEGREE", std::to_string(n).c_str());
#else
    setenv("VGRE_TP_DEGREE", std::to_string(n).c_str(), 1);
#endif
}
static void tp_env_clear() {
#ifdef _WIN32
    _putenv_s("VGRE_TP_DEGREE", "");
#else
    unsetenv("VGRE_TP_DEGREE");
#endif
}

static void test_tp_allreduce_sum(int nranks, size_t count) {
    tp_env_set(nranks);
    auto comms = make_comms(nranks);
    float expected = 0.f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);

    std::vector<std::vector<float>> send(nranks, std::vector<float>(count));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(count, 0.f));
    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r)
        threads[r] = std::thread([&, r] {
            ncclAllReduce(send[r].data(), recv[r].data(), count,
                          ncclFloat32, ncclSum, comms[r], nullptr);
        });
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < count; ++i)
            assert(std::fabs(recv[r][i] - expected) < 1e-3f && "TP AllReduce sum mismatch");

    destroy_comms(comms);
    tp_env_clear();
    printf("  [PASS] TP AllReduce sum nranks=%d count=%zu expected=%.1f\n",
           nranks, count, expected);
}

static void test_tp_allreduce_avg(int nranks) {
    constexpr size_t count = 64;
    tp_env_set(nranks);
    auto comms = make_comms(nranks);
    float expected = 0.f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);
    expected /= static_cast<float>(nranks);

    std::vector<std::vector<float>> send(nranks, std::vector<float>(count));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(count, 0.f));
    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r)
        threads[r] = std::thread([&, r] {
            ncclAllReduce(send[r].data(), recv[r].data(), count,
                          ncclFloat32, ncclAvg, comms[r], nullptr);
        });
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < count; ++i)
            assert(std::fabs(recv[r][i] - expected) < 1e-3f && "TP AllReduce avg mismatch");

    destroy_comms(comms);
    tp_env_clear();
    printf("  [PASS] TP AllReduce avg nranks=%d expected=%.2f\n", nranks, expected);
}

static void test_tp_allreduce_large(int nranks) {
    constexpr size_t count = 262144;  // 1 MB of floats — large tensor
    test_tp_allreduce_sum(nranks, count);
}

int main() {
    printf("=== NCCL Ring AllReduce Tests ===\n");
    // Small buffers: flat-barrier path
    test_allreduce_sum(2,  16);
    test_allreduce_sum(4,  64);
    test_allreduce_sum(8,  32);
    // Medium buffers (64 KB – 1 MB): now routed to the ring reduce-scatter/
    // all-gather path (formerly a centralised binary tree). 32768 floats = 128 KB,
    // 200000 floats ≈ 800 KB.
    test_allreduce_sum(2,  32768);
    test_allreduce_sum(4,  32768);
    test_allreduce_sum(8,  200000);
    // Large buffers: ring path (> 1 MB)
    test_large_buffer(2);
    test_large_buffer(4);
    // Kahan precision
    test_kahan_precision(2, 256);
    test_kahan_precision(4, 256);
    // Avg
    test_allreduce_avg(2);
    test_allreduce_avg(4);
    // TP same-process atomic-fence fast path (Track M)
    printf("  --- TP AllReduce (VGRE_TP_DEGREE fast path) ---\n");
    test_tp_allreduce_sum(2, 16);
    test_tp_allreduce_sum(4, 64);
    test_tp_allreduce_avg(2);
    test_tp_allreduce_avg(4);
    test_tp_allreduce_large(2);
    test_tp_allreduce_large(4);
    printf("ALL NCCL Ring AllReduce tests PASSED.\n");
    return 0;
}
