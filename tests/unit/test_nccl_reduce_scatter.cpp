// NCCL ReduceScatter test
// Covers: flat-barrier path (small data), ring path (>64KB), ncclAvg, multi-rank.
// Math: rank r receives Σ_{k=0}^{N-1} sendbuff_k[r*recvcount : (r+1)*recvcount].
#include "nccl.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <thread>

static std::vector<ncclComm_t> make_comms(int nranks) {
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);
    std::vector<ncclComm_t> comms(nranks);
    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r)
        threads[r] = std::thread([&, r]{ ncclCommInitRank(&comms[r], nranks, uid, r); });
    for (auto& t : threads) t.join();
    return comms;
}
static void destroy_comms(std::vector<ncclComm_t>& comms) {
    for (auto c : comms) ncclCommDestroy(c);
}

// Each rank r sends: sendbuff[r*recvcount + j] = r + 1 (flat value per rank).
// Expected recvbuff for every rank r: all recvcount elements = N*(N+1)/2.
static void test_reduce_scatter_sum(int nranks, size_t recvcount) {
    auto comms = make_comms(nranks);

    float expected = 0.f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);

    // sendbuff has nranks * recvcount elements; recvbuff has recvcount elements
    const size_t send_elems = static_cast<size_t>(nranks) * recvcount;
    std::vector<std::vector<float>> send(nranks, std::vector<float>(send_elems));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(recvcount, 0.f));

    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r]{
            ncclReduceScatter(send[r].data(), recv[r].data(), recvcount,
                              ncclFloat32, ncclSum, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < recvcount; ++i) {
            float got = recv[r][i];
            assert(std::fabs(got - expected) < 1e-3f && "ReduceScatter sum mismatch");
        }

    destroy_comms(comms);
    printf("  [PASS] ReduceScatter Sum  nranks=%d recvcount=%zu expected=%.1f\n",
           nranks, recvcount, expected);
}

// Test ncclAvg: recvbuff[i] = (N+1)/2 for all i.
static void test_reduce_scatter_avg(int nranks, size_t recvcount) {
    auto comms = make_comms(nranks);

    float expected = 0.f;
    for (int r = 0; r < nranks; ++r) expected += static_cast<float>(r + 1);
    expected /= static_cast<float>(nranks);

    const size_t send_elems = static_cast<size_t>(nranks) * recvcount;
    std::vector<std::vector<float>> send(nranks, std::vector<float>(send_elems));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(recvcount, 0.f));

    for (int r = 0; r < nranks; ++r)
        std::fill(send[r].begin(), send[r].end(), static_cast<float>(r + 1));

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r]{
            ncclReduceScatter(send[r].data(), recv[r].data(), recvcount,
                              ncclFloat32, ncclAvg, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < recvcount; ++i) {
            float got = recv[r][i];
            assert(std::fabs(got - expected) < 1e-3f && "ReduceScatter avg mismatch");
        }

    destroy_comms(comms);
    printf("  [PASS] ReduceScatter Avg  nranks=%d recvcount=%zu expected=%.2f\n",
           nranks, recvcount, expected);
}

// Test that each rank gets its own correct slice (not just the first chunk).
// sendbuff[r][i] = i (position-dependent).
// Expected: recv[rank][j] = sum_{k=0}^{N-1} send[k][rank*recvcount + j]
//                        = sum_{k=0}^{N-1} (rank*recvcount + j) = N*(rank*recvcount+j).
static void test_reduce_scatter_slice_identity(int nranks, size_t recvcount) {
    auto comms = make_comms(nranks);

    const size_t send_elems = static_cast<size_t>(nranks) * recvcount;
    std::vector<std::vector<float>> send(nranks, std::vector<float>(send_elems));
    std::vector<std::vector<float>> recv(nranks, std::vector<float>(recvcount, 0.f));

    // Every rank has same data: send[r][i] = (float)i
    for (int r = 0; r < nranks; ++r)
        for (size_t i = 0; i < send_elems; ++i)
            send[r][i] = static_cast<float>(i);

    std::vector<std::thread> threads(nranks);
    for (int r = 0; r < nranks; ++r) {
        threads[r] = std::thread([&, r]{
            ncclReduceScatter(send[r].data(), recv[r].data(), recvcount,
                              ncclFloat32, ncclSum, comms[r], nullptr);
        });
    }
    for (auto& t : threads) t.join();

    // recv[rank][j] = N * (rank*recvcount + j)
    for (int r = 0; r < nranks; ++r)
        for (size_t j = 0; j < recvcount; ++j) {
            float expected = static_cast<float>(nranks) *
                             static_cast<float>(static_cast<size_t>(r) * recvcount + j);
            float got = recv[r][j];
            float tol = std::max(1e-3f, std::fabs(expected) * 1e-5f);
            assert(std::fabs(got - expected) < tol && "ReduceScatter slice mismatch");
        }

    destroy_comms(comms);
    printf("  [PASS] ReduceScatter SliceIdentity nranks=%d recvcount=%zu\n",
           nranks, recvcount);
}

int main() {
    printf("=== NCCL ReduceScatter Tests ===\n");

    // ── Flat-barrier path (total ≤ 64 KB) ─────────────────────────────────────
    // For float32, nranks=2: threshold = 64KB/8B = 8192 elems → recvcount ≤ 8192
    test_reduce_scatter_sum(2, 16);
    test_reduce_scatter_sum(4, 16);
    test_reduce_scatter_sum(2, 64);
    test_reduce_scatter_avg(2, 32);
    test_reduce_scatter_avg(4, 32);
    test_reduce_scatter_slice_identity(2, 16);
    test_reduce_scatter_slice_identity(4, 16);

    // ── Ring path (total > 64 KB) ──────────────────────────────────────────────
    // nranks=2, float32: need recvcount > 8192 → use 16384
    // nranks=4, float32: need recvcount > 4096 → use 8192
    test_reduce_scatter_sum(2, 16384);
    test_reduce_scatter_sum(4, 8192);
    test_reduce_scatter_avg(2, 16384);
    test_reduce_scatter_avg(4, 8192);
    test_reduce_scatter_slice_identity(2, 16384);
    test_reduce_scatter_slice_identity(4, 8192);

    printf("ALL NCCL ReduceScatter tests PASSED.\n");
    return 0;
}
