// Warp shuffle emulation test (QUEUE-10)
// Validates all four __shfl*_sync variants across widths 32/16/8/4
// and types int32/float/int64/double using 32 cooperative threads.
//
// Stub architecture: thread_local storage backs the three JIT runtime
// functions so the test is self-contained and does not link against vgre.
// A C++17 mutex/condvar barrier provides the block-wide synchronisation that
// vgre_jit_block_barrier_sync() performs in a real kernel context.

#include "vgre/compiler/cpu_cuda_warp.h"

#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// ── C++17 reusable barrier ────────────────────────────────────────────────────
struct Barrier17 {
    explicit Barrier17(int n) : n_(n), arrived_(0), gen_(0) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mu_);
        int my_gen = gen_;
        if (++arrived_ == n_) {
            arrived_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen_ != my_gen; });
        }
    }
private:
    const int n_;
    int arrived_;
    int gen_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// ── JIT runtime stubs ────────────────────────────────────────────────────────

static uint64_t            g_warp_buf[32];
static void*               g_warp_ptr  = g_warp_buf;   // *pbuf  → g_warp_buf
static thread_local vgre::dim3 tl_tidx = {};            // per-thread threadIdx

static Barrier17* g_barrier = nullptr;

extern "C" {

void** vgre_jit_get_warp_buffer() {
    return &g_warp_ptr;
}

void vgre_jit_block_barrier_sync() {
    g_barrier->arrive_and_wait();
}

vgre::dim3* vgre_jit_get_threadIdx() {
    return &tl_tidx;
}

} // extern "C"

// ── Warp runner ──────────────────────────────────────────────────────────────
// Spawns 32 threads, assigns each a lane id, runs fn(lane) and collects
// the return value via an output array.

template<typename T, typename Fn>
static std::vector<T> run_warp(Fn fn) {
    std::vector<T> out(32);
    Barrier17 bar(32);
    g_barrier = &bar;
    memset(g_warp_buf, 0, sizeof(g_warp_buf));

    std::vector<std::thread> threads;
    threads.reserve(32);
    for (int l = 0; l < 32; ++l) {
        threads.emplace_back([&, l] {
            tl_tidx.x = static_cast<uint32_t>(l);
            out[l] = fn(l);
        });
    }
    for (auto& t : threads) t.join();
    g_barrier = nullptr;
    return out;
}

// ── Test helpers ─────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        assert(false);
    }
}

// ── Test 1: __shfl_sync — broadcast from a fixed lane ────────────────────────
// Invariant: all threads in a group read from group_id*W + (srcLane%W).
// With srcLane=0 and width=W, every group reads from its own group_id*W+0.
static void test_shfl_broadcast() {
    // width=32: lane 0 broadcasts its value (0) to all 31 others.
    auto out = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_sync(0xFFFFFFFF, l, 0, 32);
    });
    for (int l = 0; l < 32; ++l)
        check(out[l] == 0, "shfl_sync width=32 broadcast lane0");

    // width=8: 4 groups. Lane 0 of each group (i.e. 0,8,16,24) broadcasts.
    auto out8 = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_sync(0xFFFFFFFF, l, 0, 8);
    });
    for (int l = 0; l < 32; ++l) {
        int expected = (l / 8) * 8;  // group_id * 8 + 0
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_sync width=8 lane=%d", l);
        check(out8[l] == expected, msg);
    }

    // width=4: 8 groups. Each thread reads from lane 3 within its group.
    auto out4_src3 = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_sync(0xFFFFFFFF, l, 3, 4);
    });
    for (int l = 0; l < 32; ++l) {
        int expected = (l / 4) * 4 + 3;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_sync width=4 srcLane=3 lane=%d", l);
        check(out4_src3[l] == expected, msg);
    }

    printf("  [PASS] __shfl_sync (int, widths 4/8/32)\n");
}

// ── Test 2: __shfl_sync with float and double ─────────────────────────────────
static void test_shfl_float_double() {
    auto fout = run_warp<float>([](int l) {
        float v = static_cast<float>(l) * 0.5f;
        return vgre_cuda::__shfl_sync(0xFFFFFFFF, v, 7, 16);
    });
    for (int l = 0; l < 32; ++l) {
        float expected = static_cast<float>((l / 16) * 16 + 7) * 0.5f;
        check(fabsf(fout[l] - expected) < 1e-6f, "shfl_sync float width=16");
    }

    auto dout = run_warp<double>([](int l) {
        double v = static_cast<double>(l) * 1.5;
        return vgre_cuda::__shfl_sync(0xFFFFFFFF, v, 0, 32);
    });
    for (int l = 0; l < 32; ++l) {
        check(fabs(dout[l] - 0.0) < 1e-12, "shfl_sync double width=32 broadcast 0");
    }

    printf("  [PASS] __shfl_sync (float/double)\n");
}

// ── Test 3: __shfl_up_sync ────────────────────────────────────────────────────
// Invariant: lane l reads from lane max(l - delta, group_start).
static void test_shfl_up() {
    // width=32, delta=4: lane l reads from max(l-4, 0).
    auto out = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_up_sync(0xFFFFFFFF, l, 4, 32);
    });
    for (int l = 0; l < 32; ++l) {
        int expected = (l < 4) ? l : l - 4;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_up width=32 delta=4 lane=%d", l);
        check(out[l] == expected, msg);
    }

    // width=8, delta=2: clamp to group start.
    // Group i covers lanes [8i, 8i+7]. Lane l in group reads from max(l-2, 8i).
    auto out8 = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_up_sync(0xFFFFFFFF, l, 2, 8);
    });
    for (int l = 0; l < 32; ++l) {
        int group_start = (l / 8) * 8;
        int expected = (l - group_start < 2) ? l : l - 2;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_up width=8 delta=2 lane=%d", l);
        check(out8[l] == expected, msg);
    }

    printf("  [PASS] __shfl_up_sync (int, widths 8/32)\n");
}

// ── Test 4: __shfl_down_sync ──────────────────────────────────────────────────
// Invariant: lane l reads from min(l + delta, group_end - 1), own val if OOB.
static void test_shfl_down() {
    auto out = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_down_sync(0xFFFFFFFF, l, 4, 32);
    });
    for (int l = 0; l < 32; ++l) {
        int expected = (l + 4 >= 32) ? l : l + 4;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_down width=32 delta=4 lane=%d", l);
        check(out[l] == expected, msg);
    }

    // width=8, delta=3
    auto out8 = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_down_sync(0xFFFFFFFF, l, 3, 8);
    });
    for (int l = 0; l < 32; ++l) {
        int lane_in_grp = l % 8;
        int expected = (lane_in_grp + 3 >= 8) ? l : l + 3;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_down width=8 delta=3 lane=%d", l);
        check(out8[l] == expected, msg);
    }

    printf("  [PASS] __shfl_down_sync (int, widths 8/32)\n");
}

// ── Test 5: __shfl_xor_sync ───────────────────────────────────────────────────
// Invariant: lane l reads from group_id*W + ((lane_in_grp) XOR laneMask % W).
// laneMask=1 (bit-flip of bit 0): swaps adjacent pairs.
static void test_shfl_xor() {
    // Swap adjacent pairs within warp: lane l reads from l^1.
    auto out = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, l, 1, 32);
    });
    for (int l = 0; l < 32; ++l) {
        int expected = l ^ 1;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_xor width=32 laneMask=1 lane=%d", l);
        check(out[l] == expected, msg);
    }

    // Butterfly sum reduction: XOR with 1, 2, 4, 8, 16 → all lanes get total.
    // Starting with lane value l, after 5 butterfly steps sum = 0+1+...+31 = 496.
    auto sum_out = run_warp<int>([](int l) {
        int val = l;
        val += vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, val,  1, 32);
        val += vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, val,  2, 32);
        val += vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, val,  4, 32);
        val += vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, val,  8, 32);
        val += vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, val, 16, 32);
        return val;
    });
    for (int l = 0; l < 32; ++l)
        check(sum_out[l] == 496, "shfl_xor butterfly warp-sum == 496");

    // width=8, laneMask=4: within each 8-lane group, xor bit-2 of lane_in_grp.
    auto out8 = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_xor_sync(0xFFFFFFFF, l, 4, 8);
    });
    for (int l = 0; l < 32; ++l) {
        int group_id   = l / 8;
        int lane_in    = l % 8;
        int src_in_grp = (lane_in ^ 4) % 8;
        int expected   = group_id * 8 + src_in_grp;
        char msg[64];
        snprintf(msg, sizeof(msg), "shfl_xor width=8 laneMask=4 lane=%d", l);
        check(out8[l] == expected, msg);
    }

    printf("  [PASS] __shfl_xor_sync (int, widths 8/32, butterfly)\n");
}

// ── Test 6: partial mask — non-participating lanes return own value ────────────
// Mask 0x0000FFFF: only lanes 0-15 participate; lanes 16-31 must return own val.
static void test_partial_mask() {
    auto out = run_warp<int>([](int l) {
        return vgre_cuda::__shfl_sync(0x0000FFFFu, l, 0, 32);
    });
    for (int l = 0; l < 16; ++l)
        check(out[l] == 0, "partial mask: participating lane reads from 0");
    for (int l = 16; l < 32; ++l)
        check(out[l] == l, "partial mask: non-participating lane returns own value");

    printf("  [PASS] __shfl_sync partial mask (lanes 16-31 return own value)\n");
}

// ── Test 7: int64 / int32 identity (src == self) ──────────────────────────────
// __shfl_sync with srcLane == own lane → each thread gets its own value.
static void test_self_read() {
    // __shfl_up with delta=0: src = self for all lanes.
    auto out = run_warp<int64_t>([](int l) {
        int64_t v = static_cast<int64_t>(l) * 0x100000001LL;
        return vgre_cuda::__shfl_up_sync(0xFFFFFFFF, v, 0u, 32);
    });
    for (int l = 0; l < 32; ++l) {
        int64_t expected = static_cast<int64_t>(l) * 0x100000001LL;
        check(out[l] == expected, "shfl_up delta=0 int64 identity");
    }

    printf("  [PASS] __shfl_up_sync delta=0 int64 identity\n");
}

int main() {
    printf("=== Warp Shuffle Emulation Tests ===\n");
    test_shfl_broadcast();
    test_shfl_float_double();
    test_shfl_up();
    test_shfl_down();
    test_shfl_xor();
    test_partial_mask();
    test_self_read();
    printf("ALL warp shuffle tests PASSED.\n");
    return 0;
}
