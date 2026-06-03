// QUEUE-33: Tests for cooperative_groups::partition_copy and grid_group::sync.
// Verifies two-phase grid-sync (phase-2 data visible to all blocks) and
// partition_copy correctness via JIT-compiled CUDA kernels.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vgre/core/runtime_engine.h"
#include "vgre/common/types.h"

using namespace vgre;
using namespace vgre::core;

#define CHECK(r) do { if ((r) != VGREResult::SUCCESS) { \
    std::fprintf(stderr, "VGRE error %d at line %d\n", (int)(r), __LINE__); return 1; \
} } while(0)

// Two-phase grid-sync kernel: Phase 1 writes tid+1, barrier, Phase 2 adds 1000.
// CPU-side verification after the kernel returns: data[i] == i + 1001.
// Math: barrier = counting semaphore over B blocks; O(1) per thread, O(B) total.
static const char* GRID_SYNC_KERNEL = R"(
extern "C" __global__ void grid_sync_two_phase(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Phase 1: each thread writes its global ID + 1
    if (tid < n) data[tid] = tid + 1;
    // Grid barrier: all Phase-1 writes visible before Phase 2.
    cooperative_groups::this_grid().sync();
    // Phase 2: each thread adds 1000 — verifiable on CPU side after return
    if (tid < n) data[tid] += 1000;
}
)";

// partition_copy kernel: partitions src[0..n) by threshold using cooperative_groups.
// Elements > threshold go to out_hi; others go to out_lo.
// Uses a plain struct predicate to avoid lambda captures in JIT kernel.
static const char* PARTITION_COPY_KERNEL = R"(
struct GtPred {
    int threshold;
    bool operator()(int x) const { return x > threshold; }
};
extern "C" __global__ void partition_copy_kernel(
    int* src, int n, int threshold, int* out_hi, int* out_lo,
    int* count_hi, int* count_lo)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cooperative_groups::coalesced_group g = cooperative_groups::coalesced_threads();
        int tmp_hi[64], tmp_lo[64];
        GtPred pred{threshold};
        // partition_copy: O(n) single pass; syncs group before/after
        int* end_hi = cooperative_groups::partition_copy(
            g, src, src + n, tmp_hi, tmp_lo, pred);
        *count_hi = (int)(end_hi - tmp_hi);
        *count_lo = n - *count_hi;
        for (int i = 0; i < *count_hi; ++i) out_hi[i] = tmp_hi[i];
        for (int i = 0; i < *count_lo; ++i) out_lo[i] = tmp_lo[i];
    }
}
)";

static int test_grid_sync_two_phase() {
    RuntimeEngine& eng = RuntimeEngine::instance();

    KernelId kid;
    CHECK(eng.registerKernel("grid_sync_two_phase", GRID_SYNC_KERNEL, kid));

    const int N = 64;
    MemoryHandle hdata;
    CHECK(eng.malloc(N * sizeof(int), hdata));

    int* data = reinterpret_cast<int*>(hdata);
    memset(data, 0, N * sizeof(int));

    // args follow CUDA convention: each element is a pointer to the actual arg value
    int n_val = N;
    void* args[] = { &data, &n_val };
    dim3 grid(4, 1, 1), block(16, 1, 1);
    CHECK(eng.launchCooperativeKernel(kid, grid, block, args, 0, 0));

    // CPU-side verification: Phase 1 wrote tid+1, Phase 2 added 1000 → data[i] = i+1001
    // Grid barrier guarantees Phase 1 writes from ALL blocks precede Phase 2 additions.
    for (int i = 0; i < N; ++i) {
        if (data[i] != i + 1001) {
            fprintf(stderr, "FAIL: data[%d]=%d expected %d\n", i, data[i], i + 1001);
            return 1;
        }
    }
    printf("PASS: two-phase grid-sync (4 blocks × 16 threads, all data[i]=i+1001)\n");
    return 0;
}

static int test_partition_copy() {
    RuntimeEngine& eng = RuntimeEngine::instance();

    KernelId kid;
    CHECK(eng.registerKernel("partition_copy_kernel", PARTITION_COPY_KERNEL, kid));

    // Input: 1..16, threshold=8 → hi={9..16}, lo={1..8}
    const int N = 16, thresh = 8;
    MemoryHandle hsrc, hhi, hlo, hcnthi, hcntlo;
    CHECK(eng.malloc(N * sizeof(int), hsrc));
    CHECK(eng.malloc(N * sizeof(int), hhi));
    CHECK(eng.malloc(N * sizeof(int), hlo));
    CHECK(eng.malloc(sizeof(int), hcnthi));
    CHECK(eng.malloc(sizeof(int), hcntlo));

    int* src    = reinterpret_cast<int*>(hsrc);
    int* hi     = reinterpret_cast<int*>(hhi);
    int* lo     = reinterpret_cast<int*>(hlo);
    int* cnthi  = reinterpret_cast<int*>(hcnthi);
    int* cntlo  = reinterpret_cast<int*>(hcntlo);

    for (int i = 0; i < N; ++i) src[i] = i + 1;
    memset(hi, 0, N * sizeof(int));
    memset(lo, 0, N * sizeof(int));
    cnthi[0] = cntlo[0] = 0;

    int n_val = N, t_val = thresh;
    void* args[] = { &src, &n_val, &t_val, &hi, &lo, &cnthi, &cntlo };
    dim3 grid(1, 1, 1), block(1, 1, 1);
    CHECK(eng.launchKernel(kid, grid, block, args, 0));
    // launchKernel is async; synchronize before reading results
    CHECK(eng.synchronize());

    // Math invariant: count_hi + count_lo == N (partition_copy preserves all elements)
    if (cnthi[0] + cntlo[0] != N) {
        fprintf(stderr, "FAIL: count_hi(%d)+count_lo(%d) != N(%d)\n",
                cnthi[0], cntlo[0], N);
        return 1;
    }
    if (cnthi[0] != 8 || cntlo[0] != 8) {
        fprintf(stderr, "FAIL: expected hi=8 lo=8, got hi=%d lo=%d\n",
                cnthi[0], cntlo[0]);
        return 1;
    }
    for (int i = 0; i < cnthi[0]; ++i) {
        if (hi[i] <= thresh) { fprintf(stderr,"FAIL: hi[%d]=%d <= thresh\n",i,hi[i]); return 1; }
    }
    for (int i = 0; i < cntlo[0]; ++i) {
        if (lo[i] > thresh) { fprintf(stderr,"FAIL: lo[%d]=%d > thresh\n",i,lo[i]); return 1; }
    }

    printf("PASS: partition_copy threshold=%d (hi=%d, lo=%d, invariant hi+lo==%d)\n",
           thresh, cnthi[0], cntlo[0], N);
    return 0;
}

int main() {
    RuntimeEngine& eng = RuntimeEngine::instance();
    if (eng.initialize() != VGREResult::SUCCESS) {
        fprintf(stderr, "RuntimeEngine init failed\n");
        return 1;
    }

    int failures = 0;
    failures += test_grid_sync_two_phase();
    failures += test_partition_copy();

    eng.shutdown();
    if (failures == 0)
        printf("All cooperative_groups QUEUE-33 tests passed.\n");
    return failures;
}
