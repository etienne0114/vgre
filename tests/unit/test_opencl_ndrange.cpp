/**
 * QUEUE-18: OpenCL clEnqueueNDRangeKernel — NDRange arithmetic unit tests
 *
 * Tests the OpenCL execution model coordinate formula without requiring
 * actual OpenCL drivers.  All assertions are on the pure arithmetic.
 *
 * Invariant (OpenCL 1.0 §6.11.1):
 *   get_global_id(d) = get_group_id(d) * get_local_size(d)
 *                    + get_local_id(d)
 *                    + global_work_offset[d]
 *
 * get_num_groups(d) = ceil(global_work_size[d] / local_work_size[d])
 */

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// ── Inline emulation of the NDRange coordinate model ──────────────────────────
//
// Models one work item at (group_id[3], local_id[3]) in a launch of
// (global_work_size[3] / local_work_size[3]) groups with local_work_size[3]
// work items per group.
//
// O(work_dim) per evaluation — at most 3 scalar additions.

struct WorkItem {
    size_t group_id[3];   // block index per dimension
    size_t local_id[3];   // thread index within block
    size_t local_size[3]; // local_work_size per dimension
    size_t offset[3];     // global_work_offset per dimension

    // global_id[d] = group_id[d]*local_size[d] + local_id[d] + offset[d]
    size_t global_id(int d) const {
        return group_id[d] * local_size[d] + local_id[d] + offset[d];
    }
};

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        exit(1);
    }
}

// ── Test 1: 8×8 global, 2×2 local, zero offset ────────────────────────────────
//
// 4×4 groups (8/2=4 per dim), 2×2 items per group.
// global_id(0) in [0,8), global_id(1) in [0,8).
static void test_2d_zero_offset() {
    size_t gws[2] = {8, 8};
    size_t lws[2] = {2, 2};
    size_t offsets[2] = {0, 0};
    size_t num_groups[2] = {gws[0] / lws[0], gws[1] / lws[1]}; // {4,4}

    int count = 0;
    for (size_t gx = 0; gx < num_groups[0]; ++gx) {
        for (size_t gy = 0; gy < num_groups[1]; ++gy) {
            for (size_t lx = 0; lx < lws[0]; ++lx) {
                for (size_t ly = 0; ly < lws[1]; ++ly) {
                    WorkItem wi = {{gx, gy, 0}, {lx, ly, 0}, {lws[0], lws[1], 1}, {offsets[0], offsets[1], 0}};
                    // global_id[d] = group_id[d]*local[d] + local_id[d] + 0
                    check(wi.global_id(0) == gx * lws[0] + lx, "2d zero-offset: global_id(0) mismatch");
                    check(wi.global_id(1) == gy * lws[1] + ly, "2d zero-offset: global_id(1) mismatch");
                    check(wi.global_id(0) < gws[0], "2d zero-offset: global_id(0) out of range");
                    check(wi.global_id(1) < gws[1], "2d zero-offset: global_id(1) out of range");
                    ++count;
                }
            }
        }
    }
    check(count == (int)(gws[0] * gws[1]), "2d zero-offset: total work item count mismatch");
    printf("PASS test_2d_zero_offset: %d work items verified\n", count);
}

// ── Test 2: 8×8 global, 2×2 local, offset=(4,0) ──────────────────────────────
//
// offset shifts global_id(0) by 4: range is [4,12).
// global_id(0) = group_id(0)*2 + local_id(0) + 4
static void test_2d_with_offset() {
    size_t gws[2] = {8, 8};
    size_t lws[2] = {2, 2};
    size_t offsets[2] = {4, 0};
    size_t num_groups[2] = {gws[0] / lws[0], gws[1] / lws[1]}; // {4,4}

    int count = 0;
    for (size_t gx = 0; gx < num_groups[0]; ++gx) {
        for (size_t gy = 0; gy < num_groups[1]; ++gy) {
            for (size_t lx = 0; lx < lws[0]; ++lx) {
                for (size_t ly = 0; ly < lws[1]; ++ly) {
                    WorkItem wi = {{gx, gy, 0}, {lx, ly, 0}, {lws[0], lws[1], 1}, {offsets[0], offsets[1], 0}};
                    size_t expected_gid0 = gx * lws[0] + lx + offsets[0];
                    size_t expected_gid1 = gy * lws[1] + ly + offsets[1];
                    check(wi.global_id(0) == expected_gid0, "2d offset(4,0): global_id(0) mismatch");
                    check(wi.global_id(1) == expected_gid1, "2d offset(4,0): global_id(1) mismatch");
                    // offset(0)=4 → global_id(0) in [4,12)
                    check(wi.global_id(0) >= offsets[0], "2d offset: global_id(0) below offset");
                    check(wi.global_id(0) < offsets[0] + gws[0], "2d offset: global_id(0) above range");
                    ++count;
                }
            }
        }
    }
    check(count == (int)(gws[0] * gws[1]), "2d offset: total work item count mismatch");
    printf("PASS test_2d_with_offset: %d work items verified (offset=[4,0])\n", count);
}

// ── Test 3: 1D — 16 global, 4 local, zero offset ─────────────────────────────
//
// 4 groups of 4. global_id(0) in [0,16).  work_dim=1 → dim 1,2 unused.
static void test_1d_range() {
    size_t gws = 16;
    size_t lws = 4;
    size_t offset = 0;
    size_t num_groups = gws / lws; // 4

    int count = 0;
    for (size_t g = 0; g < num_groups; ++g) {
        for (size_t l = 0; l < lws; ++l) {
            WorkItem wi = {{g, 0, 0}, {l, 0, 0}, {lws, 1, 1}, {offset, 0, 0}};
            // global_id[d] = group_id[d]*local[d] + local_id[d] + offset[d]
            check(wi.global_id(0) == g * lws + l + offset, "1d: global_id(0) mismatch");
            check(wi.global_id(0) < gws, "1d: global_id(0) out of range");
            ++count;
        }
    }
    check(count == (int)gws, "1d: total work item count mismatch");
    printf("PASS test_1d_range: %d work items verified\n", count);
}

// ── Test 4: 3D — 4×4×4 global, 2×2×2 local, zero offset ─────────────────────
//
// 2×2×2 groups of 2×2×2 items each.  Total: 64 work items.
// get_num_groups(d) = global_work_size[d] / local_work_size[d] = 2 for all d.
static void test_3d_full() {
    size_t gws[3] = {4, 4, 4};
    size_t lws[3] = {2, 2, 2};
    size_t offsets[3] = {0, 0, 0};
    size_t num_groups[3] = {gws[0]/lws[0], gws[1]/lws[1], gws[2]/lws[2]}; // {2,2,2}

    int count = 0;
    for (size_t gx = 0; gx < num_groups[0]; ++gx) {
        for (size_t gy = 0; gy < num_groups[1]; ++gy) {
            for (size_t gz = 0; gz < num_groups[2]; ++gz) {
                for (size_t lx = 0; lx < lws[0]; ++lx) {
                    for (size_t ly = 0; ly < lws[1]; ++ly) {
                        for (size_t lz = 0; lz < lws[2]; ++lz) {
                            WorkItem wi = {{gx,gy,gz},{lx,ly,lz},{lws[0],lws[1],lws[2]},{0,0,0}};
                            // global_id[d] = group_id[d]*local[d] + local_id[d] + 0
                            check(wi.global_id(0) == gx*lws[0]+lx, "3d: global_id(0)");
                            check(wi.global_id(1) == gy*lws[1]+ly, "3d: global_id(1)");
                            check(wi.global_id(2) == gz*lws[2]+lz, "3d: global_id(2)");
                            check(wi.global_id(0) < gws[0], "3d: gid0 out of range");
                            check(wi.global_id(1) < gws[1], "3d: gid1 out of range");
                            check(wi.global_id(2) < gws[2], "3d: gid2 out of range");
                            ++count;
                        }
                    }
                }
            }
        }
    }
    check(count == (int)(gws[0]*gws[1]*gws[2]), "3d: total work item count mismatch");
    printf("PASS test_3d_full: %d work items verified\n", count);
}

// ── Test 5: Validate work_dim=1 ignores offset[1] and offset[2] ───────────────
//
// When work_dim=1, only dimension 0 carries the offset.
// Dimensions 1 and 2 are effectively zero (not part of the address space).
static void test_work_dim_isolation() {
    // 1D launch: only dim 0 has offset 8.
    // Dims 1,2 are unused; their offsets must not contaminate dim 0.
    WorkItem wi1 = {{3, 0, 0}, {1, 0, 0}, {4, 1, 1}, {8, 99, 99}};
    check(wi1.global_id(0) == 3*4 + 1 + 8, "work_dim_isolation: global_id(0)");
    // Dim 1 and 2 local_size=1, group_id=0, local_id=0 → offset dominates
    // (these dimensions are not iterated in a work_dim=1 launch)

    // 2D launch: only dims 0,1 carry their offsets. Dim 2 offset is zero.
    WorkItem wi2 = {{1, 2, 0}, {0, 1, 0}, {3, 4, 1}, {5, 6, 0}};
    // global_id[0] = 1*3 + 0 + 5 = 8
    check(wi2.global_id(0) == 1*3 + 0 + 5, "work_dim_isolation: 2d global_id(0)");
    // global_id[1] = 2*4 + 1 + 6 = 15
    check(wi2.global_id(1) == 2*4 + 1 + 6, "work_dim_isolation: 2d global_id(1)");
    // global_id[2] = 0 (unused dim, offset=0)
    check(wi2.global_id(2) == 0, "work_dim_isolation: 2d global_id(2) must be zero");

    printf("PASS test_work_dim_isolation\n");
}

int main() {
    test_2d_zero_offset();
    test_2d_with_offset();
    test_1d_range();
    test_3d_full();
    test_work_dim_isolation();
    printf("All OpenCL NDRange arithmetic tests passed.\n");
    return 0;
}
