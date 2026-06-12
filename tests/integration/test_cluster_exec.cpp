// Phase 3, Track P3-6 — end-to-end cluster execution through the executor.
//
// Drives CPUParallelExecutor::executeClustered with a real CompiledKernelFn that
// performs a Distributed-Shared-Memory reduction: each CTA writes its value into
// its own shared memory, rendezvouses at the cluster barrier (vgre_jit_cluster_
// sync), then sums every peer CTA's value read through mapa.shared::cluster
// (vgre_jit_mapa_shared_cluster). This exercises the same runtime path a JIT'd
// Hopper cluster kernel uses: the executor installs the per-CTA cluster context,
// registers each CTA's shared buffer, and runs the CTAs concurrently. A wrong
// cluster context, a missing barrier, or wrong DSMEM addressing all yield a wrong
// reduction — so a correct result proves the mechanism, not just "it ran".

#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/types.h"

#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

using vgre::dim3;
using vgre::CompiledKernelFn;
using vgre::runtime::CPUParallelExecutor;

extern "C" {
  void     vgre_jit_cluster_sync();
  void*    vgre_jit_mapa_shared_cluster(void* addr, unsigned dstRank);
  unsigned vgre_jit_cluster_ctarank();
  unsigned vgre_jit_cluster_nctarank();
}

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// "Cluster kernel": out[globalBlockRank] = sum over peers p of (rank_p + 1)*10,
// where rank_p is the peer's rank-in-cluster — computed by reading each peer's
// shared memory via Distributed Shared Memory.
static CompiledKernelFn makeReductionKernel() {
    return std::make_shared<std::function<void(
        void**, const dim3*, const dim3*, const dim3*, const dim3*, void*, size_t)>>(
        [](void** args, const dim3* blockIdx, const dim3* /*tIdx*/,
           const dim3* /*blockDim*/, const dim3* gridDim, void* smem, size_t) {
            int* s = static_cast<int*>(smem);
            const unsigned rank = vgre_jit_cluster_ctarank();
            const unsigned n    = vgre_jit_cluster_nctarank();
            s[0] = static_cast<int>((rank + 1) * 10);     // publish my value
            vgre_jit_cluster_sync();                      // all CTAs published

            long sum = 0;
            for (unsigned p = 0; p < n; ++p) {
                int* peer = static_cast<int*>(vgre_jit_mapa_shared_cluster(&s[0], p));
                sum += *peer;                             // read peer p's DSMEM
            }
            vgre_jit_cluster_sync();                      // hold smem until all read

            const unsigned br =
                (blockIdx->z * gridDim->y + blockIdx->y) * gridDim->x + blockIdx->x;
            static_cast<int*>(args[0])[br] = static_cast<int>(sum);
        });
}

// Run a clustered launch and check every block's reduction equals the per-cluster
// expected sum (sum over ranks r in [0,clusterSize) of (r+1)*10).
static bool runClusterExec(dim3 grid, dim3 cluster) {
    const unsigned clusterSize = cluster.total();
    long expect = 0;
    for (unsigned r = 0; r < clusterSize; ++r) expect += (long)((r + 1) * 10);

    std::vector<int> out(grid.total(), -1);
    void* a0 = out.data();
    void* args[1] = { a0 };

    CompiledKernelFn fn = makeReductionKernel();
    CPUParallelExecutor exec;
    auto r = exec.executeClustered(fn, grid, dim3(1, 1, 1), cluster,
                                   args, /*sharedMem=*/64);
    if (r != vgre::VGREResult::SUCCESS) return false;

    for (uint32_t i = 0; i < grid.total(); ++i)
        if (out[i] != (int)expect) return false;
    return true;
}

int main() {
    printf("=== Clustered kernel execution via executeClustered (P3-6) ===\n");

    // 4-CTA cluster spanning a 4-block grid: every block sees 10+20+30+40 = 100.
    check("4-CTA cluster: DSMEM reduction == 100 across the grid",
          runClusterExec(dim3(4, 1, 1), dim3(4, 1, 1)));

    // 2-CTA clusters tiling an 8-block grid: 4 independent clusters, each
    // reducing 10+20 = 30 (proves the cluster is scoped, ranks are in-cluster).
    check("2-CTA clusters over 8 blocks: each cluster reduces to 30",
          runClusterExec(dim3(8, 1, 1), dim3(2, 1, 1)));

    // 2x2 cluster over a 4x4 grid: 4 clusters of 4 CTAs, each reduces to 100.
    check("2x2 cluster over 4x4 grid: each reduces to 100",
          runClusterExec(dim3(4, 4, 1), dim3(2, 2, 1)));

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
