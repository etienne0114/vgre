#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/common/platform.h"
#include "vgre/core/cluster.h"
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "vgre/runtime/block_worker_pool.h"

namespace vgre {
namespace runtime {

static thread_local uint32_t t_activeMask = 0xFFFFFFFF;
static thread_local BlockBarrier *t_blockBarrier = nullptr;

// Current CTA's thread-block cluster + its rank-in-cluster (P3-6). Installed by
// the executor's clustered path on each worker thread.
static thread_local vgre::cluster::Cluster* t_cluster = nullptr;
static thread_local unsigned t_cluster_rank = 0;

void GPUThreadContext::setWarpMask(uint32_t mask) {
  t_activeMask = mask;
}

uint32_t GPUThreadContext::getWarpMask() {
  return t_activeMask;
}

void GPUThreadContext::clearWarpMask() {
  t_activeMask = 0xFFFFFFFF;
}

void GPUThreadContext::setBlockBarrier(BlockBarrier *barrier) {
  t_blockBarrier = barrier;
}

BlockBarrier* GPUThreadContext::getBlockBarrier() {
  return t_blockBarrier;
}

void GPUThreadContext::clearBlockBarrier() {
  t_blockBarrier = nullptr;
}

void GPUThreadContext::blockBarrier() {
  if (t_blockBarrier) {
    t_blockBarrier->arrive_and_wait();
  }
}

extern "C" VGRE_PUBLIC_API
void vgre_jit_set_block_barrier(void *barrier) {
  GPUThreadContext::setBlockBarrier(reinterpret_cast<BlockBarrier *>(barrier));
}

extern "C" VGRE_PUBLIC_API
void vgre_jit_clear_block_barrier() {
  GPUThreadContext::clearBlockBarrier();
}


extern "C" VGRE_PUBLIC_API
void vgre_jit_block_barrier_sync() {
  GPUThreadContext::blockBarrier();
}


extern "C" VGRE_PUBLIC_API
void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg) {
    if (threadCount <= 1) {
        task(0, arg);
        return;
    }

    // Utilize the persistent BlockWorkerPool to handle block-level concurrency
    // without the overhead of spawning/joining OS threads for every launch.
    BlockWorkerPool::instance().dispatch(threadCount, task, arg);
}

// ── Thread-block cluster runtime helpers (P3-6) ──────────────────────────────
extern "C" VGRE_PUBLIC_API
void vgre_jit_set_cluster(void* cluster, unsigned rank) {
    t_cluster = reinterpret_cast<vgre::cluster::Cluster*>(cluster);
    t_cluster_rank = rank;
}

extern "C" VGRE_PUBLIC_API
void vgre_jit_clear_cluster() {
    t_cluster = nullptr;
    t_cluster_rank = 0;
}

// barrier.cluster (cluster.sync): rendezvous all CTAs in the cluster.
extern "C" VGRE_PUBLIC_API
void vgre_jit_cluster_sync() {
    if (t_cluster) t_cluster->sync();
}

// mapa.shared::cluster: retarget a shared-memory address to peer CTA `dstRank`.
// Without a cluster, DSMEM degenerates to this CTA's own shared memory.
extern "C" VGRE_PUBLIC_API
void* vgre_jit_mapa_shared_cluster(void* addr, unsigned dstRank) {
    if (!t_cluster) return addr;
    return t_cluster->map_shared_rank(t_cluster_rank, addr, dstRank);
}

// %cluster_ctarank / %cluster_nctarank special registers.
extern "C" VGRE_PUBLIC_API
unsigned vgre_jit_cluster_ctarank() { return t_cluster_rank; }

extern "C" VGRE_PUBLIC_API
unsigned vgre_jit_cluster_nctarank() { return t_cluster ? t_cluster->size() : 1u; }

} // namespace runtime
} // namespace vgre
