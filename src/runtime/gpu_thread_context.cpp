#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/common/platform.h"
#include "vgre/core/cluster.h"
#include "vgre/core/tmem.h"
#include "vgre/compiler/wmma_emulation.h"   // tcgen05/wgmma shape helpers (MMA math)
#include <memory>
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

// ── Blackwell tcgen05 Tensor Memory (TMEM) runtime helpers (P3-7) ────────────
// One TMEM per worker thread (a CTA runs on one worker); lazily created.
static thread_local std::unique_ptr<vgre::tmem::TensorMemory> t_tmem;

static vgre::tmem::TensorMemory& tmem() {
    if (!t_tmem) t_tmem = std::make_unique<vgre::tmem::TensorMemory>();
    return *t_tmem;
}

extern "C" VGRE_PUBLIC_API
uint32_t vgre_jit_tmem_alloc(int nCols) { return tmem().alloc(nCols); }

extern "C" VGRE_PUBLIC_API
void vgre_jit_tmem_dealloc(uint32_t addr, int nCols) { tmem().dealloc(addr, nCols); }

extern "C" VGRE_PUBLIC_API
void vgre_jit_tmem_relinquish() { /* relinquish_alloc_permit: no-op on CPU */ }

extern "C" VGRE_PUBLIC_API
void vgre_jit_tcgen05_ld(uint32_t* dst, uint32_t addr, int lanes, int wordsPerLane) {
    tmem().ld(dst, addr, lanes, wordsPerLane);
}

extern "C" VGRE_PUBLIC_API
void vgre_jit_tcgen05_st(uint32_t addr, const uint32_t* src, int lanes, int wordsPerLane) {
    tmem().st(addr, src, lanes, wordsPerLane);
}

extern "C" VGRE_PUBLIC_API
void vgre_jit_tcgen05_cp(uint32_t addr, const void* smem, int lanes, int wordsPerLane) {
    tmem().cp_from_smem(addr, smem, lanes, wordsPerLane);
}

// MMA into a TMEM accumulator: compute D = A·B with the fixed-shape tensor-core
// helper into a temp, then store (or accumulate, for the K-loop) into TMEM at
// `addr`. kind: 0=bf16 1=f16 2=tf32 3=e4m3 4=e5m2 5=e4m3e5m2.
extern "C" VGRE_PUBLIC_API
void vgre_jit_tcgen05_mma(uint32_t addr, uint64_t descA, uint64_t descB,
                          int M, int N, int K, int kind, int accumulate) {
    std::vector<float> temp(static_cast<size_t>(M) * N, 0.0f);
    float* d = temp.data();
    bool handled = true;
    if (kind == 0 && M == 64 && K == 16 && N == 256)      vgre_wgmma_m64n256k16_bf16_f32(d, descA, descB);
    else if (kind == 0 && M == 64 && K == 16 && N == 128) vgre_wgmma_m64n128k16_bf16_f32(d, descA, descB);
    else if (kind == 0 && M == 64 && K == 16 && N == 64)  vgre_wgmma_m64n64k16_bf16_f32(d, descA, descB);
    else if (kind == 0 && M == 128 && K == 16 && N == 256) vgre_tcgen05_m128n256k16_bf16_f32(d, descA, descB);
    else if (kind == 1 && M == 64 && K == 16 && N == 256) vgre_wgmma_m64n256k16_f16_f32(d, descA, descB);
    else if (kind == 1 && M == 64 && K == 16 && N == 128) vgre_wgmma_m64n128k16_f16_f32(d, descA, descB);
    else if (kind == 2 && M == 64 && K == 8  && N == 256) vgre_wgmma_m64n256k8_tf32_f32(d, descA, descB);
    else if (kind == 3 && M == 64 && K == 32 && N == 256) vgre_tcgen05_m64n256k32_e4m3_f32(d, descA, descB);
    else if (kind == 3 && M == 64 && K == 32 && N == 128) vgre_tcgen05_m64n128k32_e4m3_f32(d, descA, descB);
    else if (kind == 3 && M == 64 && K == 32 && N == 64)  vgre_tcgen05_m64n64k32_e4m3_f32(d, descA, descB);
    else if (kind == 4 && M == 64 && K == 32 && N == 256) vgre_tcgen05_m64n256k32_e5m2_f32(d, descA, descB);
    else if (kind == 4 && M == 64 && K == 32 && N == 128) vgre_tcgen05_m64n128k32_e5m2_f32(d, descA, descB);
    else if (kind == 5 && M == 64 && K == 32 && N == 256) vgre_tcgen05_m64n256k32_e4m3e5m2_f32(d, descA, descB);
    else if (kind == 5 && M == 64 && K == 32 && N == 128) vgre_tcgen05_m64n128k32_e4m3e5m2_f32(d, descA, descB);
    else if (kind == 3 && M == 128 && K == 32 && N == 256) vgre_tcgen05_m128n256k32_e4m3_f32(d, descA, descB);
    else handled = false;

    if (!handled) return;                       // unknown shape → leave TMEM unchanged
    if (accumulate) tmem().accumulate_tile(addr, d, M, N);
    else            tmem().store_tile(addr, d, M, N);
}

} // namespace runtime
} // namespace vgre
