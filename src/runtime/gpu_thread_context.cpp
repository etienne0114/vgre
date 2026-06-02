#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/common/platform.h"
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "vgre/runtime/block_worker_pool.h"

namespace vgre {
namespace runtime {

static thread_local uint32_t t_activeMask = 0xFFFFFFFF;
static thread_local BlockBarrier *t_blockBarrier = nullptr;

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

} // namespace runtime
} // namespace vgre
