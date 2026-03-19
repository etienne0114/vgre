#include "vgre/runtime/gpu_thread_context.h"
#include <mutex>
#include <thread>
#include <unordered_map>
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

void GPUThreadContext::clearBlockBarrier() {
  t_blockBarrier = nullptr;
}

void GPUThreadContext::blockBarrier() {
  if (t_blockBarrier) {
    t_blockBarrier->arrive_and_wait();
  }
}

extern "C" __attribute__((visibility("default")))
void vgre_jit_set_block_barrier(void *barrier) {
  GPUThreadContext::setBlockBarrier(reinterpret_cast<BlockBarrier *>(barrier));
}

extern "C" __attribute__((visibility("default")))
void vgre_jit_clear_block_barrier() {
  GPUThreadContext::clearBlockBarrier();
}


extern "C" __attribute__((visibility("default")))
void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg) {
    vgre::runtime::BlockWorkerPool::getInstance().dispatch(threadCount, [task, arg](int tid) {
        task(tid, arg);
    });
}

} // namespace runtime
} // namespace vgre
