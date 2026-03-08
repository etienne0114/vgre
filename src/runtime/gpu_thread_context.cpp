#include "vgre/runtime/gpu_thread_context.h"
#include <mutex>
#include <thread>
#include <unordered_map>

namespace vgre {
namespace runtime {

static std::mutex g_maskMutex;
static std::unordered_map<std::thread::id, uint32_t> g_activeMasks;

void GPUThreadContext::setWarpMask(uint32_t mask) {
  std::lock_guard<std::mutex> lock(g_maskMutex);
  g_activeMasks[std::this_thread::get_id()] = mask;
}

uint32_t GPUThreadContext::getWarpMask() {
  std::lock_guard<std::mutex> lock(g_maskMutex);
  auto it = g_activeMasks.find(std::this_thread::get_id());
  return (it != g_activeMasks.end()) ? it->second : 0xFFFFFFFF;
}

void GPUThreadContext::clearWarpMask() {
  std::lock_guard<std::mutex> lock(g_maskMutex);
  g_activeMasks.erase(std::this_thread::get_id());
}

} // namespace runtime
} // namespace vgre
