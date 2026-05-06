#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  bool csv = (argc > 1 && std::string(argv[1]) == "--csv");
  auto r = vgre::core::RuntimeEngine::instance().initialize();
  if (r != vgre::VGREResult::SUCCESS) {
    std::cerr << "RuntimeEngine initialization failed\n";
    return 1;
  }

  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  constexpr size_t kBytes = 64 * 1024 * 1024;
  constexpr size_t kPage = 4096;
  vgre::MemoryHandle handle = nullptr;

  r = mm.allocateManaged(kBytes, handle, 0, 2);
  if (r != vgre::VGREResult::SUCCESS || !handle) {
    std::cerr << "allocateManaged failed\n";
    vgre::core::RuntimeEngine::instance().shutdown();
    return 2;
  }

  auto *ptr = static_cast<uint8_t*>(handle);
  auto tTouch0 = Clock::now();
  for (size_t i = 0; i < kBytes; i += kPage) {
    ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }
  auto tTouch1 = Clock::now();

  auto tPrefetch0 = Clock::now();
  mm.memPrefetchAsync(ptr, kBytes, 0);
  auto tPrefetch1 = Clock::now();

  auto tAdv0 = Clock::now();
  mm.memAdvise(ptr, kBytes, 3, 0);
  mm.memAdvise(ptr, kBytes, 4, 0);
  auto tAdv1 = Clock::now();

  mm.free(handle);
  vgre::core::RuntimeEngine::instance().shutdown();

  double touchMs = std::chrono::duration<double, std::milli>(tTouch1 - tTouch0).count();
  double prefetchMs = std::chrono::duration<double, std::milli>(tPrefetch1 - tPrefetch0).count();
  double adviseMs = std::chrono::duration<double, std::milli>(tAdv1 - tAdv0).count();

  if (csv) {
    std::cout << "metric,value\n";
    std::cout << "bytes," << kBytes << "\n";
    std::cout << "touch_ms," << touchMs << "\n";
    std::cout << "prefetch_ms," << prefetchMs << "\n";
    std::cout << "advise_ms," << adviseMs << "\n";
    return 0;
  }

  std::cout << "[BENCH] ManagedMemoryHints\n";
  std::cout << "  bytes=" << kBytes << "\n";
  std::cout << "  touch_ms=" << touchMs
            << " prefetch_ms=" << prefetchMs
            << " advise_ms=" << adviseMs << "\n";
  return 0;
}
