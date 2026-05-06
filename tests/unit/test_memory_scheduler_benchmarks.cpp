#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/memory_manager.h"
#include "vgre/runtime/vector_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

static uint64_t percentileNs(std::vector<uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>(p * static_cast<double>(values.size() - 1));
  return values[idx];
}

static void benchmarkScheduler() {
  constexpr size_t kTasks = 20000;
  vgre::core::Scheduler sched(0);

  std::vector<uint64_t> submitNs(kTasks, 0);
  std::vector<uint64_t> execLatencyNs(kTasks, 0);
  std::atomic<size_t> completed{0};

  auto t0 = Clock::now();
  for (size_t i = 0; i < kTasks; ++i) {
    submitNs[i] = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
    sched.submitConcurrentTask([i, &submitNs, &execLatencyNs, &completed]() {
      uint64_t start = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now().time_since_epoch()).count());
      execLatencyNs[i] = start - submitNs[i];
      completed.fetch_add(1, std::memory_order_relaxed);
    });
  }
  sched.waitAll();
  auto t1 = Clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double throughput = static_cast<double>(kTasks) / (sec > 0.0 ? sec : 1e-9);
  uint64_t p50 = percentileNs(execLatencyNs, 0.50);
  uint64_t p95 = percentileNs(execLatencyNs, 0.95);
  uint64_t p99 = percentileNs(execLatencyNs, 0.99);

  std::cout << "[BENCH] Scheduler\n";
  std::cout << "  tasks=" << kTasks << " completed=" << completed.load() << "\n";
  std::cout << "  throughput_tasks_per_sec=" << static_cast<uint64_t>(throughput) << "\n";
  std::cout << "  latency_ns_p50=" << p50
            << " p95=" << p95
            << " p99=" << p99 << "\n";
}

static void benchmarkVectorDot() {
  constexpr size_t kN = 4 * 1024 * 1024;
  constexpr int kIters = 20;
  std::vector<float> a(kN, 1.25f);
  std::vector<float> b(kN, 0.75f);

  auto &ve = vgre::runtime::VectorEngine::instance();
  volatile float sink = 0.0f;
  auto t0 = Clock::now();
  for (int i = 0; i < kIters; ++i) {
    sink += ve.vectorDot(a.data(), b.data(), kN);
  }
  auto t1 = Clock::now();
  (void)sink;

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double gflops = (2.0 * static_cast<double>(kN) * static_cast<double>(kIters)) /
                  ((sec > 0.0 ? sec : 1e-9) * 1e9);
  std::cout << "[BENCH] VectorDot\n";
  std::cout << "  n=" << kN << " iters=" << kIters << "\n";
  std::cout << "  achieved_gflops=" << gflops << "\n";
}

static void benchmarkManagedHints() {
  auto r = vgre::core::RuntimeEngine::instance().initialize();
  if (r != vgre::VGREResult::SUCCESS) {
    std::cout << "[BENCH] ManagedMemoryHints skipped: RuntimeEngine init failed\n";
    return;
  }

  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  constexpr size_t kBytes = 64 * 1024 * 1024;
  vgre::MemoryHandle handle = nullptr;
  r = mm.allocateManaged(kBytes, handle, 0, 2);
  if (r != vgre::VGREResult::SUCCESS || !handle) {
    std::cout << "[BENCH] ManagedMemoryHints skipped: allocateManaged failed\n";
    vgre::core::RuntimeEngine::instance().shutdown();
    return;
  }

  auto *ptr = static_cast<uint8_t*>(handle);
  constexpr size_t kPage = 4096;

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

  double touchMs =
      std::chrono::duration<double, std::milli>(tTouch1 - tTouch0).count();
  double prefetchMs =
      std::chrono::duration<double, std::milli>(tPrefetch1 - tPrefetch0).count();
  double adviseMs =
      std::chrono::duration<double, std::milli>(tAdv1 - tAdv0).count();

  std::cout << "[BENCH] ManagedMemoryHints\n";
  std::cout << "  bytes=" << kBytes << "\n";
  std::cout << "  touch_ms=" << touchMs
            << " prefetch_ms=" << prefetchMs
            << " advise_ms=" << adviseMs << "\n";
}

int main() {
  std::cout << "=== VGRE Memory+Scheduler Microbenchmarks ===\n";
  benchmarkScheduler();
  benchmarkVectorDot();
  benchmarkManagedHints();
  return 0;
}
