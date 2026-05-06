#include "vgre/core/scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static uint64_t percentileNs(std::vector<uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>(p * static_cast<double>(values.size() - 1));
  return values[idx];
}

int main(int argc, char** argv) {
  bool csv = (argc > 1 && std::string(argv[1]) == "--csv");
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
  uint64_t throughput = static_cast<uint64_t>(static_cast<double>(kTasks) / (sec > 0.0 ? sec : 1e-9));
  uint64_t p50 = percentileNs(execLatencyNs, 0.50);
  uint64_t p95 = percentileNs(execLatencyNs, 0.95);
  uint64_t p99 = percentileNs(execLatencyNs, 0.99);

  if (csv) {
    std::cout << "metric,value\n";
    std::cout << "tasks," << kTasks << "\n";
    std::cout << "completed," << completed.load() << "\n";
    std::cout << "throughput_tasks_per_sec," << throughput << "\n";
    std::cout << "latency_ns_p50," << p50 << "\n";
    std::cout << "latency_ns_p95," << p95 << "\n";
    std::cout << "latency_ns_p99," << p99 << "\n";
    return 0;
  }

  std::cout << "[BENCH] Scheduler\n";
  std::cout << "  tasks=" << kTasks << " completed=" << completed.load() << "\n";
  std::cout << "  throughput_tasks_per_sec=" << throughput << "\n";
  std::cout << "  latency_ns_p50=" << p50 << " p95=" << p95 << " p99=" << p99 << "\n";
  return 0;
}
