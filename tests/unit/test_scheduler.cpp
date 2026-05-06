/**
 * VGRE Unit Tests — Scheduler
 * Tests the coarse-grained StreamTask scheduling model.
 */
#include "vgre/core/scheduler.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

using namespace vgre;
using namespace vgre::core;

void test_basic_stream_task() {
  Scheduler sched(4);
  std::atomic<int> counter{0};

  auto future = sched.submitStreamTask(1, [&counter]() { counter++; });

  auto result = future.get();
  (void)result;
  assert(result == VGREResult::SUCCESS);
  assert(counter == 1);

  std::cout << "[PASS] Basic stream task" << std::endl;
}

void test_stream_serialization() {
  Scheduler sched(4);
  std::vector<int> order;
  std::mutex orderMutex;

  // Submit 5 tasks to the SAME stream — they should execute in order
  std::vector<std::future<VGREResult>> futures;
  for (int i = 0; i < 5; ++i) {
    futures.push_back(sched.submitStreamTask(1, [i, &order, &orderMutex]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      std::lock_guard<std::mutex> lock(orderMutex);
      order.push_back(i);
    }));
  }

  // Wait for all tasks
  for (auto &f : futures) {
    f.get();
  }

  // Verify sequential ordering
  assert(order.size() == 5);
  for (int i = 0; i < 5; ++i) {
    assert(order[i] == i);
  }

  std::cout << "[PASS] Stream serialization (5 tasks in order)" << std::endl;
}

void test_cross_stream_parallelism() {
  Scheduler sched(4);
  std::atomic<int> stream1Done{0};
  std::atomic<int> stream2Done{0};

  // Submit a slow task to stream 1
  auto f1 = sched.submitStreamTask(1, [&stream1Done]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stream1Done = 1;
  });

  // Submit a slow task to stream 2
  auto f2 = sched.submitStreamTask(2, [&stream2Done]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stream2Done = 1;
  });

  auto start = std::chrono::high_resolution_clock::now();
  f1.get();
  f2.get();
  auto end = std::chrono::high_resolution_clock::now();

  auto durationMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  assert(stream1Done == 1);
  assert(stream2Done == 1);

  // If they were serialized, it would take ~100ms. If parallel, ~50ms.
  // We allow some margin for scheduling overhead.
  assert(durationMs < 90);

  std::cout << "[PASS] Cross-stream parallelism (" << durationMs
            << "ms, expected ~50ms)" << std::endl;
}

void test_wait_stream() {
  Scheduler sched(4);
  std::atomic<int> counter{0};

  sched.submitStreamTask(42, [&counter]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    counter = 1;
  });

  sched.waitStream(42);
  assert(counter == 1);

  std::cout << "[PASS] waitStream" << std::endl;
}

void test_thread_count() {
  Scheduler sched(4);
  assert(sched.getThreadCount() == 4);

  std::cout << "[PASS] Thread count" << std::endl;
}

void test_statistics() {
  Scheduler sched(2);

  auto f = sched.submitStreamTask(0, []() {});
  f.get();
  sched.waitAll();

  // The promise is fulfilled slightly before completed_ is incremented in the worker.
  // Wait briefly to allow the worker thread to finish the loop iteration.
  while (sched.getCompletedTasks() < 1) {
    std::this_thread::yield();
  }

  assert(sched.getCompletedTasks() >= 1);

  std::cout << "[PASS] Statistics: completed=" << sched.getCompletedTasks()
            << ", pending=" << sched.getPendingTasks() << std::endl;
}

void test_resize_thread_pool() {
  Scheduler sched(2);
  sched.setThreadCount(6);
  assert(sched.getThreadCount() == 6);

  std::atomic<int> done{0};
  auto f = sched.submitConcurrentTask([&done]() { done.fetch_add(1); });
  assert(f.get() == VGREResult::SUCCESS);
  sched.waitAll();
  assert(done.load() == 1);

  std::cout << "[PASS] Resize thread pool" << std::endl;
}

void test_invalid_numa_fallback() {
  Scheduler sched(4);
  std::atomic<int> ran{0};
  auto f = sched.submitNumaTask(7, [&ran]() { ran.fetch_add(1); }, 9999, 1);
  assert(f.get() == VGREResult::SUCCESS);
  sched.waitAll();
  assert(ran.load() == 1);

  std::cout << "[PASS] Invalid NUMA node fallback" << std::endl;
}

int main() {
  std::cout << "=== VGRE Scheduler Unit Tests ===" << std::endl;

  test_basic_stream_task();
  test_stream_serialization();
  test_cross_stream_parallelism();
  test_wait_stream();
  test_thread_count();
  test_statistics();
  test_resize_thread_pool();
  test_invalid_numa_fallback();

  std::cout << "\nAll scheduler tests passed!" << std::endl;
  return 0;
}
