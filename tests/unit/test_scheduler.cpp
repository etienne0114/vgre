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

// Regression test for the SPSC liveness bug: items pushed via enqueueToStream
// must be executed even when all worker threads are idle (sleeping in cv_.wait).
// The old implementation had a predicate that did not check SPSC rings, causing
// notify_one() to wake a worker whose predicate returned false → back to sleep →
// items permanently stranded. This test would time-out (waitAll 5s) on old code.
void test_enqueue_to_stream_liveness() {
  constexpr int kThreads = 4;
  constexpr int kItems   = 200;

  Scheduler sched(kThreads);

  // Allow all workers to reach cv_.wait (idle) before we push SPSC work.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  std::atomic<int> count{0};
  for (int i = 0; i < kItems; ++i) {
    WorkItem item;
    item.streamId = static_cast<uint64_t>(i % 8);
    item.execute  = [&count]() { count.fetch_add(1, std::memory_order_relaxed); };
    sched.enqueueToStream(item.streamId, std::move(item));
  }

  sched.waitAll();
  assert(count.load() == kItems);

  std::cout << "[PASS] enqueueToStream liveness (" << kItems << " items)" << std::endl;
}

// Verify stream→worker pinning: all items pushed to stream S land on worker
// (S % numThreads). We confirm they execute and the count is exact.
void test_enqueue_to_stream_pinning() {
  constexpr int kThreads = 4;
  Scheduler sched(kThreads);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  std::atomic<int> count{0};
  constexpr int kPerStream = 50;
  for (uint64_t s = 0; s < 8; ++s) {
    for (int j = 0; j < kPerStream; ++j) {
      WorkItem item;
      item.streamId = s;
      item.execute  = [&count]() { count.fetch_add(1, std::memory_order_relaxed); };
      sched.enqueueToStream(s, std::move(item));
    }
  }

  sched.waitAll();
  assert(count.load() == 8 * kPerStream);

  std::cout << "[PASS] enqueueToStream pinning (8 streams × " << kPerStream << " items)" << std::endl;
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
  test_enqueue_to_stream_liveness();
  test_enqueue_to_stream_pinning();

  std::cout << "\nAll scheduler tests passed!" << std::endl;
  return 0;
}
