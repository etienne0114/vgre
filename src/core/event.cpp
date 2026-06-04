#include "vgre/core/event.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/stream_dep_tracker.h"

namespace vgre {
namespace core {

Event::Event() : recorded_(false), flags_(kEventDefault) {}

// Constructor respecting CUDA event creation flags.
// kEventBlockingSync: synchronize() uses cv_.wait instead of future spin-poll.
// kEventDisableTiming: record() skips timestamp capture (faster, no allocation).
Event::Event(unsigned int flags) : recorded_(false), flags_(flags) {}

VGREResult Event::record(StreamId stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  // kEventDisableTiming: skip timestamp capture entirely — use a sentinel
  // future that resolves immediately with the epoch. This avoids the overhead
  // of steady_clock::now() on the hot path when timing is not needed.
  const bool timing = !(flags_ & kEventDisableTiming);

  auto sharedPromise = std::make_shared<std::promise<TimePoint>>();
  future_ = sharedPromise->get_future().share();

  // Capture cv_ by reference — it lives in this Event which outlives the task.
  // When kEventBlockingSync is set, notify the cv_ after setting the promise so
  // that synchronize()'s cv_.wait_for() wakes immediately.
  auto task = [this, sharedPromise, timing]() {
    try {
      TimePoint tp = timing ? std::chrono::steady_clock::now() : TimePoint{};
      sharedPromise->set_value(tp);
      cv_.notify_all();  // wake any blocking synchronize() calls
    } catch (...) {}
  };

  VGREResult res;
  int priority = 0;
  (void)vgre::core::RuntimeEngine::instance().getDevice().getStreamPriority(
      stream, priority);
  
  try {
    auto fut = vgre::core::Scheduler::instance().submitStreamTask(stream, task, priority);
    if (!fut.valid()) {
      task(); // Execute synchronously as fallback
    }
    res = VGREResult::SUCCESS;
  } catch (...) {
    res = VGREResult::ERR_LAUNCH_FAILURE;
  }

  if (res == VGREResult::SUCCESS) {
    recorded_ = true;
    // Snapshot the stream's completion counter so waitEventSatisfied() can
    // determine when all work enqueued before this record() has finished.
    // O(1): reads the leaf counter and stores it in the event snapshot map.
    StreamDepTracker::instance().recordEventCounter(this, stream);
  } else {
    recorded_ = false;
  }
  return res;
}

VGREResult Event::synchronize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!recorded_) return VGREResult::ERR_INVALID_VALUE;

  auto fut = future_;
  lock.unlock();
  if (!fut.valid()) return VGREResult::ERR_INVALID_VALUE;

  static const int kTimeoutMs = []() -> int {
      const char* e = vgre_get_config("VGRE_EVENT_TIMEOUT_MS");
      if (e) { try { int v = std::stoi(e); if (v >= 100 && v <= 60000) return v; } catch (...) {} }
      return 5000;
  }();

  if (flags_ & kEventBlockingSync) {
    // Blocking-sync: park the thread on a cv_ instead of spinning on the future.
    // The worker sets the promise → future becomes ready → cv_ is notified.
    // Math invariant: wait is O(1) wakeups (the promise is set exactly once).
    std::unique_lock<std::mutex> lk(mutex_);
    bool done = cv_.wait_for(lk, std::chrono::milliseconds(kTimeoutMs),
        [&]{ return fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
    return done ? VGREResult::SUCCESS : VGREResult::ERR_TIMEOUT;
  }

  // Default: spin on the future (low-latency for short waits)
  if (fut.wait_for(std::chrono::milliseconds(kTimeoutMs)) == std::future_status::timeout)
    return VGREResult::ERR_TIMEOUT;
  return VGREResult::SUCCESS;
}

VGREResult Event::elapsedTime(const Event &start, float &outMs) const {
  if (&start == this) {
    outMs = 0.0f;
    return VGREResult::SUCCESS;
  }

  // Synchronize both events to ensure they have timestamps
  if (start.synchronize() != VGREResult::SUCCESS ||
      this->synchronize() != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  std::shared_future<TimePoint> startFuture;
  std::shared_future<TimePoint> endFuture;
  {
    std::scoped_lock lock(start.mutex_, mutex_);
    startFuture = start.future_;
    endFuture = future_;
  }
  if (!startFuture.valid() || !endFuture.valid()) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Both are valid and resolved at this point
  TimePoint tStart = startFuture.get();
  TimePoint tEnd = endFuture.get();

  std::chrono::duration<float, std::milli> diff = tEnd - tStart;
  outMs = diff.count();

  return VGREResult::SUCCESS;
}

bool Event::isQueryReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!recorded_ || !future_.valid())
    return false;

  // Check if the future is ready
  return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace core
} // namespace vgre
