#include "vgre/core/event.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"
#include "vgre/core/scheduler.h"

namespace vgre {
namespace core {

Event::Event() : recorded_(false) {}

VGREResult Event::record(StreamId stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Create a new promise for this specific recording execution
  auto sharedPromise = std::make_shared<std::promise<TimePoint>>();
  future_ = sharedPromise->get_future().share();

  auto task = [sharedPromise]() {
    try {
      sharedPromise->set_value(std::chrono::steady_clock::now());
    } catch (...) {
      // Ignore if value is already set or promise is broken
    }
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
  } else {
    recorded_ = false;
  }
  return res;
}

VGREResult Event::synchronize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!recorded_) {
    // Event hasn't even been submitted to record yet
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Future wait blocks until the promise is resolved by the worker thread
  auto fut = future_; // copy under lock to be safe
  lock.unlock();

  if (fut.valid()) {
    // Add a timeout to prevent indefinite blocking in test environments
    // Configurable via VGRE_EVENT_TIMEOUT_MS (default 5000ms)
    static const int eventTimeoutMs = []() -> int {
        const char* e = vgre_get_config("VGRE_EVENT_TIMEOUT_MS");
        if (e) {
            try {
                int v = std::stoi(e);
                if (v >= 100 && v <= 60000) return v; // 100ms to 60s range
            } catch (...) {}
        }
        return 5000; // 5 seconds default
    }();
    if (fut.wait_for(std::chrono::milliseconds(eventTimeoutMs)) == std::future_status::timeout) {
      return VGREResult::ERR_TIMEOUT;
    }
    return VGREResult::SUCCESS;
  }
  return VGREResult::ERR_INVALID_VALUE;
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
