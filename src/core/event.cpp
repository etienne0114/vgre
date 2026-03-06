#include "vgre/core/event.h"
#include "vgre/core/scheduler.h"

namespace vgre {
namespace core {

Event::Event() : recorded_(false) { future_ = promise_.get_future().share(); }

VGREResult Event::record(StreamId stream) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Reset promise/future if this event is being recorded again
  if (recorded_) {
    promise_ = std::promise<TimePoint>();
    future_ = promise_.get_future().share();
    recorded_ = false;
  }

  // We must extract the promise into a shared_ptr so the lambda can capture it
  // and resolve it when the task executes on the stream thread.
  auto sharedPromise =
      std::make_shared<std::promise<TimePoint>>(std::move(promise_));

  // Create a replacement promise for the class member, though we shouldn't
  // really need it unless recorded again (handled above). But to keep
  // `promise_` valid:
  promise_ = std::promise<TimePoint>();

  auto task = [sharedPromise]() {
    sharedPromise->set_value(std::chrono::steady_clock::now());
  };

  VGREResult res;
  // Submit the task to the given stream via the Scheduler singleton.
  // The future returned by submitStreamTask isn't directly the event future,
  // we care about the inner lambda executing.
  try {
    auto fut = Scheduler::instance().submitStreamTask(stream, task);
    res = VGREResult::SUCCESS;
  } catch (...) {
    res = VGREResult::ERROR_UNKNOWN;
  }

  if (res == VGREResult::SUCCESS) {
    recorded_ = true;
  }
  return res;
}

VGREResult Event::synchronize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!recorded_) {
    // Event hasn't even been submitted to record yet
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Future wait blocks until the promise is resolved by the worker thread
  auto fut = future_; // copy under lock to be safe
  lock.unlock();

  if (fut.valid()) {
    fut.wait();
    return VGREResult::SUCCESS;
  }
  return VGREResult::ERROR_UNKNOWN;
}

VGREResult Event::elapsedTime(const Event &start, float &outMs) const {
  // Synchronize both events to ensure they have timestamps
  if (start.synchronize() != VGREResult::SUCCESS ||
      this->synchronize() != VGREResult::SUCCESS) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Both are valid and resolved at this point
  TimePoint tStart = start.future_.get();
  TimePoint tEnd = this->future_.get();

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
