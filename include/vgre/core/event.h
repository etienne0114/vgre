#ifndef VGRE_CORE_EVENT_H
#define VGRE_CORE_EVENT_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <chrono>
#include <future>
#include <mutex>

namespace vgre {
namespace core {

/**
 * @brief Represents a synchronization point in a VGRE stream.
 *
 * Modeled after CUDA Events, this class allows measuring elapsed time
 * and synchronizing host threads against specific points in a stream's
 * execution timeline.
 */
class Event {
public:
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  Event();
  ~Event() = default;

  // ── Modifiers ────────────────────────────────────────────────────────────

  /**
   * @brief Submits this event to be recorded by the given stream.
   *
   * When the stream's worker thread reaches this point in its queue, it will
   * capture the current timestamp and resolve the internal future.
   *
   * @param stream The stream on which to record the event.
   * @return VGREResult SUCCESS on success.
   */
  VGREResult record(StreamId stream);

  // ── Operations ───────────────────────────────────────────────────────────

  /**
   * @brief Blocks the calling thread until the event has been recorded.
   * @return VGREResult SUCCESS or an error code if recording failed.
   */
  VGREResult synchronize() const;

  /**
   * @brief Calculates the elapsed time in milliseconds between a start event
   *        and this (end) event.
   *
   * Both events must be recorded (resolved) before calling this. If either
   * is not yet recorded, this will block until they are.
   *
   * @param start The starting event.
   * @param outMs Output parameter for elapsed milliseconds.
   * @return VGREResult SUCCESS on success, ERROR_NOT_READY if unrecorded
   *         events shouldn't block (though this implementation blocks).
   */
  VGREResult elapsedTime(const Event &start, float &outMs) const;

  /**
   * @brief Queries if the event has been recorded yet (non-blocking).
   * @return true if recorded, false otherwise.
   */
  bool isQueryReady() const;

private:
  mutable std::mutex mutex_;
  std::shared_future<TimePoint> future_;
  bool recorded_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_EVENT_H
