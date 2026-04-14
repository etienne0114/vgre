#ifndef VGRE_COMMON_RETRY_H
#define VGRE_COMMON_RETRY_H

/**
 * withRetry — generic retry wrapper for transient I/O and resource failures.
 *
 * Calls fn() up to maxAttempts times.  Between failures it sleeps for
 * baseDelayMs << attempt (exponential back-off: 50, 100, 200 ms by default).
 *
 * Only retries on ERR_IO and ERR_OUT_OF_MEMORY — hard errors (ERR_INVALID_VALUE,
 * ERR_COMPILATION, …) are returned immediately without retrying.
 *
 * Usage:
 *   VGREResult r = vgre::withRetry([&]() -> VGREResult { ... }, 3, 50);
 */

#include "vgre/common/error_codes.h"
#include <chrono>
#include <thread>

namespace vgre {

template <typename Fn>
VGREResult withRetry(Fn &&fn, int maxAttempts = 3, int baseDelayMs = 50) {
  VGREResult result = VGREResult::ERR_IO;
  for (int attempt = 0; attempt < maxAttempts; ++attempt) {
    result = fn();
    if (result == VGREResult::SUCCESS)
      return result;
    // Hard errors — no point retrying
    if (result != VGREResult::ERR_IO &&
        result != VGREResult::ERR_OUT_OF_MEMORY)
      return result;
    if (attempt < maxAttempts - 1) {
      int delayMs = baseDelayMs << attempt; // 50, 100, 200 ms …
      std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }
  }
  return result;
}

} // namespace vgre

#endif // VGRE_COMMON_RETRY_H
