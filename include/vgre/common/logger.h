#ifndef VGRE_COMMON_LOGGER_H
#define VGRE_COMMON_LOGGER_H

#include <chrono>
#include <ctime>
#include <deque>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace vgre {

// ── Log severity levels ────────────────────────────────────────────────────
enum class LogLevel : uint8_t {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4
};

// ── Thread-safe singleton logger ───────────────────────────────────────────
class Logger {
public:
  static Logger &instance() {
    static Logger* inst = new Logger();
    static bool envChecked = false;
    if (!envChecked) {
      envChecked = true;
      const char* envVal = std::getenv("VGRE_LOG_LEVEL");
      if (envVal) {
        std::string s(envVal);
        if (s == "DEBUG") inst->setLevel(LogLevel::DEBUG);
        else if (s == "INFO") inst->setLevel(LogLevel::INFO);
        else if (s == "WARN") inst->setLevel(LogLevel::WARN);
        else if (s == "ERROR") inst->setLevel(LogLevel::ERROR);
        else if (s == "NONE") inst->setLevel(LogLevel::NONE);
      }
    }
    return *inst;
  }

  void setLevel(LogLevel level) { minLevel_.store(level, std::memory_order_relaxed); }
  LogLevel getLevel() const { return minLevel_.load(std::memory_order_relaxed); }

  void log(LogLevel level, const std::string &component,
           const std::string &message) {
    if (level < minLevel_.load(std::memory_order_relaxed))
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &timeT);
#else
    localtime_r(&timeT, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << ms.count() << " [" << levelTag(level) << "] "
        << "[" << component << "] " << message;

    std::string fullMsg = oss.str();
    std::cerr << fullMsg << '\n';

    // Buffer for GUI
    logBuffer_.push_back(fullMsg);
    if (logBuffer_.size() > 100) {
      logBuffer_.pop_front();
    }
  }

  std::vector<std::string> getRecentLogs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<std::string>(logBuffer_.begin(), logBuffer_.end());
  }

private:
  Logger() = default;
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static const char *levelTag(LogLevel l) {
    switch (l) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO ";
    case LogLevel::WARN:
      return "WARN ";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "?????";
    }
  }

  std::atomic<LogLevel> minLevel_{LogLevel::INFO};
  std::mutex mutex_;
  std::deque<std::string> logBuffer_;
};

// ── Convenience macros ─────────────────────────────────────────────────────
#define VGRE_LOG_DEBUG(comp, msg)                                              \
  ::vgre::Logger::instance().log(::vgre::LogLevel::DEBUG, comp, msg)
#define VGRE_LOG_INFO(comp, msg)                                               \
  ::vgre::Logger::instance().log(::vgre::LogLevel::INFO, comp, msg)
#define VGRE_LOG_WARN(comp, msg)                                               \
  ::vgre::Logger::instance().log(::vgre::LogLevel::WARN, comp, msg)
#define VGRE_LOG_ERROR(comp, msg)                                              \
  ::vgre::Logger::instance().log(::vgre::LogLevel::ERROR, comp, msg)

} // namespace vgre

#endif // VGRE_COMMON_LOGGER_H
