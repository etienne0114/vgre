#ifndef VGRE_COMMON_LOGGER_H
#define VGRE_COMMON_LOGGER_H

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <deque>
#include <atomic>
#include <fstream>
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
  ERR = 3,
  NONE = 4
};

// ── Thread-safe singleton logger with ring buffer + optional file sink ─────
class Logger {
public:
  static Logger &instance() {
    static Logger* inst = []() {
      auto* l = new Logger();
      const char* envVal = std::getenv("VGRE_LOG_LEVEL");
      if (envVal) {
        std::string s(envVal);
        if      (s == "DEBUG") l->setLevel(LogLevel::DEBUG);
        else if (s == "INFO")  l->setLevel(LogLevel::INFO);
        else if (s == "WARN")  l->setLevel(LogLevel::WARN);
        else if (s == "ERROR") l->setLevel(LogLevel::ERR);
        else if (s == "NONE")  l->setLevel(LogLevel::NONE);
      }
      const char* logFile = std::getenv("VGRE_LOG_FILE");
      if (logFile && logFile[0]) {
        l->openFile(logFile);
      }
      return l;
    }();
    return *inst;
  }

  void setLevel(LogLevel level) { minLevel_.store(level, std::memory_order_relaxed); }
  LogLevel getLevel() const { return minLevel_.load(std::memory_order_relaxed); }

  // Open a log file sink (appends; thread-safe after first call).
  void openFile(const char* path) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (fileOut_.is_open()) fileOut_.close();
    fileOut_.open(path, std::ios::app);
  }

  void log(LogLevel level, const std::string &component,
           const std::string &message) {
    if (level < minLevel_.load(std::memory_order_relaxed)) return;

    // Build the formatted line once under lock to keep timestamp consistent.
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Wall-clock timestamp in microseconds
    auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto us    = std::chrono::duration_cast<std::chrono::microseconds>(
                     now.time_since_epoch()) % 1000000;

    std::ostringstream line;
    {
      std::tm tm_info{};
#if defined(_WIN32)
      localtime_s(&tm_info, &now_t);
#else
      localtime_r(&now_t, &tm_info);
#endif
      line << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(6) << us.count()
           << " [" << levelTag(level) << "] "
           << '[' << component << "] "
           << message;
    }

    const std::string& s = line.str();

    // Ring buffer: keep last 4096 log lines in memory for vgre_get_logs()
    logBuffer_.push_back(s);
    if (logBuffer_.size() > kMaxBuffered) logBuffer_.pop_front();

    // Stderr sink (always active unless level == NONE)
    if (level >= LogLevel::INFO) {
      std::cerr << s << '\n';
    }

    // File sink (if configured)
    if (fileOut_.is_open()) {
      fileOut_ << s << '\n';
      fileOut_.flush();
    }
  }

  std::vector<std::string> getRecentLogs() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return std::vector<std::string>(logBuffer_.begin(), logBuffer_.end());
  }

private:
  Logger() = default;
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  static const char *levelTag(LogLevel l) noexcept {
    switch (l) {
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO ";
    case LogLevel::WARN:  return "WARN ";
    case LogLevel::ERR:   return "ERROR";
    default:              return "?????";
    }
  }

  static constexpr size_t kMaxBuffered = 4096;

  std::atomic<LogLevel> minLevel_{LogLevel::INFO};
  mutable std::recursive_mutex mutex_;
  std::deque<std::string> logBuffer_;
  std::ofstream fileOut_;
};

// ── Compile-time path basename (strips directory prefix) ──────────────────
namespace detail {
constexpr const char* log_basename(const char* path) noexcept {
  const char* last = path;
  for (const char* p = path; *p; ++p)
    if (*p == '/' || *p == '\\') last = p + 1;
  return last;
}
} // namespace detail

// ── Convenience macros ─────────────────────────────────────────────────────
#define VGRE_LOG_DEBUG(comp, msg)                                              \
  ::vgre::Logger::instance().log(::vgre::LogLevel::DEBUG, comp, msg)
#define VGRE_LOG_INFO(comp, msg)                                               \
  ::vgre::Logger::instance().log(::vgre::LogLevel::INFO, comp, msg)
#define VGRE_LOG_WARN(comp, msg)                                               \
  ::vgre::Logger::instance().log(::vgre::LogLevel::WARN, comp, msg)
// ERROR level appends [filename:line] so failures are immediately locatable.
#define VGRE_LOG_ERROR(comp, msg)                                              \
  ::vgre::Logger::instance().log(::vgre::LogLevel::ERR, comp,                 \
      std::string(msg) + " [" +                                                \
      ::vgre::detail::log_basename(__FILE__) + ":" +                           \
      std::to_string(__LINE__) + "]")

} // namespace vgre

#endif // VGRE_COMMON_LOGGER_H
