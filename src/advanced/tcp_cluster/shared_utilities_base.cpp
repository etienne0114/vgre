/**
 * VGRE Shared Utilities - Base Utilities
 */

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace vgre {
namespace advanced {

std::string LoggingPatterns::formatMessage(const std::string& component, const std::string& operation, const std::string& details, const std::string& context) {
    std::stringstream ss; ss << "[" << component << "] " << operation;
    if (!details.empty()) ss << ": " << details;
    if (!context.empty()) ss << " (context: " << context << ")";
    return ss.str();
}

std::string LoggingPatterns::formatError(const std::string& error, const std::string& suggestion) {
    std::stringstream ss;
    ss << "Error: " << error;
    if (!suggestion.empty()) {
        ss << " | Suggestion: " << suggestion;
    }
    return ss.str();
}

std::string TimeUtils::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_info;
#ifdef _WIN32
    localtime_s(&tm_info, &time_t);
#else
    localtime_r(&time_t, &tm_info);
#endif
    std::stringstream ss; ss << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

uint64_t TimeUtils::getCurrentTimestampMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double TimeUtils::calculateElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

PlatformInfo PlatformDetection::getCurrentPlatform() {
    PlatformInfo info;
#ifdef _WIN32
    info.type = PlatformType::WINDOWS; info.name = "Windows"; info.supports_shm_local = true; info.supports_rdma = true;
#elif defined(__linux__)
    info.type = PlatformType::LINUX; info.name = "Linux"; info.supports_shm_local = true; info.supports_rdma = true;
#else
    info.type = PlatformType::UNKNOWN; info.name = "Unknown";
#endif
    return info;
}

} // namespace advanced
} // namespace vgre
