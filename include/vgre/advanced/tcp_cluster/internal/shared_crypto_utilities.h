#pragma once

#include <string>
#include <cstdint>

namespace vgre {
namespace advanced {

class CryptoUtils {
public:
    static std::string computeHmacHex(const std::string& key, const std::string& msg);
    static bool verifyHmacHex(const std::string& key, const std::string& msg, const std::string& hexTag);
    static std::string computeTokenFingerprint(const std::string& token);
};

class TimeUtils {
public:
    static std::string getCurrentTimestamp();
    static uint64_t getCurrentTimestampMs();
    static double getElapsedTimeMs(uint64_t start_ms);
    static double calculateElapsedMs(std::chrono::steady_clock::time_point start);
};

} // namespace advanced
} // namespace vgre
