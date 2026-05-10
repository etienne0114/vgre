#pragma once

#include <string>

namespace vgre {
namespace advanced {

/**
 * Shared logging utilities to eliminate string concatenation duplication
 * across TCP cluster components.
 */
class LoggingPatterns {
public:
    static std::string formatMessage(const std::string& component,
                                   const std::string& operation,
                                   const std::string& details,
                                   const std::string& context = "");
    
    static std::string formatError(const std::string& error,
                                 const std::string& suggestion);
    
    static std::string formatPerformanceMetric(const std::string& metric_name,
                                             double value,
                                             const std::string& unit = "");
    
    static std::string formatConnectionInfo(const std::string& ip,
                                          int port,
                                          const std::string& status);
    
    static std::string formatDataTransfer(size_t bytes,
                                        double duration_ms,
                                        const std::string& direction);
    
    static void logWithTimestamp(const std::string& level,
                                const std::string& component,
                                const std::string& message);

    static std::string formatSecurityAlert(const std::string& event,
                                         const std::string& severity,
                                         const std::string& suggestion);

    static std::string formatRecoveryAction(const std::string& error,
                                          const std::string& action,
                                          const std::string& component,
                                          const std::string& additional_context = "");
};

} // namespace advanced
} // namespace vgre
