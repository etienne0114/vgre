#pragma once

#include "vgre/common/error_codes.h"
#include "vgre/advanced/tcp_cluster.h"
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstdint>

namespace vgre {
namespace advanced {

/**
 * Diagnostic Logger and Operational Monitoring System
 * 
 * Provides comprehensive diagnostic logging with structured event logging,
 * metrics collection, and operational monitoring for the TCP cluster system.
 * 
 * Features:
 * - Security event logging with automated suggestions
 * - Network error logging with context and recovery recommendations
 * - Packet corruption detection and analysis
 * - Performance telemetry collection for dashboards
 * - Health monitoring and alerting
 * - Cross-platform logging with consistent formatting
 */
class DiagnosticLogger {
public:
    // Singleton access
    static DiagnosticLogger& instance();
    
    // Prevent copy/move
    DiagnosticLogger(const DiagnosticLogger&) = delete;
    DiagnosticLogger& operator=(const DiagnosticLogger&) = delete;
    DiagnosticLogger(DiagnosticLogger&&) = delete;
    DiagnosticLogger& operator=(DiagnosticLogger&&) = delete;
    
    // ── Security Event Logging ──────────────────────────────────────────────
    
    /**
     * Log security-related events with context and automated suggestions
     * @param event Event type (e.g., "TOKEN_MISMATCH", "HANDSHAKE_SUCCESS")
     * @param client_ip IP address of the client involved
     * @param token_hash Hash of the authentication token for debugging
     */
    void logSecurityEvent(const std::string& event, 
                         const std::string& client_ip,
                         const std::string& token_hash);
    
    // ── Network Error Logging ───────────────────────────────────────────────
    
    /**
     * Log network errors with context and automated recovery suggestions
     * @param error The VGREResult error code
     * @param operation The operation that failed (e.g., "send_packet", "connect")
     * @param client_ip IP address of the client involved
     */
    void logNetworkError(VGREResult error, 
                        const std::string& operation,
                        const std::string& client_ip);
    
    // ── Packet Corruption Detection ─────────────────────────────────────────
    
    /**
     * Log packet corruption events with packet analysis
     * @param packet_data Raw packet data for analysis
     * @param size Size of the packet
     * @param expected_format Expected packet format description
     */
    void logPacketCorruption(const uint8_t* packet_data, 
                           size_t size,
                           const std::string& expected_format);
    
    // ── Performance Telemetry ───────────────────────────────────────────────
    
    /**
     * Record packet transfer metrics for performance monitoring
     * @param is_send True for send operations, false for receive
     * @param bytes Number of bytes transferred
     * @param latency_ms Transfer latency in milliseconds
     * @param success Whether the transfer was successful
     */
    void recordPacketTransfer(bool is_send, size_t bytes, 
                             double latency_ms, bool success);
    
    // ── Operational Metrics ─────────────────────────────────────────────────
    
    struct OperationalMetrics {
        // Performance counters
        uint64_t total_packets_sent = 0;
        uint64_t total_packets_received = 0;
        uint64_t total_bytes_sent = 0;
        uint64_t total_bytes_received = 0;
        uint64_t total_errors = 0;
        uint64_t uptime_seconds = 0;
        
        // Network quality metrics
        double packet_loss_rate = 0.0;
        double average_latency_ms = 0.0;
        double bandwidth_utilization = 0.0;
        double connection_success_rate = 1.0;
        
        // Latency statistics
        double send_latency_avg_ms = 0.0;
        double send_latency_p95_ms = 0.0;
        double send_latency_p99_ms = 0.0;
        double recv_latency_avg_ms = 0.0;
        double recv_latency_p95_ms = 0.0;
        double recv_latency_p99_ms = 0.0;
        
        // Security metrics
        uint32_t successful_authentications = 0;
        uint32_t failed_authentications = 0;
        uint32_t total_security_events = 0;
        
        // Corruption metrics
        uint32_t total_corruptions = 0;
        double corruption_rate = 0.0;
    };
    
    /**
     * Get current operational metrics for dashboards
     * @return Current metrics snapshot
     */
    OperationalMetrics getOperationalMetrics() const;
    
    /**
     * Generate and log comprehensive operational report
     */
    void generateOperationalReport();
    
    // ── Health Monitoring ───────────────────────────────────────────────────
    
    enum class HealthLevel {
        HEALTHY = 0,
        DEGRADED = 1,
        CRITICAL = 2
    };
    
    struct HealthStatus {
        HealthLevel overall_health = HealthLevel::HEALTHY;
        std::string status_message;
        std::vector<std::string> issues;
    };
    
    /**
     * Get current system health status
     * @return Health status with issues and recommendations
     */
    HealthStatus getHealthStatus() const;
    
    /**
     * Flush metrics to external monitoring systems
     */
    void flushMetrics();

private:
    DiagnosticLogger();
    ~DiagnosticLogger();
    
    // ── Internal Data Structures ────────────────────────────────────────────
    
    struct SecurityEvent {
        std::string timestamp;
        std::string event_type;
        std::string client_ip;
        std::string token_hash;
        std::string platform;
    };
    
    struct NetworkError {
        std::string timestamp;
        VGREResult error_code;
        std::string operation;
        std::string client_ip;
        std::string platform;
    };
    
    struct PacketCorruption {
        std::string timestamp;
        size_t packet_size;
        std::string expected_format;
        std::string platform;
        std::vector<uint8_t> packet_sample;
    };
    
    struct PerformanceMetrics {
        uint64_t total_packets_sent = 0;
        uint64_t total_packets_received = 0;
        uint64_t total_bytes_sent = 0;
        uint64_t total_bytes_received = 0;
        uint64_t total_errors = 0;
        std::chrono::steady_clock::time_point start_time;
    };
    
    struct NetworkQualityMetrics {
        double packet_loss_rate = 0.0;
        double average_latency_ms = 0.0;
        double average_bandwidth_gbps = 0.0;
        double bandwidth_utilization = 0.0;
        double connection_success_rate = 1.0;
        uint64_t total_operations = 0;
        uint64_t failed_operations = 0;
    };
    
    struct SecurityMetrics {
        uint32_t successful_authentications = 0;
        uint32_t failed_authentications = 0;
        uint32_t total_security_events = 0;
    };
    
    struct CorruptionMetrics {
        uint32_t total_corruptions = 0;
        double corruption_rate = 0.0;
    };
    
    // ── Member Variables ─────────────────────────────────────────────────────
    
    mutable std::mutex events_mutex_;
    mutable std::mutex metrics_mutex_;
    
    // Event storage
    std::vector<SecurityEvent> security_events_;
    std::vector<NetworkError> network_errors_;
    std::vector<PacketCorruption> packet_corruptions_;
    
    // Performance data
    PerformanceMetrics performance_metrics_;
    NetworkQualityMetrics network_quality_;
    SecurityMetrics security_metrics_;
    CorruptionMetrics corruption_metrics_;
    
    // Latency tracking
    std::vector<double> send_latencies_;
    std::vector<double> recv_latencies_;
    
    // ── Helper Methods ───────────────────────────────────────────────────────
    
    void initializeMetrics();
    
    std::string generateSecuritySuggestion(const std::string& event);
    std::string generateNetworkSuggestion(VGREResult error, const std::string& operation);
    
    void updateSecurityMetrics(const std::string& event);
    void updateNetworkQualityMetrics(VGREResult error);
    void updateCorruptionMetrics();
    void updateBandwidthMetrics(size_t bytes, double latency_ms);
};

} // namespace advanced
} // namespace vgre