// Diagnostic Logger — structured event logging, metrics, and health monitoring.
// Uses shared utilities (LoggingPatterns, ErrorHandlingPatterns, TimeUtils, PlatformDetection)
// to avoid code duplication with shared_utilities.cpp.

#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace vgre {
namespace advanced {

// ── Internal Metrics Helpers (unique to diagnostic_logger) ──

namespace {

double calculateBandwidthGbps(size_t bytes, double duration_ms) {
    if (duration_ms <= 0.0) return 0.0;
    return (bytes / (duration_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
}

struct LatencyStats {
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    size_t sample_count = 0;
};

LatencyStats calculateLatencyStats(const std::vector<double>& latencies) {
    LatencyStats stats;
    if (latencies.empty()) return stats;

    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    stats.sample_count = sorted.size();
    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();

    double sum = 0.0;
    for (double v : sorted) sum += v;
    stats.avg_ms = sum / sorted.size();

    size_t p95 = static_cast<size_t>(sorted.size() * 0.95);
    size_t p99 = static_cast<size_t>(sorted.size() * 0.99);
    if (p95 < sorted.size()) stats.p95_ms = sorted[p95];
    if (p99 < sorted.size()) stats.p99_ms = sorted[p99];

    return stats;
}

std::string platformName() {
    return PlatformDetection::getCurrentPlatform().name;
}

} // anonymous namespace

// ── Lifecycle ──

DiagnosticLogger::DiagnosticLogger() {
    initializeMetrics();
    VGRE_LOG_INFO("DiagnosticLogger",
        LoggingPatterns::formatMessage("DiagnosticLogger", "Initialized",
            "Platform: " + platformName()));
}

DiagnosticLogger::~DiagnosticLogger() {
    flushMetrics();
}

DiagnosticLogger& DiagnosticLogger::instance() {
    static DiagnosticLogger inst;
    return inst;
}

void DiagnosticLogger::initializeMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    performance_metrics_.total_packets_sent = 0;
    performance_metrics_.total_packets_received = 0;
    performance_metrics_.total_bytes_sent = 0;
    performance_metrics_.total_bytes_received = 0;
    performance_metrics_.total_errors = 0;
    performance_metrics_.start_time = std::chrono::steady_clock::now();

    network_quality_.packet_loss_rate = 0.0;
    network_quality_.average_latency_ms = 0.0;
    network_quality_.bandwidth_utilization = 0.0;
    network_quality_.connection_success_rate = 1.0;
}

// ── Security Event Logging ──

void DiagnosticLogger::logSecurityEvent(const std::string& event,
                                        const std::string& client_ip,
                                        const std::string& token_hash) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    SecurityEvent sec_event;
    sec_event.timestamp = TimeUtils::getCurrentTimestamp();
    sec_event.event_type = event;
    sec_event.client_ip = client_ip;
    sec_event.token_hash = token_hash;
    sec_event.platform = platformName();

    security_events_.push_back(sec_event);
    if (security_events_.size() > 1000)
        security_events_.erase(security_events_.begin());

    std::string msg = LoggingPatterns::formatMessage("SecurityEvent", event,
        "Client: " + client_ip + ", Token hash: " + token_hash);

    if (event.find("FAILED") != std::string::npos ||
        event.find("REJECTED") != std::string::npos) {
        VGRE_LOG_ERROR("Security",
            LoggingPatterns::formatError(msg, generateSecuritySuggestion(event)));
    } else {
        VGRE_LOG_INFO("Security", msg);
    }

    updateSecurityMetrics(event);
}

std::string DiagnosticLogger::generateSecuritySuggestion(const std::string& event) {
    if (event.find("TOKEN_MISMATCH") != std::string::npos)
        return "Ensure VGRE_TCP_AUTH_TOKEN matches on both master and worker nodes";
    if (event.find("HANDSHAKE_TIMEOUT") != std::string::npos)
        return "Check network connectivity and increase VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC";
    if (event.find("ENCRYPTION_FAILED") != std::string::npos)
        return "Verify OpenSSL installation and entropy source availability";
    if (event.find("REPLAY_ATTACK") != std::string::npos)
        return "Check for network tampering or clock synchronization issues";
    return "Review security configuration and network setup";
}

void DiagnosticLogger::updateSecurityMetrics(const std::string& event) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (event.find("FAILED") != std::string::npos ||
        event.find("REJECTED") != std::string::npos) {
        security_metrics_.failed_authentications++;
    } else if (event.find("SUCCESS") != std::string::npos) {
        security_metrics_.successful_authentications++;
    }
    security_metrics_.total_security_events++;
}

// ── Network Error Logging ──

void DiagnosticLogger::logNetworkError(VGREResult error,
                                       const std::string& operation,
                                       const std::string& client_ip) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    NetworkError net_error;
    net_error.timestamp = TimeUtils::getCurrentTimestamp();
    net_error.error_code = error;
    net_error.operation = operation;
    net_error.client_ip = client_ip;
    net_error.platform = platformName();

    network_errors_.push_back(net_error);
    if (network_errors_.size() > 500)
        network_errors_.erase(network_errors_.begin());

    std::string error_name = ErrorHandlingPatterns::translateVGREResult(error);
    std::string suggestion = generateNetworkSuggestion(error, operation);

    VGRE_LOG_ERROR("Network",
        LoggingPatterns::formatError(
            LoggingPatterns::formatMessage("NetworkError", operation,
                error_name + " for client " + client_ip),
            suggestion));

    updateNetworkQualityMetrics(error);
}

std::string DiagnosticLogger::generateNetworkSuggestion(VGREResult error,
                                                        const std::string& operation) {
    switch (error) {
        case VGREResult::ERR_NETWORK:
            return "Check network connectivity, firewall rules, and DNS resolution";
        case VGREResult::ERR_TIMEOUT:
            return "Increase timeout values or check for network congestion";
        case VGREResult::ERR_IO:
            if (operation.find("send") != std::string::npos)
                return "Check send buffer size and network interface status";
            if (operation.find("recv") != std::string::npos)
                return "Check receive buffer size and peer connection status";
            return "Verify I/O subsystem and network interface configuration";
        case VGREResult::ERR_BUSY:
            return "Reduce connection rate or increase queue depths";
        case VGREResult::ERR_NOT_SUPPORTED:
            return "Check platform-specific network configuration (Winsock on Windows)";
        default:
            return "Review network configuration and system resources";
    }
}

void DiagnosticLogger::updateNetworkQualityMetrics(VGREResult error) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    network_quality_.total_operations++;
    if (error != VGREResult::SUCCESS) network_quality_.failed_operations++;

    double success_rate = 1.0 - (static_cast<double>(network_quality_.failed_operations) /
                                 network_quality_.total_operations);
    constexpr double alpha = 0.1;
    network_quality_.connection_success_rate =
        (1.0 - alpha) * network_quality_.connection_success_rate + alpha * success_rate;
}

// ── Packet Corruption Detection ──

void DiagnosticLogger::logPacketCorruption(const uint8_t* packet_data,
                                           size_t size,
                                           const std::string& expected_format) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    PacketCorruption corruption;
    corruption.timestamp = TimeUtils::getCurrentTimestamp();
    corruption.packet_size = size;
    corruption.expected_format = expected_format;
    corruption.platform = platformName();

    size_t sample_size = std::min(size, static_cast<size_t>(64));
    corruption.packet_sample.resize(sample_size);
    memcpy(corruption.packet_sample.data(), packet_data, sample_size);

    packet_corruptions_.push_back(corruption);
    if (packet_corruptions_.size() > 100)
        packet_corruptions_.erase(packet_corruptions_.begin());

    std::stringstream hex_dump;
    hex_dump << std::hex << std::setfill('0');
    for (size_t i = 0; i < sample_size; ++i) {
        if (i > 0 && i % 16 == 0) hex_dump << " ";
        hex_dump << std::setw(2) << static_cast<int>(packet_data[i]);
    }

    VGRE_LOG_ERROR("PacketCorruption",
        LoggingPatterns::formatError(
            LoggingPatterns::formatMessage("PacketCorruption", "Detected",
                "Size: " + std::to_string(size) + ", Expected: " + expected_format +
                ", Sample: " + hex_dump.str()),
            "Check network quality, MTU settings, and hardware integrity"));

    updateCorruptionMetrics();
}

void DiagnosticLogger::updateCorruptionMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    corruption_metrics_.total_corruptions++;
    if (performance_metrics_.total_packets_received > 0) {
        corruption_metrics_.corruption_rate =
            static_cast<double>(corruption_metrics_.total_corruptions) /
            performance_metrics_.total_packets_received;
    }
}

// ── Performance Telemetry ──

void DiagnosticLogger::recordPacketTransfer(bool is_send, size_t bytes,
                                            double latency_ms, bool success) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (is_send) {
        performance_metrics_.total_packets_sent++;
        performance_metrics_.total_bytes_sent += bytes;
        send_latencies_.push_back(latency_ms);
    } else {
        performance_metrics_.total_packets_received++;
        performance_metrics_.total_bytes_received += bytes;
        recv_latencies_.push_back(latency_ms);
    }
    if (!success) performance_metrics_.total_errors++;

    if (send_latencies_.size() > 1000) send_latencies_.erase(send_latencies_.begin());
    if (recv_latencies_.size() > 1000) recv_latencies_.erase(recv_latencies_.begin());

    updateBandwidthMetrics(bytes, latency_ms);
}

void DiagnosticLogger::updateBandwidthMetrics(size_t bytes, double latency_ms) {
    double bw = calculateBandwidthGbps(bytes, latency_ms);
    constexpr double alpha = 0.1;
    network_quality_.average_bandwidth_gbps =
        (1.0 - alpha) * network_quality_.average_bandwidth_gbps + alpha * bw;
    network_quality_.bandwidth_utilization =
        std::min(1.0, network_quality_.average_bandwidth_gbps / 1.0);
}

// ── Operational Dashboard ──

DiagnosticLogger::OperationalMetrics DiagnosticLogger::getOperationalMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    OperationalMetrics m;

    m.total_packets_sent = performance_metrics_.total_packets_sent;
    m.total_packets_received = performance_metrics_.total_packets_received;
    m.total_bytes_sent = performance_metrics_.total_bytes_sent;
    m.total_bytes_received = performance_metrics_.total_bytes_received;
    m.total_errors = performance_metrics_.total_errors;

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - performance_metrics_.start_time);
    m.uptime_seconds = uptime.count();

    m.packet_loss_rate = network_quality_.packet_loss_rate;
    m.average_latency_ms = network_quality_.average_latency_ms;
    m.bandwidth_utilization = network_quality_.bandwidth_utilization;
    m.connection_success_rate = network_quality_.connection_success_rate;

    m.successful_authentications = security_metrics_.successful_authentications;
    m.failed_authentications = security_metrics_.failed_authentications;
    m.total_security_events = security_metrics_.total_security_events;

    m.total_corruptions = corruption_metrics_.total_corruptions;
    m.corruption_rate = corruption_metrics_.corruption_rate;

    if (!send_latencies_.empty()) {
        auto s = calculateLatencyStats(send_latencies_);
        m.send_latency_avg_ms = s.avg_ms;
        m.send_latency_p95_ms = s.p95_ms;
        m.send_latency_p99_ms = s.p99_ms;
    }
    if (!recv_latencies_.empty()) {
        auto r = calculateLatencyStats(recv_latencies_);
        m.recv_latency_avg_ms = r.avg_ms;
        m.recv_latency_p95_ms = r.p95_ms;
        m.recv_latency_p99_ms = r.p99_ms;
    }
    return m;
}

void DiagnosticLogger::generateOperationalReport() {
    auto m = getOperationalMetrics();
    VGRE_LOG_INFO("DiagnosticLogger", "=== Operational Metrics Report ===");
    VGRE_LOG_INFO("DiagnosticLogger",
        LoggingPatterns::formatMessage("Performance", "Summary",
            "Uptime: " + std::to_string(m.uptime_seconds) + "s, Sent: " +
            std::to_string(m.total_packets_sent) + ", Recv: " +
            std::to_string(m.total_packets_received) + ", Errors: " +
            std::to_string(m.total_errors)));
    VGRE_LOG_INFO("DiagnosticLogger",
        LoggingPatterns::formatMessage("NetworkQuality", "Summary",
            "Success: " + std::to_string(m.connection_success_rate * 100.0) + "%, Latency: " +
            std::to_string(m.average_latency_ms) + "ms, BW util: " +
            std::to_string(m.bandwidth_utilization * 100.0) + "%"));
    VGRE_LOG_INFO("DiagnosticLogger",
        LoggingPatterns::formatMessage("Security", "Summary",
            "Auth OK: " + std::to_string(m.successful_authentications) + ", Auth fail: " +
            std::to_string(m.failed_authentications) + ", Events: " +
            std::to_string(m.total_security_events)));
    if (m.total_corruptions > 0) {
        VGRE_LOG_WARN("DiagnosticLogger",
            LoggingPatterns::formatMessage("Corruption", "Summary",
                "Total: " + std::to_string(m.total_corruptions) + ", Rate: " +
                std::to_string(m.corruption_rate * 100.0) + "%"));
    }
}

// ── Health Check ──

DiagnosticLogger::HealthStatus DiagnosticLogger::getHealthStatus() const {
    auto m = getOperationalMetrics();
    HealthStatus status;
    status.overall_health = HealthLevel::HEALTHY;

    if (m.connection_success_rate < 0.95) {
        status.overall_health = std::max(status.overall_health, HealthLevel::DEGRADED);
        status.issues.push_back("Low connection success rate: " +
            std::to_string(m.connection_success_rate * 100.0) + "%");
    }

    double error_rate = 0.0;
    if (m.total_packets_sent + m.total_packets_received > 0) {
        error_rate = static_cast<double>(m.total_errors) /
                    (m.total_packets_sent + m.total_packets_received);
    }
    if (error_rate > 0.05) {
        status.overall_health = std::max(status.overall_health, HealthLevel::DEGRADED);
        status.issues.push_back("High error rate: " + std::to_string(error_rate * 100.0) + "%");
    }

    if (m.corruption_rate > 0.01) {
        status.overall_health = std::max(status.overall_health, HealthLevel::CRITICAL);
        status.issues.push_back("High corruption rate: " +
            std::to_string(m.corruption_rate * 100.0) + "%");
    }

    if (m.failed_authentications > 0) {
        double auth_fail = static_cast<double>(m.failed_authentications) /
                          std::max(1u, m.total_security_events);
        if (auth_fail > 0.1) {
            status.overall_health = std::max(status.overall_health, HealthLevel::DEGRADED);
            status.issues.push_back("High auth failure rate: " +
                std::to_string(auth_fail * 100.0) + "%");
        }
    }

    switch (status.overall_health) {
        case HealthLevel::HEALTHY:  status.status_message = "All systems operating normally"; break;
        case HealthLevel::DEGRADED: status.status_message = "System performance degraded — monitoring required"; break;
        case HealthLevel::CRITICAL: status.status_message = "Critical issues — immediate attention required"; break;
    }
    return status;
}

void DiagnosticLogger::flushMetrics() {
    std::string filename = "/tmp/vgre_tcp_cluster_metrics.json";
    try {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        auto m = getOperationalMetrics();
        file << "{\n";
        file << "  \"timestamp\": \"" << TimeUtils::getCurrentTimestamp() << "\",\n";
        file << "  \"platform\": \"" << platformName() << "\",\n";
        file << "  \"uptime_seconds\": " << m.uptime_seconds << ",\n";
        file << "  \"total_packets_sent\": " << m.total_packets_sent << ",\n";
        file << "  \"total_packets_received\": " << m.total_packets_received << ",\n";
        file << "  \"total_bytes_sent\": " << m.total_bytes_sent << ",\n";
        file << "  \"total_bytes_received\": " << m.total_bytes_received << ",\n";
        file << "  \"total_errors\": " << m.total_errors << ",\n";
        file << "  \"connection_success_rate\": " << m.connection_success_rate << ",\n";
        file << "  \"average_latency_ms\": " << m.average_latency_ms << ",\n";
        file << "  \"bandwidth_utilization\": " << m.bandwidth_utilization << ",\n";
        file << "  \"successful_authentications\": " << m.successful_authentications << ",\n";
        file << "  \"failed_authentications\": " << m.failed_authentications << ",\n";
        file << "  \"total_corruptions\": " << m.total_corruptions << ",\n";
        file << "  \"corruption_rate\": " << m.corruption_rate << "\n";
        file << "}\n";
        file.close();
        VGRE_LOG_DEBUG("DiagnosticLogger", "Metrics exported to " + filename);
    } catch (const std::exception& e) {
        VGRE_LOG_WARN("DiagnosticLogger", "Failed to export metrics: " + std::string(e.what()));
    }
}

} // namespace advanced
} // namespace vgre