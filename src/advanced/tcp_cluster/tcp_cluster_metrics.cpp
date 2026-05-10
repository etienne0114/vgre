/**
 * VGRE TCP Cluster Manager - Metrics and Reporting Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/common/logger.h"
#include <fstream>

namespace vgre {
namespace advanced {

void TCPClusterManager::exportMetricsForMonitoring() {
    try {
        auto metrics = DiagnosticLogger::instance().getOperationalMetrics();
        auto platform = PlatformDetection::getCurrentPlatform();
        
        std::stringstream json;
        json << "{\n";
        json << "  \"timestamp\": \"" << TimeUtils::getCurrentTimestamp() << "\",\n";
        json << "  \"platform\": {\n";
        json << "    \"name\": \"" << platform.name << "\",\n";
        json << "    \"shm_supported\": " << (platform.supports_shm_local ? "true" : "false") << ",\n";
        json << "    \"rdma_supported\": " << (platform.supports_rdma ? "true" : "false") << "\n";
        json << "  },\n";
        json << "  \"cluster\": {\n";
        json << "    \"role\": \"" << (is_master_ ? "master" : "worker") << "\",\n";
        json << "    \"enabled\": " << (enabled_ ? "true" : "false") << ",\n";
        json << "    \"security_enabled\": " << (security_enabled_ ? "true" : "false") << "\n";
        json << "  },\n";
        
        size_t active = 0;
        json << "  \"nodes\": [\n";
        {
            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
            for (size_t i = 0; i < clients_.size(); ++i) {
                auto& c = clients_[i];
                if (!c) continue;
                if (c->active) active++;
                json << "    {\n";
                json << "      \"ip\": \"" << c->ip_address << "\",\n";
                json << "      \"active\": " << (c->active ? "true" : "false") << ",\n";
                json << "      \"latency_ms\": " << c->network_latency_ms << ",\n";
                json << "      \"bandwidth_gbps\": " << c->network_bandwidth_gbps << "\n";
                json << "    }" << (i == clients_.size() - 1 ? "" : ",") << "\n";
            }
        }
        json << "  ],\n";
        
        json << "  \"performance\": {\n";
        json << "    \"active_nodes\": " << active << ",\n";
        json << "    \"total_packets_sent\": " << metrics.total_packets_sent << ",\n";
        json << "    \"total_bytes_sent\": " << metrics.total_bytes_sent << ",\n";
        json << "    \"total_packets_received\": " << metrics.total_packets_received << ",\n";
        json << "    \"total_errors\": " << metrics.total_errors << "\n";
        json << "  }\n";
        json << "}";

        // Export to a local JSON file in the working directory (cross-platform)
        // In a real production environment, this could be sent to a telemetry service.
        std::ofstream f("vgre_tcp_cluster_metrics.json");
        if (f.is_open()) { f << json.str(); f.close(); }
    } catch (...) {}
}

void TCPClusterManager::generateOperationalReport() {
    VGRE_LOG_INFO("TCPCluster", "=== TCP Cluster Operational Report ===");
    size_t active = 0, total = 0;
    { std::lock_guard<std::recursive_mutex> lock(clients_mutex_); total = clients_.size(); for (const auto& c : clients_) if (c && c->active) active++; }
    VGRE_LOG_INFO("TCPCluster", "Connections: " + std::to_string(active) + "/" + std::to_string(total));
    VGRE_LOG_INFO("TCPCluster", "Performance: Sent=" + std::to_string(global_packets_sent_.load()) + " pkts, Recv=" + std::to_string(global_packets_received_.load()) + " pkts");
}

} // namespace advanced
} // namespace vgre
