// Configuration Manager Documentation — serialization (JSON/YAML/Env) and doc generation.

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <sstream>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vgre {
namespace advanced {

// ── Serialization ──

std::string ConfigurationManager::configurationToJson(const ClusterConfiguration& config) {
    std::ostringstream j;
    j << "{\n";
    j << "  \"handshake_timeout_sec\": " << config.handshake_timeout_sec << ",\n";
    j << "  \"max_queue_depth\": " << config.max_queue_depth << ",\n";
    j << "  \"max_packets_per_sec\": " << config.max_packets_per_sec << ",\n";
    j << "  \"connection_retry_attempts\": " << config.connection_retry_attempts << ",\n";
    j << "  \"connection_retry_delay_ms\": " << config.connection_retry_delay_ms << ",\n";
    j << "  \"allow_auth_fallback\": " << (config.allow_auth_fallback ? "true" : "false") << ",\n";
    j << "  \"auth_token\": \"" << config.auth_token << "\",\n";
    j << "  \"auth_token_file\": \"" << config.auth_token_file << "\",\n";
    j << "  \"require_encryption\": " << (config.require_encryption ? "true" : "false") << ",\n";
    j << "  \"shm_threshold_bytes\": " << config.shm_threshold_bytes << ",\n";
    j << "  \"rdma_threshold_bytes\": " << config.rdma_threshold_bytes << ",\n";
    j << "  \"delta_sync_threshold\": " << config.delta_sync_threshold << ",\n";
    j << "  \"coalescing_threshold_bytes\": " << config.coalescing_threshold_bytes << ",\n";
    j << "  \"cluster_nodes\": \"" << config.cluster_nodes << "\",\n";
    j << "  \"enable_mesh_topology\": " << (config.enable_mesh_topology ? "true" : "false") << ",\n";
    j << "  \"mesh_discovery_port\": " << config.mesh_discovery_port << ",\n";
    j << "  \"enable_metrics_export\": " << (config.enable_metrics_export ? "true" : "false") << ",\n";
    j << "  \"metrics_export_interval_sec\": " << config.metrics_export_interval_sec << ",\n";
    j << "  \"log_level\": \"" << config.log_level << "\",\n";
    j << "  \"enable_hot_reload\": " << (config.enable_hot_reload ? "true" : "false") << ",\n";
    j << "  \"backup_directory\": \"" << config.backup_directory << "\",\n";
    j << "  \"enable_auto_backup\": " << (config.enable_auto_backup ? "true" : "false") << ",\n";
    j << "  \"max_backup_files\": " << config.max_backup_files << ",\n";
    j << "  \"strict_validation\": " << (config.strict_validation ? "true" : "false") << ",\n";
    j << "  \"validate_network_connectivity\": " << (config.validate_network_connectivity ? "true" : "false") << ",\n";
    j << "  \"validate_file_permissions\": " << (config.validate_file_permissions ? "true" : "false") << "\n";
    j << "}";
    return j.str();
}

std::string ConfigurationManager::configurationToYaml(const ClusterConfiguration& c) {
    std::ostringstream y;
    y << "# VGRE TCP Cluster Configuration\n";
    y << "handshake_timeout_sec: " << c.handshake_timeout_sec << "\n";
    y << "max_queue_depth: " << c.max_queue_depth << "\n";
    y << "max_packets_per_sec: " << c.max_packets_per_sec << "\n";
    y << "connection_retry_attempts: " << c.connection_retry_attempts << "\n";
    y << "connection_retry_delay_ms: " << c.connection_retry_delay_ms << "\n";
    y << "allow_auth_fallback: " << (c.allow_auth_fallback ? "true" : "false") << "\n";
    y << "auth_token: \"" << c.auth_token << "\"\n";
    y << "auth_token_file: \"" << c.auth_token_file << "\"\n";
    y << "require_encryption: " << (c.require_encryption ? "true" : "false") << "\n";
    y << "shm_threshold_bytes: " << c.shm_threshold_bytes << "\n";
    y << "rdma_threshold_bytes: " << c.rdma_threshold_bytes << "\n";
    y << "delta_sync_threshold: " << c.delta_sync_threshold << "\n";
    y << "coalescing_threshold_bytes: " << c.coalescing_threshold_bytes << "\n";
    y << "cluster_nodes: \"" << c.cluster_nodes << "\"\n";
    y << "enable_mesh_topology: " << (c.enable_mesh_topology ? "true" : "false") << "\n";
    y << "mesh_discovery_port: " << c.mesh_discovery_port << "\n";
    y << "enable_metrics_export: " << (c.enable_metrics_export ? "true" : "false") << "\n";
    y << "metrics_export_interval_sec: " << c.metrics_export_interval_sec << "\n";
    y << "log_level: \"" << c.log_level << "\"\n";
    y << "enable_hot_reload: " << (c.enable_hot_reload ? "true" : "false") << "\n";
    y << "backup_directory: \"" << c.backup_directory << "\"\n";
    y << "enable_auto_backup: " << (c.enable_auto_backup ? "true" : "false") << "\n";
    y << "max_backup_files: " << c.max_backup_files << "\n";
    y << "strict_validation: " << (c.strict_validation ? "true" : "false") << "\n";
    y << "validate_network_connectivity: " << (c.validate_network_connectivity ? "true" : "false") << "\n";
    y << "validate_file_permissions: " << (c.validate_file_permissions ? "true" : "false") << "\n";
    return y.str();
}

std::string ConfigurationManager::configurationToEnv(const ClusterConfiguration& c) {
    std::ostringstream e;
    e << "# VGRE TCP Cluster Configuration Environment Variables\n";
    e << "export VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC=" << c.handshake_timeout_sec << "\n";
    e << "export VGRE_CLUSTER_MAX_QUEUE_DEPTH=" << c.max_queue_depth << "\n";
    e << "export VGRE_CLUSTER_MAX_PACKETS_PER_SEC=" << c.max_packets_per_sec << "\n";
    e << "export VGRE_CLUSTER_CONNECTION_RETRY_ATTEMPTS=" << c.connection_retry_attempts << "\n";
    e << "export VGRE_CLUSTER_CONNECTION_RETRY_DELAY_MS=" << c.connection_retry_delay_ms << "\n";
    // Production: auth fallback is not supported.
    if (!c.auth_token.empty())
        e << "export VGRE_TCP_AUTH_TOKEN=\"" << c.auth_token << "\"\n";
    if (!c.auth_token_file.empty())
        e << "export VGRE_TCP_AUTH_TOKEN_FILE=\"" << c.auth_token_file << "\"\n";
    e << "export VGRE_REQUIRE_ENCRYPTION=" << (c.require_encryption ? "1" : "0") << "\n";
    e << "export VGRE_SHM_THRESHOLD_BYTES=" << c.shm_threshold_bytes << "\n";
    e << "export VGRE_RDMA_THRESHOLD_BYTES=" << c.rdma_threshold_bytes << "\n";
    e << "export VGRE_DELTA_SYNC_THRESHOLD=" << c.delta_sync_threshold << "\n";
    e << "export VGRE_COALESCING_THRESHOLD_BYTES=" << c.coalescing_threshold_bytes << "\n";
    if (!c.cluster_nodes.empty())
        e << "export VGRE_CLUSTER_NODES=\"" << c.cluster_nodes << "\"\n";
    e << "export VGRE_ENABLE_MESH_TOPOLOGY=" << (c.enable_mesh_topology ? "1" : "0") << "\n";
    e << "export VGRE_MESH_DISCOVERY_PORT=" << c.mesh_discovery_port << "\n";
    e << "export VGRE_ENABLE_METRICS_EXPORT=" << (c.enable_metrics_export ? "1" : "0") << "\n";
    e << "export VGRE_METRICS_EXPORT_INTERVAL_SEC=" << c.metrics_export_interval_sec << "\n";
    e << "export VGRE_LOG_LEVEL=\"" << c.log_level << "\"\n";
    e << "export VGRE_CONFIG_HOT_RELOAD=" << (c.enable_hot_reload ? "1" : "0") << "\n";
    if (!c.backup_directory.empty())
        e << "export VGRE_CONFIG_BACKUP_DIR=\"" << c.backup_directory << "\"\n";
    e << "export VGRE_CONFIG_ENABLE_AUTO_BACKUP=" << (c.enable_auto_backup ? "1" : "0") << "\n";
    e << "export VGRE_CONFIG_MAX_BACKUP_FILES=" << c.max_backup_files << "\n";
    e << "export VGRE_CONFIG_STRICT_VALIDATION=" << (c.strict_validation ? "1" : "0") << "\n";
    e << "export VGRE_CONFIG_VALIDATE_NETWORK=" << (c.validate_network_connectivity ? "1" : "0") << "\n";
    e << "export VGRE_CONFIG_VALIDATE_FILES=" << (c.validate_file_permissions ? "1" : "0") << "\n";
    return e.str();
}

// ── Documentation Generation ──

std::string ConfigurationManager::generateMarkdownDocumentation(const ClusterConfiguration& config) {
    std::ostringstream md;
    md << "# VGRE TCP Cluster Configuration Documentation\n\n";
    md << "## Network Configuration\n\n";
    md << "| Parameter | Default | Description |\n";
    md << "|-----------|---------|-------------|\n";
    md << "| `handshake_timeout_sec` | " << config.handshake_timeout_sec << " | Timeout for initial handshake in seconds |\n";
    md << "| `max_queue_depth` | " << config.max_queue_depth << " | Maximum queue depth for packet processing |\n";
    md << "| `max_packets_per_sec` | " << config.max_packets_per_sec << " | Maximum packets per second rate limit |\n";
    md << "| `connection_retry_attempts` | " << config.connection_retry_attempts << " | Number of connection retry attempts |\n";
    md << "| `connection_retry_delay_ms` | " << config.connection_retry_delay_ms << " | Delay between retry attempts in ms |\n\n";
    md << "## Security Configuration\n\n";
    md << "| Parameter | Default | Description |\n";
    md << "|-----------|---------|-------------|\n";
    md << "| `allow_auth_fallback` | " << (config.allow_auth_fallback ? "true" : "false") << " | Allow authentication fallback |\n";
    md << "| `require_encryption` | " << (config.require_encryption ? "true" : "false") << " | Require encryption for all comms |\n";
    md << "| `auth_token` | (empty) | Authentication token for cluster access |\n";
    md << "| `auth_token_file` | (empty) | Path to file containing auth token |\n\n";
    return md.str();
}

std::string ConfigurationManager::generateHtmlDocumentation(const ClusterConfiguration& config) {
    std::ostringstream h;
    h << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n";
    h << "    <meta charset=\"UTF-8\">\n";
    h << "    <title>VGRE TCP Cluster Configuration</title>\n";
    h << "</head>\n<body>\n";
    h << "    <h1>VGRE TCP Cluster Configuration</h1>\n";
    h << "    <p>All available configuration options for the VGRE TCP Cluster.</p>\n";
    h << "</body>\n</html>\n";
    return h.str();
}

// ── High-level Documentation Export ──

bool ConfigurationManager::generateConfigurationDocumentation(const std::string& output_path, const std::string& format) {
    ClusterConfiguration example;
    applyProfileDefaults(example, ConfigurationProfile::DEVELOPMENT);

    std::string content;
    if (format == "markdown") content = generateMarkdownDocumentation(example);
    else if (format == "html") content = generateHtmlDocumentation(example);
    else if (format == "text") content = "VGRE TCP Cluster Configuration Documentation\n==========================================\n\n" + generateMarkdownDocumentation(example);
    else { VGRE_LOG_ERROR("ConfigurationManager", "Unsupported doc format: " + format); return false; }

    bool ok = writeFileContent(output_path, content);
    if (ok) VGRE_LOG_INFO("ConfigurationManager", "Generated docs: " + output_path);
    return ok;
}

bool ConfigurationManager::generateExampleConfigurations(const std::string& output_dir) {
    #ifdef _WIN32
    _mkdir(output_dir.c_str());
    #else
    mkdir(output_dir.c_str(), 0755);
    #endif

    bool success = true;
    std::vector<ConfigurationProfile> profiles = {
        ConfigurationProfile::DEVELOPMENT, ConfigurationProfile::STAGING, ConfigurationProfile::PRODUCTION
    };
    std::vector<std::string> names = {"development", "staging", "production"};

    for (size_t i = 0; i < profiles.size(); ++i) {
        ClusterConfiguration cfg;
        applyProfileDefaults(cfg, profiles[i]);
        if (!saveToJsonFile(output_dir + "/vgre_cluster_" + names[i] + ".json", cfg)) success = false;
        if (!saveToYamlFile(output_dir + "/vgre_cluster_" + names[i] + ".yaml", cfg)) success = false;
        if (!writeFileContent(output_dir + "/vgre_cluster_" + names[i] + ".env", configurationToEnv(cfg))) success = false;
    }
    if (success) VGRE_LOG_INFO("ConfigurationManager", "Generated example configs in: " + output_dir);
    return success;
}

bool ConfigurationManager::exportResolvedConfiguration(const std::string& output_path,
                                                       const std::string& format,
                                                       bool include_defaults) {
    ClusterConfiguration current = getCurrentConfiguration();
    std::string content;
    if      (format == "json") content = configurationToJson(current);
    else if (format == "yaml") content = configurationToYaml(current);
    else if (format == "env")  content = configurationToEnv(current);
    else { VGRE_LOG_ERROR("ConfigurationManager", "Unsupported export format: " + format); return false; }

    bool ok = writeFileContent(output_path, content);
    if (ok) VGRE_LOG_INFO("ConfigurationManager", "Exported config to: " + output_path);
    return ok;
}

} // namespace advanced
} // namespace vgre