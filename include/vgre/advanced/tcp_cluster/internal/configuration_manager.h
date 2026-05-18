#pragma once

#include "vgre/common/error_codes.h"
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <map>
#include <functional>

namespace vgre {
namespace advanced {

class TCPClusterManager;

// ── Configuration Types ──

enum class ConfigurationProfile {
    DEVELOPMENT,
    STAGING,
    PRODUCTION,
    CUSTOM
};

enum class ConfigurationChangeType {
    NETWORK_SETTINGS,
    SECURITY_SETTINGS,
    TRANSPORT_SETTINGS,
    MONITORING_SETTINGS,
    PROFILE_CHANGE,
    FILE_RELOAD,
    VALIDATION_ERROR
};

struct ConfigurationChangeEvent {
    ConfigurationChangeType type;
    std::string parameter_name;
    std::string old_value;
    std::string new_value;
    uint64_t timestamp_ms;
    std::string source; // "environment", "file", "api"
};

struct ClusterConfiguration {
    // Network configuration
    int handshake_timeout_sec = 5;
    size_t max_queue_depth = 1024;
    size_t max_packets_per_sec = 10000;
    int connection_retry_attempts = 3;
    int connection_retry_delay_ms = 1000;
    
    // Security configuration
    bool allow_auth_fallback = true;
    std::string auth_token;
    std::string auth_token_file;
    bool require_encryption = false;
    
    // Transport configuration
    size_t shm_threshold_bytes = 64 * 1024;
    size_t rdma_threshold_bytes = 256 * 1024;
    double delta_sync_threshold = 0.3;
    size_t coalescing_threshold_bytes = 4 * 1024;
    
    // Mesh topology configuration
    std::string cluster_nodes;
    bool enable_mesh_topology = false;
    int mesh_discovery_port = 7778;
    
    // Monitoring configuration
    bool enable_metrics_export = true;
    int metrics_export_interval_sec = 60;
    std::string log_level = "INFO";
    
    // Profile and file configuration
    ConfigurationProfile profile = ConfigurationProfile::DEVELOPMENT;
    std::string config_file_path;
    bool enable_hot_reload = false;
    uint64_t last_modified_time = 0;
    
    // Backup and recovery configuration
    std::string backup_directory;
    bool enable_auto_backup = true;
    int max_backup_files = 10;
    
    // Advanced validation configuration
    bool strict_validation = false;
    bool validate_network_connectivity = false;
    bool validate_file_permissions = true;
};

struct ConfigurationValidationResult {
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<std::string> suggestions;
    std::vector<std::string> security_warnings;
    std::vector<std::string> performance_warnings;
};

/**
 * Enhanced configuration management with file support, hot reloading, 
 * validation, deployment profiles, thread safety, and change notifications.
 */
class ConfigurationManager {
public:
    using ConfigurationChangeCallback = std::function<void(const ConfigurationChangeEvent&)>;

    // ── Static API (Authoritative) ──
    
    static ClusterConfiguration getClusterConfiguration();
    static bool updateConfiguration(const ClusterConfiguration& new_config, const std::string& source = "api");
    static ClusterConfiguration getCurrentConfiguration();
    
    // ── Callbacks ──
    static uint64_t registerChangeCallback(ConfigurationChangeCallback callback);
    static void unregisterChangeCallback(uint64_t callback_id);
    
    // ── File Load/Save (JSON/YAML) ──
    static bool loadFromJsonFile(const std::string& file_path, ClusterConfiguration& config);
    static bool loadFromYamlFile(const std::string& file_path, ClusterConfiguration& config);
    static bool saveToJsonFile(const std::string& file_path, const ClusterConfiguration& config);
    static bool saveToYamlFile(const std::string& file_path, const ClusterConfiguration& config);
    
    // ── Backup/Recovery ──
    static std::string createConfigurationBackup(const std::string& backup_path = "");
    static bool restoreConfigurationFromBackup(const std::string& backup_path);
    static std::vector<std::string> listConfigurationBackups(const std::string& backup_directory = "");
    static int cleanupOldBackups(const std::string& backup_directory = "", int max_backups = 10);
    
    // ── Monitoring & Reload ──
    static bool hasConfigurationChanged(const std::string& file_path, uint64_t last_modified_time);
    static bool reloadIfChanged(ClusterConfiguration& config);
    static bool startConfigurationMonitoring(int check_interval_ms = 5000);
    static void stopConfigurationMonitoring();
    
    // ── Profiles & Validation ──
    static ConfigurationProfile getConfigurationProfile();
    static void applyProfileDefaults(ClusterConfiguration& config, ConfigurationProfile profile);
    static ConfigurationValidationResult validateConfiguration(const ClusterConfiguration& config);
    static ConfigurationValidationResult validateNetworkConnectivity(const ClusterConfiguration& config);
    static ConfigurationValidationResult validateFilePermissions(const ClusterConfiguration& config);
    
    // ── Documentation ──
    static bool generateConfigurationDocumentation(const std::string& output_path, const std::string& format = "markdown");
    static bool generateExampleConfigurations(const std::string& output_dir);
    static bool exportResolvedConfiguration(const std::string& output_path, const std::string& format = "json", bool include_defaults = false);
    static std::string generateMarkdownDocumentation(const ClusterConfiguration& config);
    static std::string generateHtmlDocumentation(const ClusterConfiguration& config);

private:
    static std::mutex config_mutex_;
    static ClusterConfiguration current_config_;
    static std::map<uint64_t, ConfigurationChangeCallback> change_callbacks_;
    static std::atomic<uint64_t> next_callback_id_;
    static std::atomic<bool> monitoring_active_;
    static std::thread monitoring_thread_;
    static std::mutex monitoring_cv_mutex_;
    static std::condition_variable monitoring_cv_;

    static void notifyConfigurationChange(const ConfigurationChangeEvent& event);
    static std::vector<ConfigurationChangeEvent> compareConfigurations(const ClusterConfiguration& old_cfg, const ClusterConfiguration& new_cfg, const std::string& source = "api");
    
    // Internal parser helpers
    static bool parseJsonConfiguration(const std::string& json_content, ClusterConfiguration& config);
    static bool parseYamlConfiguration(const std::string& yaml_content, ClusterConfiguration& config);
    static std::string configurationToJson(const ClusterConfiguration& config);
    static std::string configurationToYaml(const ClusterConfiguration& config);
    static std::string configurationToEnv(const ClusterConfiguration& config);
    
    static uint64_t getFileModificationTime(const std::string& file_path);
    static bool readFileContent(const std::string& file_path, std::string& content);
    static bool writeFileContent(const std::string& file_path, const std::string& content);

public:
    // Template for env var loading
    template<typename T> static T getEnvVar(const char* name, T default_value);
    
    // Template for config value validation
    template<typename T> static void validateConfigValue(const std::string& name, T value, T min_value, T max_value, ConfigurationValidationResult& result);
};

} // namespace advanced
} // namespace vgre
