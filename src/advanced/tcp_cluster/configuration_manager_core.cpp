// Configuration Manager Core — static state, env parsing, profiles, change tracking.

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <cstring>

namespace vgre {
namespace advanced {

// ── Static Member Definitions ──

std::mutex ConfigurationManager::config_mutex_;
ClusterConfiguration ConfigurationManager::current_config_;
std::map<uint64_t, ConfigurationManager::ConfigurationChangeCallback> ConfigurationManager::change_callbacks_;
std::atomic<uint64_t> ConfigurationManager::next_callback_id_{1};
std::atomic<bool> ConfigurationManager::monitoring_active_{false};
std::thread ConfigurationManager::monitoring_thread_;

// ── getEnvVar Specializations ──

template<>
int ConfigurationManager::getEnvVar<int>(const char* name, int default_value) {
    const char* env_val = vgre_get_config(name);
    if (!env_val || env_val[0] == '\0') return default_value;
    try {
        return std::stoi(env_val);
    } catch (const std::exception&) {
        VGRE_LOG_WARN("ConfigurationManager",
            "Invalid integer for " + std::string(name) + ": " + env_val +
            ", using default " + std::to_string(default_value));
        return default_value;
    }
}

template<>
size_t ConfigurationManager::getEnvVar<size_t>(const char* name, size_t default_value) {
    const char* env_val = vgre_get_config(name);
    if (!env_val || env_val[0] == '\0') return default_value;
    try {
        return static_cast<size_t>(std::stoull(env_val));
    } catch (const std::exception&) {
        VGRE_LOG_WARN("ConfigurationManager",
            "Invalid size_t for " + std::string(name) + ": " + env_val +
            ", using default " + std::to_string(default_value));
        return default_value;
    }
}

template<>
double ConfigurationManager::getEnvVar<double>(const char* name, double default_value) {
    const char* env_val = vgre_get_config(name);
    if (!env_val || env_val[0] == '\0') return default_value;
    try {
        return std::stod(env_val);
    } catch (const std::exception&) {
        VGRE_LOG_WARN("ConfigurationManager",
            "Invalid double for " + std::string(name) + ": " + env_val +
            ", using default " + std::to_string(default_value));
        return default_value;
    }
}

template<>
bool ConfigurationManager::getEnvVar<bool>(const char* name, bool default_value) {
    const char* env_val = vgre_get_config(name);
    if (!env_val || env_val[0] == '\0') return default_value;
    std::string val_str(env_val);
    std::transform(val_str.begin(), val_str.end(), val_str.begin(), ::tolower);
    if (val_str == "1" || val_str == "true" || val_str == "yes" || val_str == "on")
        return true;
    if (val_str == "0" || val_str == "false" || val_str == "no" || val_str == "off")
        return false;
    return default_value;
}

template<>
std::string ConfigurationManager::getEnvVar<std::string>(const char* name,
                                                         std::string default_value) {
    const char* env_val = vgre_get_config(name);
    if (!env_val || env_val[0] == '\0') return default_value;
    return std::string(env_val);
}

// ── Core Configuration Management ──

ClusterConfiguration ConfigurationManager::getClusterConfiguration() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    ClusterConfiguration config;

    ConfigurationProfile profile = getConfigurationProfile();
    applyProfileDefaults(config, profile);

    std::string config_file = getEnvVar<std::string>("VGRE_CONFIG_FILE", "");
    if (!config_file.empty()) {
        config.config_file_path = config_file;
        if (config_file.size() >= 5 && config_file.substr(config_file.size() - 5) == ".json") {
            if (!loadFromJsonFile(config_file, config))
                VGRE_LOG_WARN("ConfigurationManager", "Failed to load JSON config from: " + config_file);
        } else if ((config_file.size() >= 5 && config_file.substr(config_file.size() - 5) == ".yaml") ||
                   (config_file.size() >= 4 && config_file.substr(config_file.size() - 4) == ".yml")) {
            if (!loadFromYamlFile(config_file, config))
                VGRE_LOG_WARN("ConfigurationManager", "Failed to load YAML config from: " + config_file);
        } else {
            VGRE_LOG_WARN("ConfigurationManager", "Unknown config file format: " + config_file);
        }
        config.enable_hot_reload = getEnvVar<bool>("VGRE_CONFIG_HOT_RELOAD", false);
        if (config.enable_hot_reload)
            config.last_modified_time = getFileModificationTime(config_file);
    }

    // Environment overrides (highest precedence)
    config.handshake_timeout_sec = getEnvVar<int>("VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC", config.handshake_timeout_sec);
    config.max_queue_depth = getEnvVar<size_t>("VGRE_CLUSTER_MAX_QUEUE_DEPTH", config.max_queue_depth);
    config.max_packets_per_sec = getEnvVar<size_t>("VGRE_CLUSTER_MAX_PACKETS_PER_SEC", config.max_packets_per_sec);
    config.connection_retry_attempts = getEnvVar<int>("VGRE_CLUSTER_CONNECTION_RETRY_ATTEMPTS", config.connection_retry_attempts);
    config.connection_retry_delay_ms = getEnvVar<int>("VGRE_CLUSTER_CONNECTION_RETRY_DELAY_MS", config.connection_retry_delay_ms);

    // Production: auth fallback is not supported.
    config.auth_token = getEnvVar<std::string>("VGRE_TCP_AUTH_TOKEN", config.auth_token);
    config.auth_token_file = getEnvVar<std::string>("VGRE_TCP_AUTH_TOKEN_FILE", config.auth_token_file);
    config.require_encryption = getEnvVar<bool>("VGRE_REQUIRE_ENCRYPTION", config.require_encryption);

    config.shm_threshold_bytes = getEnvVar<size_t>("VGRE_SHM_THRESHOLD_BYTES", config.shm_threshold_bytes);
    config.rdma_threshold_bytes = getEnvVar<size_t>("VGRE_RDMA_THRESHOLD_BYTES", config.rdma_threshold_bytes);
    config.delta_sync_threshold = getEnvVar<double>("VGRE_DELTA_SYNC_THRESHOLD", config.delta_sync_threshold);
    config.coalescing_threshold_bytes = getEnvVar<size_t>("VGRE_COALESCING_THRESHOLD_BYTES", config.coalescing_threshold_bytes);

    config.cluster_nodes = getEnvVar<std::string>("VGRE_CLUSTER_NODES", config.cluster_nodes);
    config.enable_mesh_topology = getEnvVar<bool>("VGRE_ENABLE_MESH_TOPOLOGY", config.enable_mesh_topology);
    config.mesh_discovery_port = getEnvVar<int>("VGRE_MESH_DISCOVERY_PORT", config.mesh_discovery_port);

    config.enable_metrics_export = getEnvVar<bool>("VGRE_ENABLE_METRICS_EXPORT", config.enable_metrics_export);
    config.metrics_export_interval_sec = getEnvVar<int>("VGRE_METRICS_EXPORT_INTERVAL_SEC", config.metrics_export_interval_sec);
    config.log_level = getEnvVar<std::string>("VGRE_LOG_LEVEL", config.log_level);

    config.backup_directory = getEnvVar<std::string>("VGRE_CONFIG_BACKUP_DIR", config.backup_directory);
    config.enable_auto_backup = getEnvVar<bool>("VGRE_CONFIG_ENABLE_AUTO_BACKUP", config.enable_auto_backup);
    config.max_backup_files = getEnvVar<int>("VGRE_CONFIG_MAX_BACKUP_FILES", config.max_backup_files);

    config.strict_validation = getEnvVar<bool>("VGRE_CONFIG_STRICT_VALIDATION", config.strict_validation);
    config.validate_network_connectivity = getEnvVar<bool>("VGRE_CONFIG_VALIDATE_NETWORK", config.validate_network_connectivity);
    config.validate_file_permissions = getEnvVar<bool>("VGRE_CONFIG_VALIDATE_FILES", config.validate_file_permissions);

    current_config_ = config;
    return config;
}

bool ConfigurationManager::updateConfiguration(const ClusterConfiguration& new_config, const std::string& source) {
    std::lock_guard<std::mutex> lock(config_mutex_);

    auto validation = validateConfiguration(new_config);
    if (!validation.is_valid) {
        VGRE_LOG_ERROR("ConfigurationManager", "Configuration update rejected — validation errors");
        for (const auto& error : validation.errors)
            VGRE_LOG_ERROR("ConfigurationManager", "  " + error);
        return false;
    }

    if (new_config.enable_auto_backup && !current_config_.config_file_path.empty()) {
        std::string backup_path = createConfigurationBackup();
        if (!backup_path.empty())
            VGRE_LOG_INFO("ConfigurationManager", "Backup created: " + backup_path);
    }

    auto change_events = compareConfigurations(current_config_, new_config, source);
    current_config_ = new_config;

    for (const auto& event : change_events)
        notifyConfigurationChange(event);

    VGRE_LOG_INFO("ConfigurationManager", "Configuration updated from " + source +
                  " (" + std::to_string(change_events.size()) + " changes)");
    return true;
}

ClusterConfiguration ConfigurationManager::getCurrentConfiguration() {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return current_config_;
}

// ── Change Notification ──

uint64_t ConfigurationManager::registerChangeCallback(ConfigurationChangeCallback callback) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    uint64_t id = next_callback_id_.fetch_add(1);
    change_callbacks_[id] = callback;
    return id;
}

void ConfigurationManager::unregisterChangeCallback(uint64_t callback_id) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    change_callbacks_.erase(callback_id);
}

void ConfigurationManager::notifyConfigurationChange(const ConfigurationChangeEvent& event) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    for (const auto& [id, callback] : change_callbacks_) {
        try { callback(event); }
        catch (const std::exception& e) {
            VGRE_LOG_ERROR("ConfigurationManager", "Callback failed: " + std::string(e.what()));
        }
    }
}

// ── Profiles ──

ConfigurationProfile ConfigurationManager::getConfigurationProfile() {
    std::string s = getEnvVar<std::string>("VGRE_DEPLOYMENT_PROFILE", "development");
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "production" || s == "prod") return ConfigurationProfile::PRODUCTION;
    if (s == "staging"    || s == "stage") return ConfigurationProfile::STAGING;
    if (s == "development"|| s == "dev")   return ConfigurationProfile::DEVELOPMENT;
    return ConfigurationProfile::CUSTOM;
}

void ConfigurationManager::applyProfileDefaults(ClusterConfiguration& config, ConfigurationProfile profile) {
    switch (profile) {
        case ConfigurationProfile::PRODUCTION:
            config.handshake_timeout_sec = 10; config.max_queue_depth = 2048;
            config.max_packets_per_sec = 50000; config.connection_retry_attempts = 5;
            config.connection_retry_delay_ms = 2000; config.allow_auth_fallback = false;
            config.require_encryption = true; config.enable_metrics_export = true;
            config.metrics_export_interval_sec = 30; config.log_level = "INFO";
            config.strict_validation = true; config.validate_network_connectivity = true;
            config.validate_file_permissions = true; config.enable_auto_backup = true;
            config.max_backup_files = 10;
            break;
        case ConfigurationProfile::STAGING:
            config.handshake_timeout_sec = 8; config.max_queue_depth = 1536;
            config.max_packets_per_sec = 25000; config.connection_retry_attempts = 4;
            config.connection_retry_delay_ms = 1500; config.allow_auth_fallback = false;
            config.require_encryption = false; config.enable_metrics_export = true;
            config.metrics_export_interval_sec = 60; config.log_level = "DEBUG";
            config.strict_validation = false; config.validate_network_connectivity = true;
            config.validate_file_permissions = false; config.enable_auto_backup = true;
            config.max_backup_files = 5;
            break;
        case ConfigurationProfile::DEVELOPMENT:
            config.handshake_timeout_sec = 5; config.max_queue_depth = 1024;
            config.max_packets_per_sec = 10000; config.connection_retry_attempts = 3;
            config.connection_retry_delay_ms = 1000; config.allow_auth_fallback = false;
            config.require_encryption = false; config.enable_metrics_export = false;
            config.metrics_export_interval_sec = 120; config.log_level = "DEBUG";
            config.strict_validation = false; config.validate_network_connectivity = false;
            config.validate_file_permissions = false; config.enable_auto_backup = false;
            config.max_backup_files = 3;
            break;
        case ConfigurationProfile::CUSTOM:
            break;
    }
    config.profile = profile;
}

// ── Configuration Diff ──

std::vector<ConfigurationChangeEvent> ConfigurationManager::compareConfigurations(
    const ClusterConfiguration& old_config,
    const ClusterConfiguration& new_config,
    const std::string& source) {

    std::vector<ConfigurationChangeEvent> events;
    auto emit = [&](const std::string& key, const std::string& old_val, const std::string& new_val) {
        if (old_val != new_val) {
            ConfigurationChangeEvent ev;
            ev.parameter_name = key; ev.old_value = old_val; ev.new_value = new_val;
            ev.source = source; ev.timestamp_ms = TimeUtils::getCurrentTimestampMs();
            events.push_back(ev);
        }
    };

    emit("handshake_timeout_sec", std::to_string(old_config.handshake_timeout_sec), std::to_string(new_config.handshake_timeout_sec));
    emit("max_queue_depth", std::to_string(old_config.max_queue_depth), std::to_string(new_config.max_queue_depth));
    emit("max_packets_per_sec", std::to_string(old_config.max_packets_per_sec), std::to_string(new_config.max_packets_per_sec));
    emit("connection_retry_attempts", std::to_string(old_config.connection_retry_attempts), std::to_string(new_config.connection_retry_attempts));
    emit("connection_retry_delay_ms", std::to_string(old_config.connection_retry_delay_ms), std::to_string(new_config.connection_retry_delay_ms));
    emit("allow_auth_fallback", old_config.allow_auth_fallback ? "true" : "false", new_config.allow_auth_fallback ? "true" : "false");
    emit("auth_token", old_config.auth_token.empty() ? "(empty)" : "(set)", new_config.auth_token.empty() ? "(empty)" : "(set)");
    emit("auth_token_file", old_config.auth_token_file, new_config.auth_token_file);
    emit("require_encryption", old_config.require_encryption ? "true" : "false", new_config.require_encryption ? "true" : "false");
    emit("shm_threshold_bytes", std::to_string(old_config.shm_threshold_bytes), std::to_string(new_config.shm_threshold_bytes));
    emit("rdma_threshold_bytes", std::to_string(old_config.rdma_threshold_bytes), std::to_string(new_config.rdma_threshold_bytes));
    emit("delta_sync_threshold", std::to_string(old_config.delta_sync_threshold), std::to_string(new_config.delta_sync_threshold));
    emit("coalescing_threshold_bytes", std::to_string(old_config.coalescing_threshold_bytes), std::to_string(new_config.coalescing_threshold_bytes));
    emit("cluster_nodes", old_config.cluster_nodes, new_config.cluster_nodes);
    emit("enable_mesh_topology", old_config.enable_mesh_topology ? "true" : "false", new_config.enable_mesh_topology ? "true" : "false");
    emit("mesh_discovery_port", std::to_string(old_config.mesh_discovery_port), std::to_string(new_config.mesh_discovery_port));
    emit("enable_metrics_export", old_config.enable_metrics_export ? "true" : "false", new_config.enable_metrics_export ? "true" : "false");
    emit("metrics_export_interval_sec", std::to_string(old_config.metrics_export_interval_sec), std::to_string(new_config.metrics_export_interval_sec));
    emit("log_level", old_config.log_level, new_config.log_level);
    emit("enable_hot_reload", old_config.enable_hot_reload ? "true" : "false", new_config.enable_hot_reload ? "true" : "false");
    emit("backup_directory", old_config.backup_directory, new_config.backup_directory);
    emit("enable_auto_backup", old_config.enable_auto_backup ? "true" : "false", new_config.enable_auto_backup ? "true" : "false");
    emit("max_backup_files", std::to_string(old_config.max_backup_files), std::to_string(new_config.max_backup_files));

    return events;
}

} // namespace advanced
} // namespace vgre
