/**
 * Configuration Manager Hot Reload and Monitoring Operations
 * 
 * Handles hot reload functionality and configuration file monitoring
 * for the ConfigurationManager.
 */

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <chrono>
#include <thread>

namespace vgre {
namespace advanced {

bool ConfigurationManager::hasConfigurationChanged(const std::string& file_path, uint64_t last_modified_time) {
    uint64_t current_time = getFileModificationTime(file_path);
    return current_time > last_modified_time;
}

bool ConfigurationManager::reloadIfChanged(ClusterConfiguration& config) {
    if (!config.enable_hot_reload || config.config_file_path.empty()) {
        return false;
    }
    
    if (!hasConfigurationChanged(config.config_file_path, config.last_modified_time)) {
        return false;
    }
    
    VGRE_LOG_INFO("ConfigurationManager", "Configuration file changed, reloading: " + config.config_file_path);
    
    ClusterConfiguration new_config = config;
    bool success = false;
    
    // Determine file format and reload
    std::string file_path = config.config_file_path;
    if (file_path.size() >= 5 && file_path.substr(file_path.size() - 5) == ".json") {
        success = loadFromJsonFile(file_path, new_config);
    } else if ((file_path.size() >= 5 && file_path.substr(file_path.size() - 5) == ".yaml") ||
               (file_path.size() >= 4 && file_path.substr(file_path.size() - 4) == ".yml")) {
        success = loadFromYamlFile(file_path, new_config);
    }
    
    if (success) {
        // Override with environment variables (they take precedence)
        ClusterConfiguration env_config = getClusterConfiguration();
        
        // Apply environment overrides to the reloaded config
        new_config.handshake_timeout_sec = env_config.handshake_timeout_sec;
        new_config.max_queue_depth = env_config.max_queue_depth;
        new_config.max_packets_per_sec = env_config.max_packets_per_sec;
        new_config.connection_retry_attempts = env_config.connection_retry_attempts;
        new_config.connection_retry_delay_ms = env_config.connection_retry_delay_ms;
        new_config.allow_auth_fallback = env_config.allow_auth_fallback;
        new_config.auth_token = env_config.auth_token;
        new_config.auth_token_file = env_config.auth_token_file;
        new_config.require_encryption = env_config.require_encryption;
        
        // Update configuration
        success = updateConfiguration(new_config, "hot_reload");
        config = new_config;
    }
    
    if (!success) {
        VGRE_LOG_ERROR("ConfigurationManager", "Failed to reload configuration from: " + file_path);
    }
    
    return success;
}

bool ConfigurationManager::startConfigurationMonitoring(int check_interval_ms) {
    if (state().monitoring_active.load()) {
        VGRE_LOG_WARN("ConfigurationManager", "Configuration monitoring is already active");
        return false;
    }
    
    state().monitoring_active.store(true);
    
    state().monitoring_thread = std::thread([check_interval_ms]() {
        VGRE_LOG_INFO("ConfigurationManager", "Started configuration monitoring (interval: " + 
                      std::to_string(check_interval_ms) + "ms)");
        
        while (state().monitoring_active.load()) {
            try {
                ClusterConfiguration config = getCurrentConfiguration();
                
                if (config.enable_hot_reload && !config.config_file_path.empty()) {
                    ClusterConfiguration updated_config = config;
                    if (reloadIfChanged(updated_config)) {
                        VGRE_LOG_INFO("ConfigurationManager", "Configuration hot-reloaded successfully");
                    }
                }
                std::unique_lock<std::mutex> lock(state().state().monitoring_cvmutex);
                state().monitoring_cv.wait_for(lock, std::chrono::milliseconds(check_interval_ms), []() { return !state().monitoring_active.load(); });
            } catch (const std::exception& e) {
                VGRE_LOG_ERROR("ConfigurationManager", "Configuration monitoring error: " + std::string(e.what()));
                std::unique_lock<std::mutex> lock(state().state().monitoring_cvmutex);
                state().monitoring_cv.wait_for(lock, std::chrono::milliseconds(check_interval_ms * 2), []() { return !state().monitoring_active.load(); }); // Back off on error
            }
        }
        
        VGRE_LOG_INFO("ConfigurationManager", "Configuration monitoring stopped");
    });
    
    return true;
}

void ConfigurationManager::stopConfigurationMonitoring() {
    if (state().monitoring_active.load()) {
        state().monitoring_active.store(false);
        state().monitoring_cv.notify_all();
        if (state().monitoring_thread.joinable()) {
            state().monitoring_thread.join();
        }
        VGRE_LOG_INFO("ConfigurationManager", "Configuration monitoring stopped");
    }
}

} // namespace advanced
} // namespace vgre