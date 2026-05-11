/**
 * Configuration Manager Validation Operations
 * 
 * Handles configuration validation, network connectivity testing,
 * and file permission validation for the ConfigurationManager.
 */

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace vgre {
namespace advanced {

// Template method implementations for validateConfigValue (must be defined before use)
template<>
void ConfigurationManager::validateConfigValue<int>(const std::string& name, int value, int min_value, int max_value,
                                                   ConfigurationValidationResult& result) {
    if (value < min_value || value > max_value) {
        result.errors.push_back(name + " value " + std::to_string(value) + 
                               " is out of range [" + std::to_string(min_value) + 
                               ", " + std::to_string(max_value) + "]");
        result.suggestions.push_back("Set " + name + " to a value between " + 
                                   std::to_string(min_value) + " and " + std::to_string(max_value));
    }
}

template<>
void ConfigurationManager::validateConfigValue<size_t>(const std::string& name, size_t value, size_t min_value, size_t max_value,
                                                      ConfigurationValidationResult& result) {
    if (value < min_value || value > max_value) {
        result.errors.push_back(name + " value " + std::to_string(value) + 
                               " is out of range [" + std::to_string(min_value) + 
                               ", " + std::to_string(max_value) + "]");
        result.suggestions.push_back("Set " + name + " to a value between " + 
                                   std::to_string(min_value) + " and " + std::to_string(max_value));
    }
}

template<>
void ConfigurationManager::validateConfigValue<double>(const std::string& name, double value, double min_value, double max_value,
                                                      ConfigurationValidationResult& result) {
    if (value < min_value || value > max_value) {
        result.errors.push_back(name + " value " + std::to_string(value) + 
                               " is out of range [" + std::to_string(min_value) + 
                               ", " + std::to_string(max_value) + "]");
        result.suggestions.push_back("Set " + name + " to a value between " + 
                                   std::to_string(min_value) + " and " + std::to_string(max_value));
    }
}

ConfigurationValidationResult ConfigurationManager::validateConfiguration(const ClusterConfiguration& config) {
    ConfigurationValidationResult result;
    
    // Validate timeout values
    validateConfigValue("handshake_timeout_sec", config.handshake_timeout_sec, 1, 300, result);
    validateConfigValue("connection_retry_delay_ms", config.connection_retry_delay_ms, 100, 60000, result);
    validateConfigValue("metrics_export_interval_sec", config.metrics_export_interval_sec, 1, 3600, result);
    
    // Validate queue and rate limits
    validateConfigValue("max_queue_depth", config.max_queue_depth, static_cast<size_t>(64), static_cast<size_t>(1048576), result);
    validateConfigValue("max_packets_per_sec", config.max_packets_per_sec, static_cast<size_t>(100), static_cast<size_t>(1000000), result);
    validateConfigValue("connection_retry_attempts", config.connection_retry_attempts, 1, 20, result);
    
    // Validate transport thresholds
    validateConfigValue("shm_threshold_bytes", config.shm_threshold_bytes, static_cast<size_t>(1024), static_cast<size_t>(1073741824), result);
    validateConfigValue("rdma_threshold_bytes", config.rdma_threshold_bytes, static_cast<size_t>(4096), static_cast<size_t>(1073741824), result);
    validateConfigValue("delta_sync_threshold", config.delta_sync_threshold, 0.01, 1.0, result);
    validateConfigValue("coalescing_threshold_bytes", config.coalescing_threshold_bytes, static_cast<size_t>(64), static_cast<size_t>(65536), result);
    
    // Validate mesh topology settings
    validateConfigValue("mesh_discovery_port", config.mesh_discovery_port, 1024, 65535, result);
    validateConfigValue("max_backup_files", config.max_backup_files, 1, 100, result);
    
    // Validate log level
    std::vector<std::string> valid_log_levels = {"DEBUG", "INFO", "WARN", "ERROR"};
    if (std::find(valid_log_levels.begin(), valid_log_levels.end(), config.log_level) == valid_log_levels.end()) {
        result.errors.push_back("Invalid log_level: " + config.log_level + " (must be DEBUG, INFO, WARN, or ERROR)");
        result.suggestions.push_back("Set VGRE_LOG_LEVEL to one of: DEBUG, INFO, WARN, ERROR");
    }
    
    // Validate cluster nodes format if specified
    if (!config.cluster_nodes.empty()) {
        std::istringstream ss(config.cluster_nodes);
        std::string node;
        while (std::getline(ss, node, ',')) {
            // Trim whitespace
            node.erase(0, node.find_first_not_of(" \t"));
            node.erase(node.find_last_not_of(" \t") + 1);
            
            // Basic IP:port validation
            size_t colon_pos = node.find(':');
            if (colon_pos == std::string::npos) {
                result.warnings.push_back("Cluster node missing port: " + node + " (assuming default port 7777)");
            } else {
                std::string port_str = node.substr(colon_pos + 1);
                try {
                    int port = std::stoi(port_str);
                    if (port < 1024 || port > 65535) {
                        result.errors.push_back("Invalid port in cluster node: " + node + " (port must be 1024-65535)");
                    }
                } catch (const std::exception&) {
                    result.errors.push_back("Invalid port format in cluster node: " + node);
                }
            }
        }
    }
    
    // Validate file paths if specified
    if (!config.config_file_path.empty()) {
        std::ifstream file(config.config_file_path);
        if (!file.good()) {
            result.warnings.push_back("Configuration file not accessible: " + config.config_file_path);
        }
    }
    
    if (!config.auth_token_file.empty()) {
        std::ifstream file(config.auth_token_file);
        if (!file.good()) {
            result.errors.push_back("Auth token file not accessible: " + config.auth_token_file);
            result.suggestions.push_back("Ensure auth token file exists and is readable");
        }
    }
    
    if (!config.backup_directory.empty()) {
        // Check if backup directory exists or can be created
        #ifdef _WIN32
        DWORD attrs = GetFileAttributes(config.backup_directory.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES) {
            result.warnings.push_back("Backup directory does not exist: " + config.backup_directory + " (will be created if needed)");
        }
        #else
        struct stat st;
        if (stat(config.backup_directory.c_str(), &st) != 0) {
            result.warnings.push_back("Backup directory does not exist: " + config.backup_directory + " (will be created if needed)");
        }
        #endif
    }
    
    // Security validation
    if (config.require_encryption && config.auth_token.empty() && config.auth_token_file.empty()) {
        result.errors.push_back("Encryption required but no auth token specified");
        result.suggestions.push_back("Set VGRE_TCP_AUTH_TOKEN or VGRE_TCP_AUTH_TOKEN_FILE");
    }
    
    if (config.allow_auth_fallback && config.auth_token.empty() && config.auth_token_file.empty()) {
        result.warnings.push_back("Auth fallback requested but is not supported in production builds");
        result.suggestions.push_back("Set VGRE_TCP_AUTH_TOKEN or VGRE_TCP_AUTH_TOKEN_FILE");
    }
    
    // Performance warnings
    if (config.max_queue_depth < 512) {
        result.performance_warnings.push_back("Low max_queue_depth may limit throughput under high load");
    }
    
    if (config.max_packets_per_sec < 1000) {
        result.performance_warnings.push_back("Low max_packets_per_sec may limit cluster performance");
    }
    
    if (config.shm_threshold_bytes > config.rdma_threshold_bytes) {
        result.warnings.push_back("SHM threshold is higher than RDMA threshold - RDMA may never be used");
    }
    
    // Set validation result
    result.is_valid = result.errors.empty();
    
    if (!result.is_valid) {
        VGRE_LOG_ERROR("ConfigurationManager", "Configuration validation failed with " + 
                       std::to_string(result.errors.size()) + " errors");
    } else if (!result.warnings.empty() || !result.performance_warnings.empty()) {
        VGRE_LOG_WARN("ConfigurationManager", "Configuration validation passed with " + 
                      std::to_string(result.warnings.size()) + " warnings and " +
                      std::to_string(result.performance_warnings.size()) + " performance warnings");
    } else {
        VGRE_LOG_INFO("ConfigurationManager", "Configuration validation passed successfully");
    }
    
    return result;
}

ConfigurationValidationResult ConfigurationManager::validateNetworkConnectivity(const ClusterConfiguration& config) {
    ConfigurationValidationResult result;
    
    if (!config.validate_network_connectivity) {
        result.is_valid = true;
        return result;
    }
    
    VGRE_LOG_INFO("ConfigurationManager", "Validating network connectivity for cluster nodes");
    
    // Parse cluster nodes and test connectivity
    if (!config.cluster_nodes.empty()) {
        std::istringstream ss(config.cluster_nodes);
        std::string node;
        int tested_nodes = 0;
        int successful_connections = 0;
        
        while (std::getline(ss, node, ',')) {
            // Trim whitespace
            node.erase(0, node.find_first_not_of(" \t"));
            node.erase(node.find_last_not_of(" \t") + 1);
            
            if (node.empty()) continue;
            
            tested_nodes++;
            
            // Parse IP and port
            std::string ip = node;
            int port = 7777; // Default port
            
            size_t colon_pos = node.find(':');
            if (colon_pos != std::string::npos) {
                ip = node.substr(0, colon_pos);
                try {
                    port = std::stoi(node.substr(colon_pos + 1));
                } catch (const std::exception&) {
                    result.errors.push_back("Invalid port format in node: " + node);
                    continue;
                }
            }
            
            // Test connectivity (simple socket connection test)
            #ifdef _WIN32
            WSADATA wsaData;
            if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
                result.errors.push_back("Failed to initialize Winsock for connectivity test");
                continue;
            }
            #endif
            
            vgre_socket_t test_socket = socket(AF_INET, SOCK_STREAM, 0);
            if (test_socket == vgre::common::VGRE_INVALID_SOCKET) {
                result.warnings.push_back("Failed to create test socket for node: " + node);
                #ifdef _WIN32
                WSACleanup();
                #endif
                continue;
            }
            
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            
            #ifdef _WIN32
            addr.sin_addr.s_addr = inet_addr(ip.c_str());
            #else
            if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
                result.warnings.push_back("Invalid IP address format: " + ip);
                close(test_socket);
                continue;
            }
            #endif
            
            // Set socket timeout for quick test
            #ifdef _WIN32
            DWORD timeout = 2000; // 2 seconds
            setsockopt(test_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
            setsockopt(test_socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
            #else
            struct timeval timeout;
            timeout.tv_sec = 2;
            timeout.tv_usec = 0;
            setsockopt(test_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(test_socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            #endif
            
            // Attempt connection
            if (connect(test_socket, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                successful_connections++;
                VGRE_LOG_DEBUG("ConfigurationManager", "Successfully connected to node: " + node);
            } else {
                result.warnings.push_back("Failed to connect to node: " + node + " (may not be running)");
            }
            
            #ifdef _WIN32
            closesocket(test_socket);
            WSACleanup();
            #else
            close(test_socket);
            #endif
        }
        
        if (tested_nodes > 0) {
            double success_rate = static_cast<double>(successful_connections) / tested_nodes;
            if (success_rate < 0.5) {
                result.warnings.push_back("Low connectivity success rate: " + 
                                        std::to_string(successful_connections) + "/" + 
                                        std::to_string(tested_nodes) + " nodes reachable");
            }
            
            VGRE_LOG_INFO("ConfigurationManager", "Network connectivity test: " + 
                          std::to_string(successful_connections) + "/" + 
                          std::to_string(tested_nodes) + " nodes reachable");
        }
    }
    
    result.is_valid = result.errors.empty();
    return result;
}

ConfigurationValidationResult ConfigurationManager::validateFilePermissions(const ClusterConfiguration& config) {
    ConfigurationValidationResult result;
    
    if (!config.validate_file_permissions) {
        result.is_valid = true;
        return result;
    }
    
    VGRE_LOG_INFO("ConfigurationManager", "Validating file permissions for configuration");
    
    // Check config file permissions
    if (!config.config_file_path.empty()) {
        std::ifstream file(config.config_file_path);
        if (!file.good()) {
            result.errors.push_back("Cannot read configuration file: " + config.config_file_path);
        } else {
            // Check if file is writable (for hot reload updates)
            std::ofstream test_write(config.config_file_path, std::ios::app);
            if (!test_write.good()) {
                result.warnings.push_back("Configuration file is not writable: " + config.config_file_path + 
                                        " (hot reload updates will fail)");
            }
        }
    }
    
    // Check auth token file permissions
    if (!config.auth_token_file.empty()) {
        std::ifstream file(config.auth_token_file);
        if (!file.good()) {
            result.errors.push_back("Cannot read auth token file: " + config.auth_token_file);
        } else {
            // Check file permissions for security
            #ifndef _WIN32
            struct stat file_stat;
            if (stat(config.auth_token_file.c_str(), &file_stat) == 0) {
                // Check if file is readable by others (security risk)
                if (file_stat.st_mode & (S_IRGRP | S_IROTH)) {
                    result.security_warnings.push_back("Auth token file is readable by group/others: " + 
                                                      config.auth_token_file + " (security risk)");
                }
            }
            #endif
        }
    }
    
    // Check backup directory permissions
    if (!config.backup_directory.empty()) {
        #ifdef _WIN32
        DWORD attrs = GetFileAttributes(config.backup_directory.c_str());
        if (attrs != INVALID_FILE_ATTRIBUTES) {
            // Directory exists, check if we can write to it
            std::string test_file = config.backup_directory + "\\test_write.tmp";
            std::ofstream test(test_file);
            if (test.good()) {
                test.close();
                std::remove(test_file.c_str());
            } else {
                result.errors.push_back("Cannot write to backup directory: " + config.backup_directory);
            }
        }
        #else
        struct stat dir_stat;
        if (stat(config.backup_directory.c_str(), &dir_stat) == 0) {
            if (S_ISDIR(dir_stat.st_mode)) {
                // Check write permissions
                if (access(config.backup_directory.c_str(), W_OK) != 0) {
                    result.errors.push_back("No write permission for backup directory: " + config.backup_directory);
                }
            } else {
                result.errors.push_back("Backup directory path is not a directory: " + config.backup_directory);
            }
        }
        #endif
    }
    
    result.is_valid = result.errors.empty();
    return result;
}

} // namespace advanced
} // namespace vgre