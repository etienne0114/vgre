/**
 * Configuration Manager Backup and Recovery Operations
 * 
 * Handles backup creation, restoration, and cleanup operations
 * for the ConfigurationManager.
 */

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "vgre/common/os_backend.h"
#if defined(_WIN32)
#include <direct.h>   // _mkdir
#else
#include <dirent.h>   // opendir/readdir — backup file listing
#endif

namespace vgre {
namespace advanced {

std::string ConfigurationManager::createConfigurationBackup(const std::string& backup_path) {
    std::lock_guard<std::mutex> lock(state().config_mutex);
    
    std::string actual_backup_path = backup_path;
    
    // Generate backup path if not provided
    if (actual_backup_path.empty()) {
        std::string backup_dir = state().current_config.backup_directory;
        if (backup_dir.empty()) {
            backup_dir = "./config_backups";
        }
        
        // Create backup directory if it doesn't exist
        #ifdef _WIN32
        _mkdir(backup_dir.c_str());
        #else
        mkdir(backup_dir.c_str(), 0755);
        #endif
        
        // Generate timestamped backup filename
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << backup_dir << "/vgre_config_backup_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".json";
        actual_backup_path = ss.str();
    }
    
    // Save current configuration as backup
    if (saveToJsonFile(actual_backup_path, state().current_config)) {
        // Clean up old backups if auto-cleanup is enabled
        if (state().current_config.enable_auto_backup && state().current_config.max_backup_files > 0) {
            cleanupOldBackups(state().current_config.backup_directory, state().current_config.max_backup_files);
        }
        return actual_backup_path;
    }
    
    return "";
}

bool ConfigurationManager::restoreConfigurationFromBackup(const std::string& backup_path) {
    ClusterConfiguration backup_config;
    
    if (!loadFromJsonFile(backup_path, backup_config)) {
        VGRE_LOG_ERROR("ConfigurationManager", "Failed to load backup configuration from: " + backup_path);
        return false;
    }
    
    // Validate backup configuration
    auto validation = validateConfiguration(backup_config);
    if (!validation.is_valid) {
        VGRE_LOG_ERROR("ConfigurationManager", "Backup configuration is invalid");
        for (const auto& error : validation.errors) {
            VGRE_LOG_ERROR("ConfigurationManager", "Validation error: " + error);
        }
        return false;
    }
    
    // Update configuration
    bool success = updateConfiguration(backup_config, "backup_restore");
    
    if (success) {
        VGRE_LOG_INFO("ConfigurationManager", "Configuration restored from backup: " + backup_path);
    }
    
    return success;
}

std::vector<std::string> ConfigurationManager::listConfigurationBackups(const std::string& backup_directory) {
    std::vector<std::string> backups;
    
    std::string search_dir = backup_directory;
    if (search_dir.empty()) {
        std::lock_guard<std::mutex> lock(state().config_mutex);
        search_dir = state().current_config.backup_directory;
        if (search_dir.empty()) {
            search_dir = "./config_backups";
        }
    }
    
    #ifdef _WIN32
    WIN32_FIND_DATAA findFileData;
    std::string search_pattern = search_dir + "\\vgre_config_backup_*.json";
    HANDLE hFind = FindFirstFileA(search_pattern.c_str(), &findFileData);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string full_path = search_dir + "\\" + findFileData.cFileName;
            backups.push_back(full_path);
        } while (FindNextFileA(hFind, &findFileData) != 0);
        FindClose(hFind);
    }
    #else
    DIR* dir = opendir(search_dir.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find("vgre_config_backup_") == 0 && filename.find(".json") != std::string::npos) {
                std::string full_path = search_dir + "/" + filename;
                backups.push_back(full_path);
            }
        }
        closedir(dir);
    }
    #endif
    
    // Sort by modification time (newest first)
    std::sort(backups.begin(), backups.end(), [](const std::string& a, const std::string& b) {
        return getFileModificationTime(a) > getFileModificationTime(b);
    });
    
    return backups;
}

int ConfigurationManager::cleanupOldBackups(const std::string& backup_directory, int max_backups) {
    auto backups = listConfigurationBackups(backup_directory);
    
    int removed_count = 0;
    if (static_cast<int>(backups.size()) > max_backups) {
        // Remove oldest backups beyond the limit
        for (size_t i = max_backups; i < backups.size(); ++i) {
            if (std::remove(backups[i].c_str()) == 0) {
                removed_count++;
                VGRE_LOG_DEBUG("ConfigurationManager", "Removed old backup: " + backups[i]);
            }
        }
    }
    
    return removed_count;
}

} // namespace advanced
} // namespace vgre