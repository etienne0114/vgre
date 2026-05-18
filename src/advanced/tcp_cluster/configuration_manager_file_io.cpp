// Configuration Manager File I/O — load, save, parse, read/write helpers.

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace advanced {

// ── File Load/Save ──

bool ConfigurationManager::loadFromJsonFile(const std::string& file_path, ClusterConfiguration& config) {
    std::string content;
    if (!readFileContent(file_path, content)) return false;
    bool ok = parseJsonConfiguration(content, config);
    if (ok) {
        config.config_file_path = file_path;
        config.last_modified_time = getFileModificationTime(file_path);
        VGRE_LOG_INFO("ConfigurationManager", "Loaded JSON config from: " + file_path);
    }
    return ok;
}

bool ConfigurationManager::loadFromYamlFile(const std::string& file_path, ClusterConfiguration& config) {
    std::string content;
    if (!readFileContent(file_path, content)) return false;
    bool ok = parseYamlConfiguration(content, config);
    if (ok) {
        config.config_file_path = file_path;
        config.last_modified_time = getFileModificationTime(file_path);
        VGRE_LOG_INFO("ConfigurationManager", "Loaded YAML config from: " + file_path);
    }
    return ok;
}

bool ConfigurationManager::saveToJsonFile(const std::string& file_path, const ClusterConfiguration& config) {
    bool ok = writeFileContent(file_path, configurationToJson(config));
    if (ok) VGRE_LOG_INFO("ConfigurationManager", "Saved JSON config to: " + file_path);
    return ok;
}

bool ConfigurationManager::saveToYamlFile(const std::string& file_path, const ClusterConfiguration& config) {
    bool ok = writeFileContent(file_path, configurationToYaml(config));
    if (ok) VGRE_LOG_INFO("ConfigurationManager", "Saved YAML config to: " + file_path);
    return ok;
}

// ── JSON Parser ──

bool ConfigurationManager::parseJsonConfiguration(const std::string& json_content, ClusterConfiguration& config) {
    size_t pos = 0;

    auto skipWhitespace = [&]() {
        while (pos < json_content.size() && std::isspace(json_content[pos])) pos++;
    };

    auto parseString = [&]() -> std::string {
        skipWhitespace();
        if (pos >= json_content.size() || json_content[pos] != '"') return "";
        pos++;
        std::string result;
        while (pos < json_content.size() && json_content[pos] != '"') {
            if (json_content[pos] == '\\' && pos + 1 < json_content.size()) {
                pos++;
                switch (json_content[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '\\': result += '\\'; break;
                    case '"': result += '"'; break;
                    default: result += json_content[pos]; break;
                }
            } else {
                result += json_content[pos];
            }
            pos++;
        }
        if (pos < json_content.size()) pos++;
        return result;
    };

    auto parseNumber = [&]() -> double {
        skipWhitespace();
        std::string num;
        while (pos < json_content.size() &&
               (std::isdigit(json_content[pos]) || json_content[pos] == '.' ||
                json_content[pos] == '-' || json_content[pos] == '+' ||
                json_content[pos] == 'e' || json_content[pos] == 'E')) {
            num += json_content[pos++];
        }
        try { return std::stod(num); } catch (...) { return 0.0; }
    };

    auto parseBool = [&]() -> bool {
        skipWhitespace();
        if (pos + 4 <= json_content.size() && json_content.substr(pos, 4) == "true") { pos += 4; return true; }
        if (pos + 5 <= json_content.size() && json_content.substr(pos, 5) == "false") { pos += 5; return false; }
        return false;
    };

    auto findKey = [&](const std::string& key) -> bool {
        size_t start = pos;
        while (pos < json_content.size()) {
            skipWhitespace();
            if (pos >= json_content.size() || json_content[pos] != '"') { pos++; continue; }
            std::string found = parseString();
            skipWhitespace();
            if (pos < json_content.size() && json_content[pos] == ':') {
                pos++;
                if (found == key) return true;
            }
        }
        pos = start;
        return false;
    };

    try {
        // Network
        if (findKey("handshake_timeout_sec")) config.handshake_timeout_sec = static_cast<int>(parseNumber());
        pos = 0; if (findKey("max_queue_depth")) config.max_queue_depth = static_cast<size_t>(parseNumber());
        pos = 0; if (findKey("max_packets_per_sec")) config.max_packets_per_sec = static_cast<size_t>(parseNumber());
        pos = 0; if (findKey("connection_retry_attempts")) config.connection_retry_attempts = static_cast<int>(parseNumber());
        pos = 0; if (findKey("connection_retry_delay_ms")) config.connection_retry_delay_ms = static_cast<int>(parseNumber());
        // Security
        pos = 0; if (findKey("allow_auth_fallback")) config.allow_auth_fallback = parseBool();
        pos = 0; if (findKey("auth_token")) config.auth_token = parseString();
        pos = 0; if (findKey("auth_token_file")) config.auth_token_file = parseString();
        pos = 0; if (findKey("require_encryption")) config.require_encryption = parseBool();
        // Transport
        pos = 0; if (findKey("shm_threshold_bytes")) config.shm_threshold_bytes = static_cast<size_t>(parseNumber());
        pos = 0; if (findKey("rdma_threshold_bytes")) config.rdma_threshold_bytes = static_cast<size_t>(parseNumber());
        pos = 0; if (findKey("delta_sync_threshold")) config.delta_sync_threshold = parseNumber();
        pos = 0; if (findKey("coalescing_threshold_bytes")) config.coalescing_threshold_bytes = static_cast<size_t>(parseNumber());
        // Mesh
        pos = 0; if (findKey("cluster_nodes")) config.cluster_nodes = parseString();
        pos = 0; if (findKey("enable_mesh_topology")) config.enable_mesh_topology = parseBool();
        pos = 0; if (findKey("mesh_discovery_port")) config.mesh_discovery_port = static_cast<int>(parseNumber());
        // Monitoring
        pos = 0; if (findKey("enable_metrics_export")) config.enable_metrics_export = parseBool();
        pos = 0; if (findKey("metrics_export_interval_sec")) config.metrics_export_interval_sec = static_cast<int>(parseNumber());
        pos = 0; if (findKey("log_level")) config.log_level = parseString();
        // Backup
        pos = 0; if (findKey("enable_hot_reload")) config.enable_hot_reload = parseBool();
        pos = 0; if (findKey("backup_directory")) config.backup_directory = parseString();
        pos = 0; if (findKey("enable_auto_backup")) config.enable_auto_backup = parseBool();
        pos = 0; if (findKey("max_backup_files")) config.max_backup_files = static_cast<int>(parseNumber());
        // Validation
        pos = 0; if (findKey("strict_validation")) config.strict_validation = parseBool();
        pos = 0; if (findKey("validate_network_connectivity")) config.validate_network_connectivity = parseBool();
        pos = 0; if (findKey("validate_file_permissions")) config.validate_file_permissions = parseBool();
        return true;
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("ConfigurationManager", "JSON parse error: " + std::string(e.what()));
        return false;
    }
}

// ── YAML Parser ──

bool ConfigurationManager::parseYamlConfiguration(const std::string& yaml_content, ClusterConfiguration& config) {
    std::istringstream stream(yaml_content);
    std::string line;

    while (std::getline(stream, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue;

        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);

        try {
            if      (key == "handshake_timeout_sec")    config.handshake_timeout_sec = std::stoi(value);
            else if (key == "max_queue_depth")          config.max_queue_depth = std::stoull(value);
            else if (key == "max_packets_per_sec")      config.max_packets_per_sec = std::stoull(value);
            else if (key == "connection_retry_attempts") config.connection_retry_attempts = std::stoi(value);
            else if (key == "connection_retry_delay_ms") config.connection_retry_delay_ms = std::stoi(value);
            else if (key == "allow_auth_fallback")      config.allow_auth_fallback = (value == "true" || value == "1");
            else if (key == "auth_token")               config.auth_token = value;
            else if (key == "auth_token_file")          config.auth_token_file = value;
            else if (key == "require_encryption")       config.require_encryption = (value == "true" || value == "1");
            else if (key == "shm_threshold_bytes")      config.shm_threshold_bytes = std::stoull(value);
            else if (key == "rdma_threshold_bytes")     config.rdma_threshold_bytes = std::stoull(value);
            else if (key == "delta_sync_threshold")     config.delta_sync_threshold = std::stod(value);
            else if (key == "coalescing_threshold_bytes") config.coalescing_threshold_bytes = std::stoull(value);
            else if (key == "cluster_nodes")            config.cluster_nodes = value;
            else if (key == "enable_mesh_topology")     config.enable_mesh_topology = (value == "true" || value == "1");
            else if (key == "mesh_discovery_port")      config.mesh_discovery_port = std::stoi(value);
            else if (key == "enable_metrics_export")    config.enable_metrics_export = (value == "true" || value == "1");
            else if (key == "metrics_export_interval_sec") config.metrics_export_interval_sec = std::stoi(value);
            else if (key == "log_level")                config.log_level = value;
            else if (key == "enable_hot_reload")        config.enable_hot_reload = (value == "true" || value == "1");
            else if (key == "backup_directory")         config.backup_directory = value;
            else if (key == "enable_auto_backup")       config.enable_auto_backup = (value == "true" || value == "1");
            else if (key == "max_backup_files")         config.max_backup_files = std::stoi(value);
            else if (key == "strict_validation")        config.strict_validation = (value == "true" || value == "1");
            else if (key == "validate_network_connectivity") config.validate_network_connectivity = (value == "true" || value == "1");
            else if (key == "validate_file_permissions") config.validate_file_permissions = (value == "true" || value == "1");
        } catch (const std::exception& e) {
            VGRE_LOG_WARN("ConfigurationManager", "Failed to parse YAML key '" + key + "': " + value);
        }
    }
    return true;
}

// ── Low-level File Helpers ──

uint64_t ConfigurationManager::getFileModificationTime(const std::string& file_path) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (GetFileAttributesEx(file_path.c_str(), GetFileExInfoStandard, &fileInfo)) {
        ULARGE_INTEGER ull;
        ull.LowPart = fileInfo.ftLastWriteTime.dwLowDateTime;
        ull.HighPart = fileInfo.ftLastWriteTime.dwHighDateTime;
        return ull.QuadPart / 10000;
    }
    return 0;
#else
    struct stat file_stat;
    if (stat(file_path.c_str(), &file_stat) == 0)
        return static_cast<uint64_t>(file_stat.st_mtime) * 1000;
    return 0;
#endif
}

bool ConfigurationManager::readFileContent(const std::string& file_path, std::string& content) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        VGRE_LOG_ERROR("ConfigurationManager", "Cannot open for reading: " + file_path);
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    content.resize(file_size);
    file.read(&content[0], file_size);
    if (file.fail() && !file.eof()) {
        VGRE_LOG_ERROR("ConfigurationManager", "Read error: " + file_path);
        return false;
    }
    return true;
}

bool ConfigurationManager::writeFileContent(const std::string& file_path, const std::string& content) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        VGRE_LOG_ERROR("ConfigurationManager", "Cannot open for writing: " + file_path);
        return false;
    }
    file.write(content.c_str(), content.size());
    if (file.fail()) {
        VGRE_LOG_ERROR("ConfigurationManager", "Write error: " + file_path);
        return false;
    }
    return true;
}

} // namespace advanced
} // namespace vgre