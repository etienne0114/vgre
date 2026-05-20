// Configuration Manager File I/O — load, save, parse, read/write helpers.

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <fstream>
#include <sstream>
#if !defined(_WIN32)
#include <sys/stat.h>
#endif

#include "vgre/common/os_backend.h"
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4244 4267 4100 4127 4624)
#endif
#include <llvm/Support/JSON.h>
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

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

// ── JSON Parser (llvm::json) ──

bool ConfigurationManager::parseJsonConfiguration(const std::string& json_content, ClusterConfiguration& config) {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json_content);
    if (!parsed) {
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        os << parsed.takeError();
        VGRE_LOG_ERROR("ConfigurationManager", "JSON parse error: " + os.str());
        return false;
    }

    const llvm::json::Object* obj = parsed->getAsObject();
    if (!obj) {
        VGRE_LOG_ERROR("ConfigurationManager", "JSON root must be an object");
        return false;
    }

    // Helper lambdas for type-safe extraction with graceful fallback.
    auto getInt = [&](llvm::StringRef key, int &out) {
        if (auto *v = obj->get(key))
            if (auto n = v->getAsInteger()) out = static_cast<int>(*n);
    };
    auto getUInt64 = [&](llvm::StringRef key, size_t &out) {
        if (auto *v = obj->get(key))
            if (auto n = v->getAsInteger()) out = static_cast<size_t>(*n);
    };
    auto getDouble = [&](llvm::StringRef key, double &out) {
        if (auto *v = obj->get(key))
            if (auto n = v->getAsNumber()) out = *n;
    };
    auto getBool = [&](llvm::StringRef key, bool &out) {
        if (auto *v = obj->get(key))
            if (auto b = v->getAsBoolean()) out = *b;
    };
    auto getString = [&](llvm::StringRef key, std::string &out) {
        if (auto *v = obj->get(key))
            if (auto s = v->getAsString()) out = s->str();
    };

    // Network
    getInt   ("handshake_timeout_sec",    config.handshake_timeout_sec);
    getUInt64("max_queue_depth",          config.max_queue_depth);
    getUInt64("max_packets_per_sec",      config.max_packets_per_sec);
    getInt   ("connection_retry_attempts",config.connection_retry_attempts);
    getInt   ("connection_retry_delay_ms",config.connection_retry_delay_ms);
    // Security
    getBool  ("allow_auth_fallback",      config.allow_auth_fallback);
    getString("auth_token",               config.auth_token);
    getString("auth_token_file",          config.auth_token_file);
    getBool  ("require_encryption",       config.require_encryption);
    // Transport
    getUInt64("shm_threshold_bytes",      config.shm_threshold_bytes);
    getUInt64("rdma_threshold_bytes",     config.rdma_threshold_bytes);
    getDouble("delta_sync_threshold",     config.delta_sync_threshold);
    getUInt64("coalescing_threshold_bytes",config.coalescing_threshold_bytes);
    // Mesh
    getString("cluster_nodes",            config.cluster_nodes);
    getBool  ("enable_mesh_topology",     config.enable_mesh_topology);
    getInt   ("mesh_discovery_port",      config.mesh_discovery_port);
    // Monitoring
    getBool  ("enable_metrics_export",    config.enable_metrics_export);
    getInt   ("metrics_export_interval_sec", config.metrics_export_interval_sec);
    getString("log_level",                config.log_level);
    // Backup
    getBool  ("enable_hot_reload",        config.enable_hot_reload);
    getString("backup_directory",         config.backup_directory);
    getBool  ("enable_auto_backup",       config.enable_auto_backup);
    getInt   ("max_backup_files",         config.max_backup_files);
    // Validation
    getBool  ("strict_validation",        config.strict_validation);
    getBool  ("validate_network_connectivity", config.validate_network_connectivity);
    getBool  ("validate_file_permissions",config.validate_file_permissions);
    return true;
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
    if (GetFileAttributesExA(file_path.c_str(), GetFileExInfoStandard, &fileInfo)) {
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