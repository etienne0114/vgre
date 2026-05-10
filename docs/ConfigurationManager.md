# ConfigurationManager Enhancement Documentation

## Overview

The ConfigurationManager has been significantly enhanced to provide production-ready configuration management for the VGRE TCP Cluster system. This enhancement addresses Task 6.1 of the TCP Cluster Production Readiness specification.

## Key Features

### 1. Configuration File Support (JSON/YAML)

The ConfigurationManager now supports loading configuration from both JSON and YAML files:

```bash
# Set configuration file via environment variable
export VGRE_CONFIG_FILE="/etc/vgre/cluster-config.yaml"
# or
export VGRE_CONFIG_FILE="/etc/vgre/cluster-config.json"
```

**JSON Example:**
```json
{
  "handshake_timeout_sec": 10,
  "max_queue_depth": 2048,
  "allow_auth_fallback": false,
  "auth_token": "your-secure-token-here",
  "require_encryption": true,
  "cluster_nodes": "master:7777,worker1:7777,worker2:7777",
  "log_level": "INFO"
}
```

**YAML Example:**
```yaml
# VGRE TCP Cluster Configuration

# Network Configuration
handshake_timeout_sec: 10
max_queue_depth: 2048
max_packets_per_sec: 50000

# Security Configuration
allow_auth_fallback: false
auth_token: "your-secure-token-here"
require_encryption: true

# Mesh Topology Configuration
cluster_nodes: "master:7777,worker1:7777,worker2:7777"
enable_mesh_topology: true

# Monitoring Configuration
log_level: "INFO"
```

### 2. Hot Reloading

Enable automatic configuration reloading when files change:

```bash
export VGRE_CONFIG_HOT_RELOAD=true
```

The system will automatically detect file changes and reload configuration with validation.

### 3. Configuration Profiles

Support for deployment-specific profiles with optimized defaults:

```bash
export VGRE_DEPLOYMENT_PROFILE=production
# Options: development, staging, production, custom
```

**Profile Characteristics:**

| Setting | Development | Staging | Production |
|---------|-------------|---------|------------|
| Handshake Timeout | 5s | 8s | 10s |
| Max Queue Depth | 1024 | 1536 | 2048 |
| Auth Fallback | Enabled | Enabled | **Disabled** |
| Encryption Required | No | No | **Yes** |
| Log Level | DEBUG | DEBUG | INFO |
| Metrics Export | Disabled | Enabled | Enabled |

### 4. Advanced Validation

Comprehensive validation with detailed error reporting, warnings, and suggestions:

```cpp
auto validation = ConfigurationManager::validateConfiguration(config);
if (!validation.is_valid) {
    for (const auto& error : validation.errors) {
        std::cout << "ERROR: " << error << std::endl;
    }
    for (const auto& suggestion : validation.suggestions) {
        std::cout << "SUGGESTION: " << suggestion << std::endl;
    }
}
```

**Validation Rules:**
- Network timeouts: 1-300 seconds
- Queue depths: 1-1,000,000 entries
- Auth tokens: minimum 16 characters
- Port numbers: 1-65535
- Profile-specific security checks

### 5. Thread-Safe Configuration Access

All configuration operations are thread-safe and can be called concurrently from multiple threads.

## Environment Variables

### Core Configuration
- `VGRE_CONFIG_FILE`: Path to JSON/YAML configuration file
- `VGRE_CONFIG_HOT_RELOAD`: Enable hot reloading (true/false)
- `VGRE_DEPLOYMENT_PROFILE`: Configuration profile (development/staging/production/custom)

### Network Settings
- `VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC`: Handshake timeout (default: profile-dependent)
- `VGRE_CLUSTER_MAX_QUEUE_DEPTH`: Maximum queue depth (default: profile-dependent)
- `VGRE_CLUSTER_MAX_PACKETS_PER_SEC`: Rate limiting (default: 10000)
- `VGRE_CLUSTER_CONNECTION_RETRY_ATTEMPTS`: Retry attempts (default: 3)
- `VGRE_CLUSTER_CONNECTION_RETRY_DELAY_MS`: Retry delay (default: 1000ms)

### Security Settings
- `VGRE_ALLOW_AUTH_FALLBACK`: Enable auth fallback mode (default: profile-dependent)
- `VGRE_TCP_AUTH_TOKEN`: Authentication token
- `VGRE_TCP_AUTH_TOKEN_FILE`: Path to token file
- `VGRE_REQUIRE_ENCRYPTION`: Require encryption (default: profile-dependent)

### Transport Settings
- `VGRE_SHM_THRESHOLD_BYTES`: SHM transport threshold (default: 64KB)
- `VGRE_RDMA_THRESHOLD_BYTES`: RDMA transport threshold (default: 256KB)
- `VGRE_DELTA_SYNC_THRESHOLD`: Delta sync threshold (default: 0.3)
- `VGRE_COALESCING_THRESHOLD_BYTES`: Coalescing threshold (default: 4KB)

### Mesh Topology Settings
- `VGRE_CLUSTER_NODES`: Comma-separated list of cluster nodes
- `VGRE_ENABLE_MESH_TOPOLOGY`: Enable mesh topology (default: false)
- `VGRE_MESH_DISCOVERY_PORT`: Discovery port (default: 7778)

### Monitoring Settings
- `VGRE_ENABLE_METRICS_EXPORT`: Enable metrics export (default: profile-dependent)
- `VGRE_METRICS_EXPORT_INTERVAL_SEC`: Export interval (default: 60s)
- `VGRE_LOG_LEVEL`: Log level (ERROR/WARN/INFO/DEBUG/TRACE)

## API Reference

### Core Methods

```cpp
// Get complete configuration with profile and file support
ClusterConfiguration getClusterConfiguration();

// Load from specific file formats
bool loadFromJsonFile(const std::string& file_path, ClusterConfiguration& config);
bool loadFromYamlFile(const std::string& file_path, ClusterConfiguration& config);

// Save configuration to files
bool saveToJsonFile(const std::string& file_path, const ClusterConfiguration& config);
bool saveToYamlFile(const std::string& file_path, const ClusterConfiguration& config);

// Hot reloading support
bool hasConfigurationChanged(const std::string& file_path, uint64_t last_modified_time);
bool reloadIfChanged(ClusterConfiguration& config);

// Profile management
ConfigurationProfile getConfigurationProfile();
void applyProfileDefaults(ClusterConfiguration& config, ConfigurationProfile profile);

// Validation with detailed reporting
ConfigurationValidationResult validateConfiguration(const ClusterConfiguration& config);

// Example generation for documentation
bool generateExampleConfigurations(const std::string& output_dir);
```

### Configuration Structures

```cpp
enum class ConfigurationProfile {
    DEVELOPMENT,
    STAGING,
    PRODUCTION,
    CUSTOM
};

struct ConfigurationValidationResult {
    bool is_valid;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::vector<std::string> suggestions;
};

struct ClusterConfiguration {
    // Network configuration
    int handshake_timeout_sec;
    size_t max_queue_depth;
    size_t max_packets_per_sec;
    int connection_retry_attempts;
    int connection_retry_delay_ms;
    
    // Security configuration
    bool allow_auth_fallback;
    std::string auth_token;
    std::string auth_token_file;
    bool require_encryption;
    
    // Transport configuration
    size_t shm_threshold_bytes;
    size_t rdma_threshold_bytes;
    double delta_sync_threshold;
    size_t coalescing_threshold_bytes;
    
    // Mesh topology configuration
    std::string cluster_nodes;
    bool enable_mesh_topology;
    int mesh_discovery_port;
    
    // Monitoring configuration
    bool enable_metrics_export;
    int metrics_export_interval_sec;
    std::string log_level;
    
    // Profile and file configuration
    ConfigurationProfile profile;
    std::string config_file_path;
    bool enable_hot_reload;
    uint64_t last_modified_time;
};
```

## Usage Examples

### Basic Usage

```cpp
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"

// Load configuration with profile and environment variable support
auto config = ConfigurationManager::getClusterConfiguration();

// Validate configuration
auto validation = ConfigurationManager::validateConfiguration(config);
if (!validation.is_valid) {
    // Handle validation errors
    for (const auto& error : validation.errors) {
        VGRE_LOG_ERROR("Config", error);
    }
    return false;
}

// Use configuration
tcp_cluster_manager.initialize(config);
```

### Hot Reloading

```cpp
// Enable hot reloading
setenv("VGRE_CONFIG_HOT_RELOAD", "1", 1);
setenv("VGRE_CONFIG_FILE", "/etc/vgre/cluster.yaml", 1);

auto config = ConfigurationManager::getClusterConfiguration();

// In your main loop or timer
if (ConfigurationManager::reloadIfChanged(config)) {
    VGRE_LOG_INFO("Config", "Configuration reloaded successfully");
    // Apply new configuration
    tcp_cluster_manager.updateConfiguration(config);
}
```

### Profile-Specific Deployment

```cpp
// Production deployment
setenv("VGRE_DEPLOYMENT_PROFILE", "production", 1);
auto prod_config = ConfigurationManager::getClusterConfiguration();
// prod_config.allow_auth_fallback == false
// prod_config.require_encryption == true
// prod_config.handshake_timeout_sec == 10

// Development deployment
setenv("VGRE_DEPLOYMENT_PROFILE", "development", 1);
auto dev_config = ConfigurationManager::getClusterConfiguration();
// dev_config.allow_auth_fallback == true
// dev_config.require_encryption == false
// dev_config.handshake_timeout_sec == 5
```

### Configuration File Generation

```cpp
// Generate example configurations for documentation
ConfigurationManager::generateExampleConfigurations("/etc/vgre/examples/");
// Creates:
// - /etc/vgre/examples/vgre-development.json
// - /etc/vgre/examples/vgre-development.yaml
// - /etc/vgre/examples/vgre-production.json
// - /etc/vgre/examples/vgre-production.yaml
```

## Integration Points

### Diagnostic Logger Integration

The ConfigurationManager integrates with the diagnostic logger for configuration change events:

```cpp
// Configuration changes are automatically logged
VGRE_LOG_INFO("ConfigurationManager", "Configuration reloaded successfully");
VGRE_LOG_WARN("ConfigurationManager", "Configuration validation warning: ...");
VGRE_LOG_ERROR("ConfigurationManager", "Configuration validation error: ...");
```

### Mesh Topology Configuration

The ConfigurationManager provides configuration for mesh topology:

```cpp
if (config.enable_mesh_topology) {
    mesh_manager.initialize(config.cluster_nodes, config.mesh_discovery_port);
}
```

### Security Manager Integration

Security settings are automatically applied:

```cpp
security_manager.setAuthMode(config.allow_auth_fallback ? 
    SecurityManager::AuthMode::FALLBACK : 
    SecurityManager::AuthMode::STRICT);
security_manager.setEncryptionRequired(config.require_encryption);
```

## Backward Compatibility

The enhanced ConfigurationManager maintains full backward compatibility:

- All existing environment variables continue to work
- Default values remain unchanged for development profile
- Existing code using `getClusterConfiguration()` works without modification
- Environment variables take precedence over file configuration

## Performance Considerations

- Configuration loading is optimized for startup time
- Hot reloading uses efficient file modification time checking
- JSON/YAML parsing is lightweight and doesn't require external dependencies
- Thread-safe operations use minimal locking
- Configuration validation is fast and suitable for hot reload scenarios

## Security Considerations

- Configuration files should have appropriate file permissions (600 or 640)
- Auth tokens in configuration files should be protected
- Hot reloading validates new configuration before applying
- Profile-specific security warnings help prevent misconfigurations
- Token file support allows separation of secrets from main configuration

## Testing

The ConfigurationManager includes comprehensive test coverage:

```bash
# Run the standalone test
./test_config_simple

# Test with different profiles
VGRE_DEPLOYMENT_PROFILE=production ./test_config_simple
VGRE_DEPLOYMENT_PROFILE=staging ./test_config_simple

# Test with configuration files
echo '{"handshake_timeout_sec": 15}' > /tmp/test.json
VGRE_CONFIG_FILE=/tmp/test.json ./test_config_simple
```

## Migration Guide

### From Environment Variables Only

**Before:**
```bash
export VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC=10
export VGRE_ALLOW_AUTH_FALLBACK=0
export VGRE_TCP_AUTH_TOKEN="my-token"
```

**After (Option 1 - Keep environment variables):**
```bash
# No changes needed - environment variables still work
export VGRE_DEPLOYMENT_PROFILE=production  # Optional: use profile
```

**After (Option 2 - Use configuration file):**
```yaml
# /etc/vgre/cluster.yaml
handshake_timeout_sec: 10
allow_auth_fallback: false
auth_token: "my-token"
```

```bash
export VGRE_CONFIG_FILE="/etc/vgre/cluster.yaml"
export VGRE_DEPLOYMENT_PROFILE=production
```

### Adding Hot Reloading

```bash
# Enable hot reloading for existing configuration file
export VGRE_CONFIG_HOT_RELOAD=true
```

```cpp
// In your application loop
if (ConfigurationManager::reloadIfChanged(config)) {
    // Handle configuration changes
    updateSystemConfiguration(config);
}
```

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```
   ERROR: Cannot open file for reading: /path/to/config.yaml
   ```
   - Check file path and permissions
   - Verify VGRE_CONFIG_FILE environment variable

2. **Invalid JSON/YAML syntax**
   ```
   WARN: Failed to load JSON config from: /path/to/config.json
   ```
   - Validate JSON/YAML syntax
   - Check for trailing commas in JSON

3. **Validation errors**
   ```
   ERROR: Invalid handshake timeout: 0 (must be 1-300 seconds)
   ```
   - Review validation messages and suggestions
   - Check configuration value ranges

4. **Hot reloading not working**
   - Ensure VGRE_CONFIG_HOT_RELOAD=true
   - Check file modification permissions
   - Verify file system supports modification time updates

### Debug Mode

Enable debug logging to troubleshoot configuration issues:

```bash
export VGRE_LOG_LEVEL=DEBUG
```

This will show detailed configuration loading and validation information.

## Future Enhancements

Potential future enhancements for the ConfigurationManager:

1. **Configuration Schema Validation**: JSON Schema or YAML schema validation
2. **Configuration Encryption**: Encrypted configuration file support
3. **Remote Configuration**: Support for loading configuration from remote sources
4. **Configuration Versioning**: Track configuration changes over time
5. **Configuration Templates**: Template-based configuration generation
6. **Configuration Diff**: Show differences between configurations
7. **Configuration Backup**: Automatic backup of working configurations

## Conclusion

The enhanced ConfigurationManager provides a robust, production-ready configuration system that supports:

- ✅ JSON and YAML configuration files
- ✅ Hot reloading with validation
- ✅ Advanced validation with detailed error reporting
- ✅ Configuration profiles for different deployment environments
- ✅ Comprehensive documentation and examples
- ✅ Thread-safe configuration access
- ✅ Backward compatibility with existing environment variables

This enhancement significantly improves the production readiness of the VGRE TCP Cluster system by providing enterprise-grade configuration management capabilities.