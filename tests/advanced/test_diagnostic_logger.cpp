/**
 * Test suite for Diagnostic Logger and Shared Utilities
 * 
 * Validates the diagnostic logging system and shared utility classes
 * created for task 5.2: Enhanced diagnostic logging and operational monitoring.
 */

#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

using namespace vgre::advanced;

class DiagnosticLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset any previous state
        auto& logger = DiagnosticLogger::instance();
        // Logger is a singleton, so we can't reset it completely
        // but we can test its functionality
    }
};

// ── Test Shared Utility Classes ─────────────────────────────────────────────

TEST_F(DiagnosticLoggerTest, LoggingPatternsFormatMessage) {
    std::string result = LoggingPatterns::formatMessage("TestComponent", "TestOperation", 
                                                       "Test details", "Test context");
    
    EXPECT_TRUE(result.find("TestComponent") != std::string::npos);
    EXPECT_TRUE(result.find("TestOperation") != std::string::npos);
    EXPECT_TRUE(result.find("Test details") != std::string::npos);
    EXPECT_TRUE(result.find("Test context") != std::string::npos);
}

TEST_F(DiagnosticLoggerTest, LoggingPatternsFormatError) {
    std::string result = LoggingPatterns::formatError("Test error", "Test suggestion");
    
    EXPECT_TRUE(result.find("Test error") != std::string::npos);
    EXPECT_TRUE(result.find("Test suggestion") != std::string::npos);
    EXPECT_TRUE(result.find("Suggestion:") != std::string::npos);
}

TEST_F(DiagnosticLoggerTest, ErrorHandlingPatternsTranslateVGREResult) {
    EXPECT_EQ(ErrorHandlingPatterns::translateVGREResult(VGREResult::SUCCESS), "SUCCESS");
    EXPECT_EQ(ErrorHandlingPatterns::translateVGREResult(VGREResult::ERR_NETWORK), "NETWORK_ERROR");
    EXPECT_EQ(ErrorHandlingPatterns::translateVGREResult(VGREResult::ERR_TIMEOUT), "TIMEOUT");
}

TEST_F(DiagnosticLoggerTest, ErrorHandlingPatternsRetryableErrors) {
    EXPECT_TRUE(ErrorHandlingPatterns::isRetryableError(VGREResult::ERR_NETWORK));
    EXPECT_TRUE(ErrorHandlingPatterns::isRetryableError(VGREResult::ERR_TIMEOUT));
    EXPECT_TRUE(ErrorHandlingPatterns::isRetryableError(VGREResult::ERR_BUSY));
    
    EXPECT_FALSE(ErrorHandlingPatterns::isRetryableError(VGREResult::ERR_OUT_OF_MEMORY));
    EXPECT_FALSE(ErrorHandlingPatterns::isRetryableError(VGREResult::SUCCESS));
}

TEST_F(DiagnosticLoggerTest, ErrorHandlingPatternsCriticalErrors) {
    EXPECT_TRUE(ErrorHandlingPatterns::isCriticalError(VGREResult::ERR_OUT_OF_MEMORY));
    EXPECT_TRUE(ErrorHandlingPatterns::isCriticalError(VGREResult::ERR_NOT_INITIALIZED));
    
    EXPECT_FALSE(ErrorHandlingPatterns::isCriticalError(VGREResult::ERR_NETWORK));
    EXPECT_FALSE(ErrorHandlingPatterns::isCriticalError(VGREResult::SUCCESS));
}

TEST_F(DiagnosticLoggerTest, PlatformDetectionGetCurrentPlatform) {
    auto platform_info = PlatformDetection::getCurrentPlatform();
    
    EXPECT_NE(platform_info.type, PlatformType::UNKNOWN);
    EXPECT_FALSE(platform_info.name.empty());
    EXPECT_FALSE(platform_info.socket_api.empty());
    
    // All supported platforms should support UDP broadcast
    EXPECT_TRUE(platform_info.supports_udp_broadcast);
}

TEST_F(DiagnosticLoggerTest, PlatformDetectionCompatibility) {
    EXPECT_TRUE(PlatformDetection::isPlatformCompatible(PlatformType::WINDOWS, PlatformType::LINUX));
    EXPECT_TRUE(PlatformDetection::isPlatformCompatible(PlatformType::LINUX, PlatformType::MACOS));
    EXPECT_TRUE(PlatformDetection::isPlatformCompatible(PlatformType::WINDOWS, PlatformType::MACOS));
    
    EXPECT_FALSE(PlatformDetection::isPlatformCompatible(PlatformType::UNKNOWN, PlatformType::LINUX));
}

TEST_F(DiagnosticLoggerTest, ConfigurationManagerEnvironmentVariables) {
    // Test with default values (no environment variables set)
    int timeout = ConfigurationManager::getEnvVar<int>("NONEXISTENT_VAR", 42);
    EXPECT_EQ(timeout, 42);
    
    bool flag = ConfigurationManager::getEnvVar<bool>("NONEXISTENT_BOOL", true);
    EXPECT_TRUE(flag);
    
    std::string str = ConfigurationManager::getEnvVar<std::string>("NONEXISTENT_STR", "default");
    EXPECT_EQ(str, "default");
}

TEST_F(DiagnosticLoggerTest, ConfigurationManagerClusterConfiguration) {
    auto config = ConfigurationManager::getClusterConfiguration();
    
    // Verify default values are reasonable
    EXPECT_GT(config.handshake_timeout_sec, 0);
    EXPECT_GT(config.max_queue_depth, 0);
    EXPECT_GT(config.max_packets_per_sec, 0);
    EXPECT_GE(config.delta_sync_threshold, 0.0);
    EXPECT_LE(config.delta_sync_threshold, 1.0);
}

TEST_F(DiagnosticLoggerTest, PacketUtilsVSBPConstruction) {
    const char* test_payload = "test_data";
    size_t payload_size = strlen(test_payload);
    
    auto packet = PacketUtils::constructVSBPPacket(PacketType::TELEMETRY, 
                                                  test_payload, payload_size, 123);
    
    EXPECT_GE(packet.size(), sizeof(VSBPHeader) + payload_size);
    
    // Verify header
    const VSBPHeader* header = reinterpret_cast<const VSBPHeader*>(packet.data());
    EXPECT_EQ(header->magic, VSBP_MAGIC);
    EXPECT_EQ(header->version, VSBP_VERSION);
    EXPECT_EQ(header->type, static_cast<uint16_t>(PacketType::TELEMETRY));
    EXPECT_EQ(header->sequence, 123);
    EXPECT_EQ(header->payloadSize, payload_size);
}

TEST_F(DiagnosticLoggerTest, PacketUtilsValidation) {
    VSBPHeader valid_header;
    valid_header.magic = VSBP_MAGIC;
    valid_header.version = VSBP_VERSION;
    valid_header.type = static_cast<uint16_t>(PacketType::TELEMETRY);
    valid_header.payloadSize = 100;
    
    EXPECT_TRUE(PacketUtils::validateVSBPHeader(valid_header));
    
    // Test invalid magic
    VSBPHeader invalid_header = valid_header;
    invalid_header.magic = 0x12345678;
    EXPECT_FALSE(PacketUtils::validateVSBPHeader(invalid_header));
    
    // Test invalid version
    invalid_header = valid_header;
    invalid_header.version = 0x9999;
    EXPECT_FALSE(PacketUtils::validateVSBPHeader(invalid_header));
}

TEST_F(DiagnosticLoggerTest, PacketUtilsTypeToString) {
    EXPECT_EQ(PacketUtils::packetTypeToString(PacketType::TELEMETRY), "TELEMETRY");
    EXPECT_EQ(PacketUtils::packetTypeToString(PacketType::LAUNCH_KERNEL), "LAUNCH_KERNEL");
    EXPECT_EQ(PacketUtils::packetTypeToString(PacketType::RESPONSE), "RESPONSE");
}

TEST_F(DiagnosticLoggerTest, TimeUtilsTimestamp) {
    std::string timestamp = TimeUtils::getCurrentTimestamp();
    EXPECT_FALSE(timestamp.empty());
    
    // Should contain date and time components
    EXPECT_TRUE(timestamp.find("-") != std::string::npos); // Date separators
    EXPECT_TRUE(timestamp.find(":") != std::string::npos); // Time separators
    EXPECT_TRUE(timestamp.find(".") != std::string::npos); // Milliseconds
}

TEST_F(DiagnosticLoggerTest, TimeUtilsElapsed) {
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    double elapsed = TimeUtils::calculateElapsedMs(start);
    EXPECT_GE(elapsed, 10.0); // Should be at least 10ms
    EXPECT_LT(elapsed, 100.0); // Should be less than 100ms (reasonable upper bound)
}

TEST_F(DiagnosticLoggerTest, TimeUtilsFormatDuration) {
    EXPECT_EQ(TimeUtils::formatDuration(500), "500 ms");
    EXPECT_TRUE(TimeUtils::formatDuration(1500).find("s") != std::string::npos);
    EXPECT_TRUE(TimeUtils::formatDuration(65000).find("min") != std::string::npos);
}

// ── Test Diagnostic Logger Functionality ───────────────────────────────────

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerSingleton) {
    auto& logger1 = DiagnosticLogger::instance();
    auto& logger2 = DiagnosticLogger::instance();
    
    // Should be the same instance
    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerSecurityEvent) {
    auto& logger = DiagnosticLogger::instance();
    
    // This should not throw
    EXPECT_NO_THROW(logger.logSecurityEvent("TEST_EVENT", "192.168.1.100", "abc123"));
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerNetworkError) {
    auto& logger = DiagnosticLogger::instance();
    
    // This should not throw
    EXPECT_NO_THROW(logger.logNetworkError(VGREResult::ERR_NETWORK, "test_operation", "192.168.1.100"));
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerPacketCorruption) {
    auto& logger = DiagnosticLogger::instance();
    
    uint8_t test_data[] = {0x01, 0x02, 0x03, 0x04};
    
    // This should not throw
    EXPECT_NO_THROW(logger.logPacketCorruption(test_data, sizeof(test_data), "TEST_PACKET"));
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerPerformanceMetrics) {
    auto& logger = DiagnosticLogger::instance();
    
    // Record some test metrics
    logger.recordPacketTransfer(true, 1024, 5.0, true);  // Send
    logger.recordPacketTransfer(false, 512, 3.0, true);  // Receive
    
    auto metrics = logger.getOperationalMetrics();
    
    // Should have recorded the transfers
    EXPECT_GT(metrics.total_packets_sent, 0);
    EXPECT_GT(metrics.total_packets_received, 0);
    EXPECT_GT(metrics.total_bytes_sent, 0);
    EXPECT_GT(metrics.total_bytes_received, 0);
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerHealthStatus) {
    auto& logger = DiagnosticLogger::instance();
    
    auto health = logger.getHealthStatus();
    
    // Should have a valid health level
    EXPECT_GE(static_cast<int>(health.overall_health), 0);
    EXPECT_LE(static_cast<int>(health.overall_health), 2);
    
    // Should have a status message
    EXPECT_FALSE(health.status_message.empty());
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerOperationalReport) {
    auto& logger = DiagnosticLogger::instance();
    
    // This should not throw
    EXPECT_NO_THROW(logger.generateOperationalReport());
}

TEST_F(DiagnosticLoggerTest, DiagnosticLoggerMetricsExport) {
    auto& logger = DiagnosticLogger::instance();
    
    // This should not throw
    EXPECT_NO_THROW(logger.flushMetrics());
}

// ── Integration Tests ───────────────────────────────────────────────────────

TEST_F(DiagnosticLoggerTest, IntegrationSharedUtilitiesWithDiagnosticLogger) {
    auto& logger = DiagnosticLogger::instance();
    
    // Test error handling integration
    std::string error_msg = ErrorHandlingPatterns::translateVGREResult(VGREResult::ERR_NETWORK);
    std::string suggestion = ErrorHandlingPatterns::generateRecoverySuggestion(VGREResult::ERR_NETWORK, "test_context");
    
    EXPECT_FALSE(error_msg.empty());
    EXPECT_FALSE(suggestion.empty());
    
    // Test logging pattern integration
    std::string formatted_msg = LoggingPatterns::formatMessage("TestComponent", "TestOp", error_msg);
    EXPECT_FALSE(formatted_msg.empty());
    
    // Test platform detection integration
    auto platform = PlatformDetection::getCurrentPlatform();
    EXPECT_NE(platform.type, PlatformType::UNKNOWN);
}

TEST_F(DiagnosticLoggerTest, IntegrationConfigurationValidation) {
    auto config = ConfigurationManager::getClusterConfiguration();
    
    // This should not throw
    EXPECT_NO_THROW(ConfigurationManager::validateConfiguration(config));
    
    // Test with invalid configuration
    ClusterConfiguration invalid_config = config;
    invalid_config.handshake_timeout_sec = -1; // Invalid
    invalid_config.delta_sync_threshold = 2.0; // Invalid (> 1.0)
    
    // Should handle invalid configuration gracefully
    EXPECT_NO_THROW(ConfigurationManager::validateConfiguration(invalid_config));
}

// ── Performance Tests ───────────────────────────────────────────────────────

TEST_F(DiagnosticLoggerTest, PerformanceLoggingOverhead) {
    auto& logger = DiagnosticLogger::instance();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Log many events to test performance
    for (int i = 0; i < 1000; ++i) {
        logger.recordPacketTransfer(true, 1024, 1.0, true);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete in reasonable time (less than 100ms for 1000 operations)
    EXPECT_LT(duration.count(), 100000); // 100ms in microseconds
}

TEST_F(DiagnosticLoggerTest, PerformancePacketConstruction) {
    const char* test_payload = "test_data_for_performance_testing";
    size_t payload_size = strlen(test_payload);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Construct many packets to test performance
    for (int i = 0; i < 1000; ++i) {
        auto packet = PacketUtils::constructVSBPPacket(PacketType::TELEMETRY, 
                                                      test_payload, payload_size, i);
        // Prevent optimization from removing the loop
        volatile size_t size = packet.size();
        (void)size;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete in reasonable time (less than 50ms for 1000 operations)
    EXPECT_LT(duration.count(), 50000); // 50ms in microseconds
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}