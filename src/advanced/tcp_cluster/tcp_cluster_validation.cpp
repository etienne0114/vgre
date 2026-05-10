#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster_validation.h"
#include "vgre/advanced/tcp_cluster_protocol.h"
#include "vgre/common/logger.h"
#include "vgre/advanced/secure_channel.h"
#include <string>

namespace vgre {
namespace advanced {

// ── MemoryAlignmentValidator Implementation ──────────────────────────────────

VGREResult MemoryAlignmentValidator::validateAllPacketStructures() {
  VGRE_LOG_DEBUG("TCPCluster", "Validating cross-platform packet structure alignment");
  
  bool all_valid = true;
  
  // Validate platform-dependent packet structures at runtime
  all_valid &= validateTelemetryPacket();
  all_valid &= validateRemoteCommandPacket();
  all_valid &= validateResponsePacket();
  all_valid &= validateCapabilityPacket();
  all_valid &= validateSecureHandshakePacket();
  all_valid &= validatePartitionDispatchPacket();
  all_valid &= validatePartitionResultPacket();
  all_valid &= validateCreditReportPacket();
  all_valid &= validateCollectiveOpPacket();
  
  if (!all_valid) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Cross-platform memory alignment validation failed - "
      "this may cause ACCESS_VIOLATION crashes on Windows workers");
    return VGREResult::ERR_NOT_SUPPORTED;
  }
  
  VGRE_LOG_INFO("TCPCluster", "Cross-platform memory alignment validation passed");
  return VGREResult::SUCCESS;
}

bool MemoryAlignmentValidator::validateTelemetryPacket() {
  size_t expected_min = 1024;  // vgre_telemetry_t should be at least 1KB
  size_t actual = sizeof(TelemetryPacket);
  
  if (actual < expected_min) {
    VGRE_LOG_ERROR("TCPCluster", 
      "TelemetryPacket size validation failed: expected >= " + 
      std::to_string(expected_min) + ", got " + std::to_string(actual));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "TelemetryPacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validateRemoteCommandPacket() {
  size_t actual = sizeof(RemoteCommandPacket);
  
  // Check that size_t and int are reasonable sizes
  if (sizeof(size_t) != 8 && sizeof(size_t) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Unsupported size_t size: " + std::to_string(sizeof(size_t)) + 
      " bytes - must be 4 or 8 bytes");
    return false;
  }
  
  if (sizeof(int) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Unsupported int size: " + std::to_string(sizeof(int)) + 
      " bytes - must be 4 bytes");
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "RemoteCommandPacket size: " + std::to_string(actual) + " bytes " +
    "(size_t=" + std::to_string(sizeof(size_t)) + ", int=" + std::to_string(sizeof(int)) + ")");
  return true;
}

bool MemoryAlignmentValidator::validateResponsePacket() {
  size_t actual = sizeof(ResponsePacket);
  
  // VGREResult should be 4 bytes (int32_t)
  if (sizeof(VGREResult) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "VGREResult size validation failed: expected 4 bytes, got " + 
      std::to_string(sizeof(VGREResult)));
    return false;
  }
  
  size_t expected = 8 + 4;  // uint64_t + VGREResult
  if (actual < expected) {
    VGRE_LOG_ERROR("TCPCluster", 
      "ResponsePacket size validation failed: expected >= " + 
      std::to_string(expected) + ", got " + std::to_string(actual));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "ResponsePacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validateCapabilityPacket() {
  size_t actual = sizeof(CapabilityPacket);
  
  // Check platform-dependent types
  if (sizeof(int) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CapabilityPacket int size validation failed: expected 4 bytes, got " + 
      std::to_string(sizeof(int)));
    return false;
  }
  
  if (sizeof(bool) != 1) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CapabilityPacket bool size validation failed: expected 1 byte, got " + 
      std::to_string(sizeof(bool)));
    return false;
  }
  
  // Minimum expected size: 5 ints + 1 uint64_t + 1 bool + 192 chars
  size_t expected_min = 5*4 + 8 + 1 + 192;  // ~221 bytes minimum
  if (actual < expected_min) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CapabilityPacket size validation failed: expected >= " + 
      std::to_string(expected_min) + ", got " + std::to_string(actual));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "CapabilityPacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validateSecureHandshakePacket() {
  size_t actual = sizeof(SecureHandshakePacket);
  
  // Should be nonce + key_verification
  size_t expected = crypto::kNonceLen + crypto::kSHA256DigestLen;
  if (actual != expected) {
    VGRE_LOG_ERROR("TCPCluster", 
      "SecureHandshakePacket size validation failed: expected " + 
      std::to_string(expected) + ", got " + std::to_string(actual));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "SecureHandshakePacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validatePartitionDispatchPacket() {
  size_t actual = sizeof(PartitionDispatchPacket);
  
  // Check int size consistency
  if (sizeof(int) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "PartitionDispatchPacket int size validation failed: expected 4 bytes, got " + 
      std::to_string(sizeof(int)));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "PartitionDispatchPacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validatePartitionResultPacket() {
  size_t actual = sizeof(PartitionResultPacket);
  
  // Check double alignment
  if (sizeof(double) != 8) {
    VGRE_LOG_ERROR("TCPCluster", 
      "PartitionResultPacket double size validation failed: expected 8 bytes, got " + 
      std::to_string(sizeof(double)));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "PartitionResultPacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validateCreditReportPacket() {
  size_t actual = sizeof(CreditReportPacket);
  
  // Check double and int sizes
  if (sizeof(double) != 8) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CreditReportPacket double size validation failed: expected 8 bytes, got " + 
      std::to_string(sizeof(double)));
    return false;
  }
  
  if (sizeof(int) != 4) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CreditReportPacket int size validation failed: expected 4 bytes, got " + 
      std::to_string(sizeof(int)));
    return false;
  }
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "CreditReportPacket size: " + std::to_string(actual) + " bytes");
  return true;
}

bool MemoryAlignmentValidator::validateCollectiveOpPacket() {
  size_t actual = sizeof(CollectiveOpPacket);
  if (actual != 24) {
    VGRE_LOG_ERROR("TCPCluster", 
      "CollectiveOpPacket size validation failed: expected 24 bytes, got " + 
      std::to_string(actual));
    return false;
  }
  return true;
}

// ── MeshTopologyManager Implementation ───────────────────────────────────────

VGREResult MeshTopologyManager::initializeMeshTopology(TCPClusterManager* cluster) {
  VGRE_LOG_INFO("TCPCluster", "Initializing cross-platform mesh topology");
  
  // Detect local platform
  detail::PlatformType local_platform = detail::detect_platform();
  std::string platform_name = getPlatformName(local_platform);
  
  VGRE_LOG_INFO("TCPCluster", 
    "Local platform: " + platform_name + " (can be master or worker)");
  
  // Validate cross-platform compatibility
  VGREResult validation_result = validateCrossPlatformSupport();
  if (validation_result != VGREResult::SUCCESS) {
    return validation_result;
  }
  
  // Initialize mesh peer discovery
  cluster->parseMeshPeers();
  
  VGRE_LOG_INFO("TCPCluster", "Mesh topology initialized successfully");
  return VGREResult::SUCCESS;
}

VGREResult MeshTopologyManager::validateCrossPlatformSupport() {
  // Ensure all required cross-platform features are available
  bool all_supported = true;
  
  // Check memory alignment
  if (!detail::validate_endianness()) {
    VGRE_LOG_ERROR("TCPCluster", "Platform endianness not supported for mesh topology");
    all_supported = false;
  }
  
  // Check struct packing
  if (sizeof(VSBPHeader) != 24) {
    VGRE_LOG_ERROR("TCPCluster", 
      "VSBPHeader size mismatch: expected 24 bytes, got " + 
      std::to_string(sizeof(VSBPHeader)));
    all_supported = false;
  }
  
  // Check platform detection
  detail::PlatformType platform = detail::detect_platform();
  if (platform == detail::PlatformType::UNKNOWN) {
    VGRE_LOG_ERROR("TCPCluster", "Unknown platform - mesh topology not supported");
    all_supported = false;
  }
  
  return all_supported ? VGREResult::SUCCESS : VGREResult::ERR_NOT_SUPPORTED;
}

std::string MeshTopologyManager::getPlatformName(detail::PlatformType platform) {
  switch (platform) {
    case detail::PlatformType::WINDOWS: return "Windows";
    case detail::PlatformType::LINUX:   return "Linux";
    case detail::PlatformType::MACOS:   return "macOS";
    default:                             return "Unknown";
  }
}

} // namespace advanced
} // namespace vgre
