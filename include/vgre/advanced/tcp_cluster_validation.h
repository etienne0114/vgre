#pragma once

#include "vgre/common/error_codes.h"
#include <cstdint>
#include <cstddef>
#include <string>

// ── Cross-Platform Memory Alignment Validation ──────────────────────────────
// Helpers for consistent struct packing and memory layout across platforms.
// Supports mesh topology where any platform (Windows, Linux, macOS) can be master or worker.

namespace vgre {
namespace advanced {

// Forward declarations
class TCPClusterManager;
struct ClientConnection;

namespace detail {
  // Platform-independent memory alignment validation
  template<typename T>
  constexpr bool validate_struct_size(size_t expected_size) {
    return sizeof(T) == expected_size;
  }
  
  // Validate struct alignment requirements
  template<typename T>
  constexpr bool validate_struct_alignment(size_t expected_alignment) {
    return alignof(T) <= expected_alignment;
  }
  
  // Cross-platform endianness validation
  inline bool validate_endianness() {
    constexpr uint32_t test_value = 0x12345678;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&test_value);
    return bytes[0] == 0x78; // Little-endian check
  }
  
  // Mesh topology platform detection
  enum class PlatformType : uint8_t {
    WINDOWS = 1,
    LINUX = 2,
    MACOS = 3,
    UNKNOWN = 255
  };
  
  constexpr PlatformType detect_platform() {
#ifdef _WIN32
    return PlatformType::WINDOWS;
#elif defined(__linux__)
    return PlatformType::LINUX;
#elif defined(__APPLE__)
    return PlatformType::MACOS;
#else
    return PlatformType::UNKNOWN;
#endif
  }
} // namespace detail

  /**
   * Validates struct sizes and memory layout consistency across platforms
   * to prevent ACCESS_VIOLATION crashes in cross-platform deployments.
   */
  class MemoryAlignmentValidator {
  public:
    static VGREResult validateAllPacketStructures();
    
  private:
    static bool validateTelemetryPacket();
    static bool validateRemoteCommandPacket();
    static bool validateResponsePacket();
    static bool validateCapabilityPacket();
    static bool validateSecureHandshakePacket();
    static bool validatePartitionDispatchPacket();
    static bool validatePartitionResultPacket();
    static bool validateCreditReportPacket();
    static bool validateCollectiveOpPacket();
  };

  /**
   * Manages cross-platform mesh topology where any platform can be master or worker.
   */
  class MeshTopologyManager {
  public:
    static VGREResult initializeMeshTopology(TCPClusterManager* cluster);
    static VGREResult validateCrossPlatformSupport();
    static std::string getPlatformName(detail::PlatformType platform);
  };

} // namespace advanced
} // namespace vgre
