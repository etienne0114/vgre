#pragma once

#include "vgre/common/error_codes.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/shared_types.h"
#include "vgre/common/sockets.h"
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

class SecureChannel;

class NetworkUtils {
public:
    static VGREResult configureSocket(vgre::common::vgre_socket_t fd);
};

class PacketUtils {
public:
    static std::vector<uint8_t> constructVSBPPacket(PacketType type, const void* payload, size_t payload_size, uint32_t sequence);
    static bool validateVSBPHeader(const VSBPHeader& header);
    static std::string packetTypeToString(PacketType type);
};

class PlatformDetection {
public:
    static PlatformInfo getCurrentPlatform();
    static std::string getPlatformName(PlatformType type);
    static bool isPlatformCompatible(PlatformType local, PlatformType remote);
    static bool supportsCrossPlatformFeature(PlatformType platform, const std::string& feature);
};

} // namespace advanced
} // namespace vgre
