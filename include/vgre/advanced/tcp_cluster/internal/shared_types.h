#pragma once

#include <cstdint>
#include <string>

namespace vgre {
namespace advanced {

enum class PlatformType : uint8_t {
    WINDOWS = 1,
    LINUX = 2,
    MACOS = 3,
    UNKNOWN = 255
};

struct PlatformInfo {
    PlatformType type;
    std::string name;
    bool supports_shm_local;
    bool supports_rdma;
    bool supports_udp_broadcast;
    std::string socket_api;
};

} // namespace advanced
} // namespace vgre
