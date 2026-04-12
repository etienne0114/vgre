#ifndef VGRE_ADVANCED_TCP_CLUSTER_PACKET_HANDLER_H
#define VGRE_ADVANCED_TCP_CLUSTER_PACKET_HANDLER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/sockets.h"
#include "vgre/advanced/tcp_cluster.h"
#include <vector>
#include <atomic>
#include <cstdint>

namespace vgre {
namespace advanced {

// Forward declarations
class SecureChannel;

/**
 * PacketHandler - Handles VSBP packet construction, sending, and receiving
 * 
 * This module encapsulates all packet-level operations including:
 * - VSBP packet construction with headers
 * - Packet sending (queued and direct)
 * - Packet receiving and parsing
 * - Sequence number management
 */
class PacketHandler {
public:
  PacketHandler();
  ~PacketHandler() = default;

  // Packet construction
  std::vector<uint8_t> constructPacket(PacketType type, const void* payload, size_t payloadLen);

  // Packet sending
  VGREResult sendPacket(vgre_socket_t fd, PacketType type, const void* payload, 
                        size_t payloadLen, SecureChannel* sc = nullptr);
  
  VGREResult sendPacketDirect(vgre_socket_t fd, PacketType type, const void* payload,
                              size_t payloadLen, SecureChannel* sc = nullptr);

  // Packet receiving
  int recvPacket(vgre_socket_t fd, std::vector<uint8_t>& outBuffer, SecureChannel* sc = nullptr);

  // VSBP header parsing
  bool parseVSBPHeader(const uint8_t* data, size_t dataSize, VSBPHeader& header);

  // Statistics
  uint64_t getPacketsSent() const { return packets_sent_.load(); }
  uint64_t getPacketsReceived() const { return packets_received_.load(); }
  uint64_t getBytesSent() const { return bytes_sent_.load(); }
  uint64_t getBytesReceived() const { return bytes_received_.load(); }

private:
  std::atomic<uint32_t> sequence_counter_{1};
  std::atomic<uint64_t> packets_sent_{0};
  std::atomic<uint64_t> packets_received_{0};
  std::atomic<uint64_t> bytes_sent_{0};
  std::atomic<uint64_t> bytes_received_{0};
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_PACKET_HANDLER_H
