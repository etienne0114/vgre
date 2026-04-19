#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"
#include "vgre/common/input_validation.h"
#include "vgre/common/sockets.h"
#include <cstring>
#include <algorithm>

namespace vgre {
namespace advanced {

// Using vgre::common types and helpers
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;

namespace {

// Helper function for blocking send with timeout
static bool send_all(vgre_socket_t sock, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t sent = 0;
  constexpr int SEND_TIMEOUT_SECONDS = 5;
  auto start = std::chrono::steady_clock::now();
  
  while (sent < len) {
    // Check for timeout on blocking sends
    if (std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count() > SEND_TIMEOUT_SECONDS) {
      return false;
    }

    int n = send(sock, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && vgre_is_would_block(vgre_get_last_socket_error())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return false;
    }
    sent += n;
  }
  return true;
}

} // anonymous namespace

PacketHandler::PacketHandler() {}

// ── Unified Packet Construction ──────────────────────────────────────────
std::vector<uint8_t> PacketHandler::constructPacket(PacketType type, const void* payload, size_t payloadLen) {
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  std::vector<uint8_t> packet(totalLen);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(packet.data());
  header->magic = VSBP_MAGIC;
  header->version = VSBP_VERSION;
  header->type = static_cast<uint16_t>(type);
  header->sequence = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
  header->payloadSize = payloadLen;
  
  if (payload && payloadLen > 0) {
    std::memcpy(packet.data() + sizeof(VSBPHeader), payload, payloadLen);
  }
  
  return packet;
}

// ── Packet Sending (Queued) ──────────────────────────────────────────────
VGREResult PacketHandler::sendPacket(vgre_socket_t fd, PacketType type, 
                                     const void* payload, size_t payloadLen, 
                                     SecureChannel* sc) {
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // CRITICAL: Validate packet size to prevent buffer overflow
  if (common::InputValidator::validatePacketSize(payloadLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("PacketHandler", "Packet size validation failed: " + std::to_string(payloadLen));
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Additional check for total packet size
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  if (common::InputValidator::validatePacketSize(totalLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("PacketHandler", "Total packet size validation failed: " + std::to_string(totalLen));
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Validate and construct packet — actual transmission is handled by
  // TCPClusterManager::send_packet() which enqueues into the TSS2 priority
  // queues.  Do NOT count stats here; stats are updated in sendPacketDirect()
  // when bytes actually leave the socket.
  (void)constructPacket(type, payload, payloadLen);

  return VGREResult::SUCCESS;
}

// ── Packet Sending (Direct) ──────────────────────────────────────────────
VGREResult PacketHandler::sendPacketDirect(vgre_socket_t fd, PacketType type, 
                                           const void* payload, size_t payloadLen, 
                                           SecureChannel* sc) {
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Use unified packet construction
  std::vector<uint8_t> staging = constructPacket(type, payload, payloadLen);

  // Update statistics
  packets_sent_.fetch_add(1, std::memory_order_relaxed);
  bytes_sent_.fetch_add(staging.size(), std::memory_order_relaxed);

  // Send directly (bypass queue)
  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size());
  } else {
    // Fallback to direct send without encryption
    bool success = send_all(fd, staging.data(), staging.size());
    return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
  }
}

// ── Packet Receiving ─────────────────────────────────────────────────────
int PacketHandler::recvPacket(vgre_socket_t fd, std::vector<uint8_t>& outBuffer, 
                              SecureChannel* sc) {
  if (sc && sc->isInitialized()) {
    std::vector<uint8_t> payload;
    VGREResult r = sc->recvSecure(fd, payload);
    if (r == VGREResult::SUCCESS) {
      packets_received_.fetch_add(1, std::memory_order_relaxed);
      // Total wire size includes secure packet header
      bytes_received_.fetch_add(static_cast<uint64_t>(payload.size() + sizeof(SecurePacketHeader)), 
                                std::memory_order_relaxed);
      outBuffer.insert(outBuffer.end(), payload.begin(), payload.end());
      return static_cast<int>(payload.size());
    } else if (r == VGREResult::ERR_TIMEOUT) {
      return 0;
    }
    return -1; // Error
  } else {
    char temp[8192];
    int n = recv(fd, temp, sizeof(temp), 0);
    if (n > 0) {
      packets_received_.fetch_add(1, std::memory_order_relaxed);
      bytes_received_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
      outBuffer.insert(outBuffer.end(), temp, temp + n);
      return n;
    } else if (n == 0) {
      return -1; // Disconnected
    } else {
      if (vgre_is_would_block(vgre_get_last_socket_error())) return 0;
      return -1;
    }
  }
}

// ── VSBP Header Parsing ──────────────────────────────────────────────────
bool PacketHandler::parseVSBPHeader(const uint8_t* data, size_t dataSize, 
                                    VSBPHeader& header) {
  // Validate input
  if (!data || dataSize < sizeof(VSBPHeader)) {
    return false;
  }

  // Copy header
  std::memcpy(&header, data, sizeof(VSBPHeader));

  // Validate magic and version
  if (header.magic != VSBP_MAGIC || header.version != VSBP_VERSION) {
    VGRE_LOG_ERROR("PacketHandler", 
                   "VSBP protocol violation: invalid magic (0x" + 
                   std::to_string(header.magic) + ") or version (" + 
                   std::to_string(header.version) + ")");
    return false;
  }

  // Validate payload size doesn't exceed buffer
  if (dataSize < sizeof(VSBPHeader) + header.payloadSize) {
    VGRE_LOG_DEBUG("PacketHandler", 
                   "Incomplete packet: need " + 
                   std::to_string(sizeof(VSBPHeader) + header.payloadSize) + 
                   " bytes, have " + std::to_string(dataSize));
    return false;
  }

  return true;
}

} // namespace advanced
} // namespace vgre
