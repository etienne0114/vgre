#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/input_validation.h"
#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_send_all;

std::vector<uint8_t> TCPClusterManager::constructPacket(PacketType type, const void* payload, size_t payloadLen) {
  return packet_handler_->constructPacket(type, payload, payloadLen);
}

VGREResult TCPClusterManager::waitForData(vgre_socket_t fd, int timeout_ms) {
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  vgre_pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  
  int result = vgre_poll(&pfd, 1, timeout_ms);
  
  if (result < 0) {
    // Poll error
    return VGREResult::ERR_IO;
  } else if (result == 0) {
    // Timeout
    return VGREResult::ERR_TIMEOUT;
  } else {
    // Data available
    if (pfd.revents & POLLIN) {
      return VGREResult::SUCCESS;
    } else if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      return VGREResult::ERR_IO;
    }
    return VGREResult::SUCCESS;
  }
}

std::string TCPClusterManager::hexDump(const uint8_t* data, size_t max_bytes) {
  if (!data || max_bytes == 0) {
    return "";
  }
  
  std::ostringstream oss;
  for (size_t i = 0; i < max_bytes; ++i) {
    if (i > 0 && i % 16 == 0) {
      oss << "\n";
    } else if (i > 0 && i % 8 == 0) {
      oss << "  ";
    } else if (i > 0) {
      oss << " ";
    }
    
    // Format as 2-digit hex
    char buf[4];
    snprintf(buf, sizeof(buf), "%02x", data[i]);
    oss << buf;
  }
  return oss.str();
}

VGREResult TCPClusterManager::send_packet(vgre_socket_t fd, PacketType type, const void *payload, size_t payloadLen, vgre::advanced::SecureChannel *sc) {
  auto start_time = std::chrono::steady_clock::now();
  
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    ErrorHandlingPatterns::logErrorWithContext(VGREResult::ERR_INVALID_VALUE, 
                                              "send_packet", "TCPCluster", 
                                              "Invalid socket descriptor");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // CRITICAL: Validate packet size to prevent buffer overflow
  if (common::InputValidator::validatePacketSize(payloadLen) != VGREResult::SUCCESS) {
    std::string error_msg = "Packet size validation failed: " + std::to_string(payloadLen);
    VGRE_LOG_ERROR("TCPCluster", error_msg);
    DiagnosticLogger::instance().logPacketCorruption(
      static_cast<const uint8_t*>(payload), payloadLen, 
      "Valid " + PacketUtils::packetTypeToString(type) + " packet");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Additional check for total packet size
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  if (common::InputValidator::validatePacketSize(totalLen) != VGREResult::SUCCESS) {
    std::string error_msg = "Total packet size validation failed: " + std::to_string(totalLen);
    VGRE_LOG_ERROR("TCPCluster", error_msg);
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Use packet handler for packet construction
  std::vector<uint8_t> staging = packet_handler_->constructPacket(type, payload, payloadLen);

  uint32_t priority = 1; // Default: LOW (Data/Bulk)
  if (type == PacketType::RESPONSE || type == PacketType::PARTITION_RESULT ||
      type == PacketType::TELEMETRY || type == PacketType::SECURE_HANDSHAKE ||
      type == PacketType::SECURE_HANDSHAKE_ACK || type == PacketType::ROTATE_KEY ||
      type == PacketType::CREDIT_REPORT || type == PacketType::COOP_BARRIER_SYNC ||
      type == PacketType::COOP_BARRIER_RESUME) {
    priority = 0; // HIGH (Control/Sync)
  }

  // Master side lookup — enqueue into per-client TSS2 queue
  bool foundMasterClient = false;
  {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
          if (client && client->socket_fd == fd) {
              std::lock_guard<std::mutex> tx_lock(client->tx_mutex);
              // B1: TSS2 queue depth cap — prevent unbounded memory growth when
              // a worker is slow or stalled. Configurable via VGRE_CLUSTER_MAX_QUEUE_DEPTH.
              static const size_t kMaxQueueDepth = []() -> size_t {
                  const char* env = vgre_get_config("VGRE_CLUSTER_MAX_QUEUE_DEPTH");
                  if (env) {
                      try { long v = std::stol(env); if (v > 0) return static_cast<size_t>(v); }
                      catch (...) {}
                  }
                  return 1024;
              }();
              if (client->high_priority_tx.size() + client->low_priority_tx.size() >= kMaxQueueDepth) {
                  VGRE_LOG_WARN("TCPCluster", "TX queue full for " + client->ip_address +
                      " (" + std::to_string(client->high_priority_tx.size() + client->low_priority_tx.size()) +
                      " packets) — dropping");
                  foundMasterClient = true; // Don't fall through to direct send
                  return VGREResult::ERR_BUSY;
              }
              ClientConnection::OutgoingPacket pkt;
              pkt.data = std::move(staging);
              pkt.priority = priority;
              if (priority == 0) { // HIGH priority
                  client->high_priority_tx.push_front(std::move(pkt));
              } else {
                  client->low_priority_tx.push_back(std::move(pkt));
              }
              foundMasterClient = true;
              break;
          }
      }
  }
  if (foundMasterClient) return VGREResult::SUCCESS;

  // Worker-side path: enqueue into worker-local TSS2 queues (drained by clientLoop)
  if (!is_master_ && fd == client_fd_) {
      std::lock_guard<std::mutex> lock(client_tx_mutex_);
      ClientConnection::OutgoingPacket pkt;
      pkt.data = std::move(staging);
      pkt.priority = priority;
      if (priority == 0) { // HIGH priority
          client_high_priority_tx_.push_front(std::move(pkt));
      } else {
          client_low_priority_tx_.push_back(std::move(pkt));
      }
      return VGREResult::SUCCESS;
  }

  // Direct send fallback (for non-queued scenarios like handshake packets)
  // This handles cases where the socket is valid but not yet in the client list
  VGREResult result;
  if (sc && sc->isInitialized()) {
      result = sc->sendSecure(fd, staging.data(), staging.size());
      if (result == VGREResult::SUCCESS) {
          global_packets_sent_++;
          global_bytes_sent_ += staging.size();
      }
  } else if (sc) {
      // SecureChannel exists but not yet initialized — only handshake
      // bootstrapping packets may be sent in plaintext.
      if (type != PacketType::SECURE_HANDSHAKE &&
          type != PacketType::SECURE_HANDSHAKE_ACK) {
          VGRE_LOG_ERROR("TCPCluster",
              "Plaintext send_packet rejected: secure channel not initialized for packet type " +
              std::to_string(static_cast<int>(type)));
          result = VGREResult::ERR_AUTH_FAILED;
      } else {
          bool success = vgre_send_all(fd, staging.data(), staging.size());
          if (success) {
              global_packets_sent_++;
              global_bytes_sent_ += staging.size();
              result = VGREResult::SUCCESS;
          } else {
              result = VGREResult::ERR_IO;
          }
      }
  } else {
      // No SecureChannel: security not requested for this connection.
      bool success = vgre_send_all(fd, staging.data(), staging.size());
      if (success) {
          global_packets_sent_++;
          global_bytes_sent_ += staging.size();
          result = VGREResult::SUCCESS;
      } else {
          result = VGREResult::ERR_IO;
      }
  }
  
  // Record diagnostic metrics
  double elapsed_ms = TimeUtils::calculateElapsedMs(start_time);
  DiagnosticLogger::instance().recordPacketTransfer(true, staging.size(), elapsed_ms, 
                                                   result == VGREResult::SUCCESS);
  
  if (result != VGREResult::SUCCESS) {
    // Find client IP for error logging
    std::string client_ip = "unknown";
    {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (const auto& client : clients_) {
        if (client && client->socket_fd == fd) {
          client_ip = client->ip_address;
          break;
        }
      }
    }
    DiagnosticLogger::instance().logNetworkError(result, "send_packet_" + 
                                                PacketUtils::packetTypeToString(type), client_ip);
  }
  
  return result;
}

VGREResult TCPClusterManager::send_packet_direct(vgre_socket_t fd, PacketType type, const void *payload, size_t payloadLen, vgre::advanced::SecureChannel *sc) {
  auto start_time = std::chrono::steady_clock::now();
  
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    ErrorHandlingPatterns::logErrorWithContext(VGREResult::ERR_INVALID_VALUE, 
                                              "send_packet_direct", "TCPCluster", 
                                              "Invalid socket descriptor");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Use unified packet construction
  std::vector<uint8_t> staging = constructPacket(type, payload, payloadLen);

  // Update statistics
  global_packets_sent_++;
  global_bytes_sent_ += staging.size();

  // Send directly (bypass queue)
  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size());
  } else if (sc) {
    // SecureChannel exists but not initialized — handshake packets only.
    if (type != PacketType::SECURE_HANDSHAKE &&
        type != PacketType::SECURE_HANDSHAKE_ACK) {
      VGRE_LOG_ERROR("TCPCluster",
          "Plaintext send_packet_direct rejected: secure channel not initialized for packet type " +
          std::to_string(static_cast<int>(type)));
      return VGREResult::ERR_AUTH_FAILED;
    }
    bool success = vgre_send_all(fd, staging.data(), staging.size());
    return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
  } else {
    // No SecureChannel: security not requested.
    bool success = vgre_send_all(fd, staging.data(), staging.size());
    return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
  }
}

int TCPClusterManager::recv_packet(vgre_socket_t fd, std::vector<uint8_t> &outBuffer, vgre::advanced::SecureChannel *sc) {
  // Delegate to PacketHandler
  int result = packet_handler_->recvPacket(fd, outBuffer, sc);
  
  // Update global statistics (PacketHandler updates its own stats)
  if (result > 0) {
    global_packets_received_++;
    if (sc && sc->isInitialized()) {
      global_bytes_received_ += static_cast<uint64_t>(result + sizeof(SecurePacketHeader));
    } else {
      global_bytes_received_ += static_cast<uint64_t>(result);
    }
  }
  
  return result;
}

void TCPClusterManager::flush_tx_queues(std::shared_ptr<ClientConnection> clientPtr) {
    if (!clientPtr) return;
    auto &client = *clientPtr;
    if (!client.active || client.socket_fd == VGRE_INVALID_SOCKET) return;
    
    std::lock_guard<std::mutex> tx_lock(client.tx_mutex);
    
    // 1. Drain High Priority (Sync/Control)
    // We insert at front for urgency, so we pop from back to maintain order within priority
    while (!client.high_priority_tx.empty()) {
        auto &pkt = client.high_priority_tx.back();
        bool success = false;
        if (client.secure_channel && client.secure_channel->isInitialized()) {
            success = (client.secure_channel->sendSecure(client.socket_fd, pkt.data.data(), pkt.data.size()) == VGREResult::SUCCESS);
        } else if (client.secure_channel) {
            // Queued packets are never handshakes — reject plaintext when security is expected.
            VGRE_LOG_ERROR("TCPCluster", "flush_tx_queues: dropping high-priority packet — secure channel not initialized");
            client.high_priority_tx.pop_back();
            continue;
        } else if (security_enabled_) {
            // Global security enabled but no secure channel assigned — reject.
            VGRE_LOG_ERROR("TCPCluster", "flush_tx_queues: dropping high-priority packet — security enabled but no channel");
            client.high_priority_tx.pop_back();
            continue;
        } else {
            success = vgre_send_all(client.socket_fd, pkt.data.data(), pkt.data.size());
        }
        
        if (success) {
            global_packets_sent_++;
            global_bytes_sent_ += pkt.data.size();
            client.packets_sent++;
            client.high_priority_tx.pop_back();
        } else {
            return; // Socket buffer full
        }
    }

    // 2. Drain Low Priority (Data/Bulk)
    while (!client.low_priority_tx.empty()) {
        auto &qItem = client.low_priority_tx.front();
        bool success = false;
        if (client.secure_channel && client.secure_channel->isInitialized()) {
            success = (client.secure_channel->sendSecure(client.socket_fd, qItem.data.data(), qItem.data.size()) == VGREResult::SUCCESS);
        } else if (client.secure_channel) {
            // Queued packets are never handshakes — reject plaintext when security is expected.
            VGRE_LOG_ERROR("TCPCluster", "flush_tx_queues: dropping low-priority packet — secure channel not initialized");
            client.low_priority_tx.pop_front();
            continue;
        } else if (security_enabled_) {
            VGRE_LOG_ERROR("TCPCluster", "flush_tx_queues: dropping low-priority packet — security enabled but no channel");
            client.low_priority_tx.pop_front();
            continue;
        } else {
            success = vgre_send_all(client.socket_fd, qItem.data.data(), qItem.data.size(), &enabled_);
        }
        
        // Trace disabled in production — uncomment for debugging only:
        // if (is_master_ && qItem.data.size() >= sizeof(VSBPHeader)) {
        //     VSBPHeader h; memcpy(&h, qItem.data.data(), sizeof(VSBPHeader));
        //     VGRE_LOG_DEBUG("TCPCluster", "[TX] Type=" + std::to_string((int)h.type) + " Size=" + std::to_string(qItem.data.size()));
        // }

        if (success) {
            global_packets_sent_++;
            global_bytes_sent_ += qItem.data.size();
            client.packets_sent++;
            client.low_priority_tx.pop_front();
        } else {
            return; // Socket buffer full
        }
    }
}

VGREResult TCPClusterManager::broadcastPacket(PacketType type, const void *payload,
                                      size_t payloadLen) {
  if (!is_master_) return VGREResult::ERR_NOT_SUPPORTED;
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  VGREResult result = VGREResult::SUCCESS;
  for (auto &client : clients_) {
    // B4: Only broadcast to fully-authenticated nodes when security is enabled.
    // Nodes that have TCP-connected but not yet completed the HMAC handshake
    // must not receive broadcast data before their identity is confirmed.
    if (!client || !client->active) continue;
    if (security_enabled_ && !client->security_established) continue;
    VGREResult send_result = send_packet(client->socket_fd, type, payload, payloadLen, client->secure_channel.get());
    if (send_result != VGREResult::SUCCESS) {
      result = send_result;
    }
  }
  return result;
}

} // namespace advanced
} // namespace vgre
