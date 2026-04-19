#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/workload_partitioner.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/input_validation.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/runtime/vector_engine.h"
#include "vgre/common/sockets.h"

// SIMD intrinsics headers
#if defined(__AVX2__)
#include <immintrin.h>  // AVX2
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>  // SSE2
#endif

// System Headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>

// Socket headers (for sockaddr_in and other socket structures)
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

namespace vgre {
namespace advanced {

// Using vgre::common types and helpers
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_setsockopt;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_get_last_socket_error;

namespace {
// Helper function for sending all bytes
static bool send_all(vgre_socket_t sock, const void *buf, size_t len, const std::atomic<bool>* enabled = nullptr) {
  const char *p = static_cast<const char *>(buf);
  size_t sent = 0;
  auto start = std::chrono::steady_clock::now();
  while (sent < len) {
    if (enabled && !enabled->load()) return false;
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() > 5) {
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
} // namespace

// ── Unified Packet Construction ──────────────────────────────────────────
std::vector<uint8_t> TCPClusterManager::constructPacket(PacketType type, const void* payload, size_t payloadLen) {
  return packet_handler_->constructPacket(type, payload, payloadLen);
}

// ── Delta-Sync Logic Extraction ──────────────────────────────────────────
VGREResult TCPClusterManager::syncPointerToWorker(void* ptr, uint64_t handle, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->syncPointerToWorker(ptr, handle, client);
}

VGREResult TCPClusterManager::sendDeltaSync(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSync(ptr, handle, dirtyRanges, client);
}

VGREResult TCPClusterManager::sendDeltaSyncWithRetry(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncWithRetry(ptr, handle, dirtyRanges, client);
}

VGREResult TCPClusterManager::sendDeltaSyncSHM(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncSHM(ptr, handle, dirtyRanges, client);
}

VGREResult TCPClusterManager::sendDeltaSyncTCP(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncTCP(ptr, handle, dirtyRanges, client);
}

VGREResult TCPClusterManager::sendFullSync(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSync(ptr, handle, size, client);
}

VGREResult TCPClusterManager::sendFullSyncSHM(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSyncSHM(ptr, handle, size, client);
}

VGREResult TCPClusterManager::sendFullSyncTCP(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSyncTCP(ptr, handle, size, client);
}

// ── Argument Serialization Logic Extraction ──────────────────────────────
VGREResult TCPClusterManager::streamArgumentsToWorker(void** args, int num_args, uint64_t kernel_id, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->streamArgumentsToWorker(args, num_args, kernel_id, client);
}

VGREResult TCPClusterManager::sendStructArg(void* arg, int arg_index, uint64_t kernel_id, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendStructArg(arg, arg_index, kernel_id, client);
}

VGREResult TCPClusterManager::sendPointerArg(void* arg, int arg_index, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendPointerArg(arg, arg_index, client);
}

VGREResult TCPClusterManager::sendScalarArg(void* arg, int arg_index, ArgType type, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendScalarArg(arg, arg_index, type, client);
}

// ── Blocking I/O Helper ───────────────────────────────────────────────────
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

// ── Diagnostic Helper ─────────────────────────────────────────────────────
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
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // CRITICAL: Validate packet size to prevent buffer overflow
  if (common::InputValidator::validatePacketSize(payloadLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Packet size validation failed: " + std::to_string(payloadLen));
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Additional check for total packet size
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  if (common::InputValidator::validatePacketSize(totalLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Total packet size validation failed: " + std::to_string(totalLen));
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Use unified packet construction
  std::vector<uint8_t> staging = constructPacket(type, payload, payloadLen);

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
                  const char* env = std::getenv("VGRE_CLUSTER_MAX_QUEUE_DEPTH");
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
  if (sc && sc->isInitialized()) {
      VGREResult result = sc->sendSecure(fd, staging.data(), staging.size());
      if (result == VGREResult::SUCCESS) {
          global_packets_sent_++;
          global_bytes_sent_ += staging.size();
      }
      return result;
  } else {
      // Fallback to direct send without encryption
      bool success = send_all(fd, staging.data(), staging.size());
      if (success) {
          global_packets_sent_++;
          global_bytes_sent_ += staging.size();
          return VGREResult::SUCCESS;
      }
      return VGREResult::ERR_IO;
  }
}

VGREResult TCPClusterManager::send_packet_direct(vgre_socket_t fd, PacketType type, const void *payload, size_t payloadLen, vgre::advanced::SecureChannel *sc) {
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
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
  } else {
    // Fallback to direct send without encryption
    bool success = send_all(fd, staging.data(), staging.size());
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
        if (client.secureChannel && client.secureChannel->isInitialized()) {
            success = (client.secureChannel->sendSecure(client.socket_fd, pkt.data.data(), pkt.data.size()) == VGREResult::SUCCESS);
        } else {
            success = send_all(client.socket_fd, pkt.data.data(), pkt.data.size());
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
        if (client.secureChannel && client.secureChannel->isInitialized()) {
            success = (client.secureChannel->sendSecure(client.socket_fd, qItem.data.data(), qItem.data.size()) == VGREResult::SUCCESS);
        } else {
            success = send_all(client.socket_fd, qItem.data.data(), qItem.data.size(), &enabled_);
        }
        
        // Trace disabled in production — uncomment for debugging only:
        // if (is_master_ && qItem.data.size() >= sizeof(VSBPHeader)) {
        //     VSBPHeader h; std::memcpy(&h, qItem.data.data(), sizeof(VSBPHeader));
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

// Constructor, destructor, initialize, shutdown → tcp_cluster/tcp_cluster_manager.cpp
// serverLoop                                  → tcp_cluster/server_loop.cpp
// clientLoop, processClientStagingBuffer      → tcp_cluster/client_loop.cpp

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
    VGREResult send_result = send_packet(client->socket_fd, type, payload, payloadLen, client->secureChannel.get());
    if (send_result != VGREResult::SUCCESS) {
      result = send_result;
    }
  }
  return result;
}







VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  return dispatch_manager_->launchRemoteKernel(
      worker_idx, kernel_id, grid_dim, block_dim, args, num_args, shared_mem);
}

void TCPClusterManager::broadcastKernelRegistration(uint64_t kernel_id,
                                                   const std::string &name,
                                                   const std::string &source) {
  dispatch_manager_->broadcastKernelRegistration(kernel_id, name, source);
}

int TCPClusterManager::getFirstActiveWorker() const {
  if (!is_master_) {
    return -1;
  }
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  for (size_t i = 0; i < clients_.size(); ++i) {
    if (clients_[i] && clients_[i]->active) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void TCPClusterManager::handleRemoteCommand(const RemoteCommandPacket &pkt) {
  dispatch_manager_->handleRemoteCommand(pkt);
}

void TCPClusterManager::broadcastLocalTelemetry(
    const vgre_telemetry_t &telemetry) {
  if (enabled_ && !is_master_) {
    std::lock_guard<std::mutex> lock(client_mutex_);
    client_telemetry_buffer_ = telemetry;
    send_packet(client_fd_, PacketType::TELEMETRY, &telemetry, sizeof(vgre_telemetry_t), client_secure_channel_.get());
  }
}

void TCPClusterManager::aggregateRemoteTelemetry(
    vgre_telemetry_t &outCombined) {
  if (enabled_ && is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (const auto &client : clients_) {
      if (!client || !client->active)
        continue;
      outCombined.gflops += client->last_telemetry.gflops;
      outCombined.memory_bandwidth_gbps +=
          client->last_telemetry.memory_bandwidth_gbps;
      outCombined.memory_used_bytes += client->last_telemetry.memory_used_bytes;
      outCombined.active_kernels += client->last_telemetry.active_kernels;
      outCombined.active_threads += client->last_telemetry.active_threads;
    }
  }
}

void TCPClusterManager::getConnectedNodes(std::vector<TCPClusterManager::ClusterNodeInfo> &outNodes) const {
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  outNodes.clear();
  for (const auto &c : clients_) {
    // Only report nodes that are alive and have delivered real hardware info via
    // CAPABILITY. Nodes still handshaking or reconnecting show cpu_cores=0 until
    // CAPABILITY arrives — exclude them to keep dashboard data accurate.
    if (!c || !c->active) continue;
    if (!c->capability_received && !c->is_local) continue;
    TCPClusterManager::ClusterNodeInfo info;
    info.ip_address = c->ip_address;
    info.port = c->port;
    info.last_telemetry = c->last_telemetry;
    info.active = c->active;
    info.cpu_cores = c->cpu_cores;
    info.cpu_memory = c->cpu_memory;
    info.has_igpu = c->has_igpu;
    std::memcpy(info.igpu_name, c->igpu_name, sizeof(info.igpu_name));
    info.security_established = c->security_established;
    info.is_authenticating = c->is_authenticating;
    outNodes.push_back(std::move(info));
  }
}

// ── Phase 5: Security ─────────────────────────────────────────────────────

VGREResult TCPClusterManager::enableSecurity(bool enabled) {
  return security_manager_->enableSecurity(enabled);
}

SessionInfo TCPClusterManager::getSecurityInfo() const {
  return security_manager_->getSecurityInfo();
}

VGREResult TCPClusterManager::performSecureHandshake(std::shared_ptr<ClientConnection> clientPtr) {
  return security_manager_->performServerHandshake(clientPtr);
}

VGREResult TCPClusterManager::performClientSecureHandshake() {
  return security_manager_->performClientHandshake();
}

// ── Phase 5: Partitioned Kernel Dispatch ──────────────────────────────────

VGREResult TCPClusterManager::launchPartitionedKernel(
    uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args,
    size_t shared_mem) {
  return dispatch_manager_->launchPartitionedKernel(
      kernel_id, grid_dim, block_dim, args, num_args, shared_mem);
}

VGREResult TCPClusterManager::collectPartitionResults(
    uint64_t kernel_id, uint32_t total_partitions, int timeout_ms) {
  return dispatch_manager_->collectPartitionResults(
      kernel_id, total_partitions, timeout_ms);
}

VGREResult TCPClusterManager::waitForRemoteResult(uint64_t kernel_id, int timeout_ms) {
  return dispatch_manager_->waitForRemoteResult(kernel_id, timeout_ms);
}

void TCPClusterManager::handlePartitionDispatch(
    const PartitionDispatchPacket &pkt) {
  dispatch_manager_->handlePartitionDispatch(pkt);
}

void TCPClusterManager::parseProactiveNodes() {
    const char* env = std::getenv("VGRE_CLUSTER_NODES");
    if (!env) return;

    proactive_worker_addresses_.clear();
    std::string nodes(env);
    size_t start = 0;
    size_t end = nodes.find(',');
    while (end != std::string::npos) {
        proactive_worker_addresses_.push_back(nodes.substr(start, end - start));
        start = end + 1;
        end = nodes.find(',', start);
    }
    proactive_worker_addresses_.push_back(nodes.substr(start));
    
    for (auto& addr : proactive_worker_addresses_) {
        VGRE_LOG_INFO("TCPCluster", "Master: Registered proactive connection target: " + addr);
    }
}

void TCPClusterManager::syncToIPC() {
  if (!is_master_ || !enabled_) return;

  std::vector<vgre_cluster_node_t> ipcNodes;
  {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);

      for (const auto& c : clients_) {
          // Only nodes that are alive AND have delivered a CAPABILITY packet are
          // shown in the dashboard. Nodes in the handshake phase or that reconnected
          // before CAPABILITY arrives would otherwise show cpu_cores=0 / memory=0.
          // Local (SHM) nodes bypass CAPABILITY and are always safe to display.
          // Dead entries stay in clients_ so launchRemoteKernel() index arithmetic
          // stays valid.
          if (!c || !c->active) continue;
          if (!c->capability_received && !c->is_local) continue;

          vgre_cluster_node_t node{};
          std::strncpy(node.address, c->ip_address.c_str(), sizeof(node.address) - 1);
          node.port = c->port;
          node.cpu_cores = c->cpu_cores;
          node.memory_bytes = c->cpu_memory;
          node.latency_ms = c->last_telemetry.avg_kernel_latency_ms;

          node.available = 1; // active node (secure or plain)

          std::strncpy(node.igpu_name, c->igpu_name, sizeof(node.igpu_name) - 1);
          ipcNodes.push_back(node);
      }
  }

  // Push updated list to Shared Memory for dashboard visibility
  vgre::advanced::IPCManager::instance().updateClusterNodes(ipcNodes);
}

// ══════════════════════════════════════════════════════════════════════════════
// Phase 1: Missing Method Implementations
// ══════════════════════════════════════════════════════════════════════════════

void TCPClusterManager::reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id) {
  // Validation: only workers can report compute metrics
  if (!enabled_ || is_master_) {
    return;
  }

  // Create credit report packet
  CreditReportPacket packet;
  packet.compute_seconds = seconds;
  packet.cpu_cores = cores;
  packet.kernel_id = kernel_id;
  packet.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();

  // Send to master
  if (client_fd_ != VGRE_INVALID_SOCKET) {
    send_packet(client_fd_, PacketType::CREDIT_REPORT, &packet, sizeof(packet), client_secure_channel_.get());
    VGRE_LOG_DEBUG("TCPCluster", "Reported compute: " + std::to_string(seconds) + 
                   "s on " + std::to_string(cores) + " cores for kernel " + 
                   std::to_string(kernel_id));
  }
}

VGREResult TCPClusterManager::allReduce(void* ptr, size_t count, int datatype) {
  return collective_ops_manager_->allReduce(ptr, count, datatype);
}

} // namespace advanced
} // namespace vgre
